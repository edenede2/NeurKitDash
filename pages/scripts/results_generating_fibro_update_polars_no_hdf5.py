"""
Updated Results Generating Script for Fibro Project
Compatible with new Parquet-based data storage (no HDF5)
"""
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from time import perf_counter

import neurokit2 as nk
import numpy as np
import pandas as pd
import polars as pl

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pages.data_manager import DataManager, atomic_write
from pages.paths import DATA_PATH, TEMPLATES_PATH

logger = logging.getLogger(__name__)


def main(project):
    started = perf_counter()
    data_manager = DataManager(project)
    with open(TEMPLATES_PATH / f"{project}_template.json", encoding="utf-8") as file:
        sampling_rate = int(json.load(file)["parameters"]["sampling_rate"])

    if not data_manager.has_signal_data() or not (DATA_PATH / f"{project}_mindware_events.parquet").exists():
        logger.error("Signal and event data are required for project %s", project)
        return 1

    queue_path = DATA_PATH / f"{project}_subjects_to_run.csv"
    import_events = pl.scan_parquet(data_manager.events_parquet_path)
    if queue_path.exists():
        queue = pl.read_csv(queue_path, schema_overrides={"subject_id": pl.String})
    else:
        queue = (import_events.select("subject_id", "event_name").unique()
                 .with_columns(project=pl.lit(project), status=pl.lit("New")).collect())
        atomic_write(queue_path, queue.write_csv)

    keys = ["subject_id", "event_name"]
    queued = queue.filter((pl.col("project") == project) & (pl.col("status") == "New")).select(keys).unique()
    if queued.is_empty():
        logger.info("No pending events for project %s", project)
        return 0

    events_by_subject = defaultdict(list)
    events = import_events.join(queued.lazy(), on=keys, how="semi").collect()
    for event in events.iter_rows(named=True):
        events_by_subject[event["subject_id"]].append(event)
    logger.info("Analyzing %d queued events in project %s (%d subjects, %d Hz)",
                queued.height, project, len(events_by_subject), sampling_rate)

    def closest_indices(time, values):
        right = np.clip(np.searchsorted(time, values), 1, len(time) - 1)
        left = right - 1
        return np.where(values - time[left] < time[right] - values, left, right)

    def compute_eda_metrics_in_window(gsc_signal, sub_id, start_t, end_t):
        """Calculate EDA for one event, using the same independent sliding windows."""
        try:
            if len(gsc_signal) == 0:
                raise ValueError("No EDA samples in this event")
            # Process EDA signal with NeuroKit2
            eda_data, info = nk.eda_process(gsc_signal, sampling_rate=sampling_rate)

            # Calculate overall EDA metrics
            overall_scl_mean = float(np.mean(eda_data['EDA_Tonic']))
            overall_scr_mean = float(np.mean(eda_data['EDA_Phasic']))
            overall_scr_std = float(np.std(eda_data['EDA_Phasic']))
            overall_scr_count = float(np.sum(eda_data['SCR_Peaks']))

            # Sliding window analysis: 2-minute windows with 1-minute stride
            window_duration = 120.0  # 2 minutes in seconds
            stride = 60.0  # 1 minute stride

            window_size = int(window_duration * sampling_rate)  # samples in 2 minutes
            stride_size = int(stride * sampling_rate)  # samples in 1 minute

            # Lists to store sliding window metrics
            sliding_scl_means = []
            sliding_scr_means = []
            sliding_scr_stds = []
            sliding_scr_counts = []

            # Perform sliding window analysis
            total_samples = len(gsc_signal)

            if total_samples >= window_size:
                for i in range(0, total_samples - window_size + 1, stride_size):
                    window_signal = gsc_signal[i:i + window_size]

                    try:
                        # Process this window
                        window_eda, _ = nk.eda_process(window_signal, sampling_rate=sampling_rate)

                        # Calculate metrics for this window
                        sliding_scl_means.append(float(np.mean(window_eda['EDA_Tonic'])))
                        sliding_scr_means.append(float(np.mean(window_eda['EDA_Phasic'])))
                        sliding_scr_stds.append(float(np.std(window_eda['EDA_Phasic'])))
                        sliding_scr_counts.append(float(np.sum(window_eda['SCR_Peaks'])))
                    except Exception:
                        logger.debug("EDA window failed for %s at sample %d", sub_id, i, exc_info=True)
                        # If a window fails, append NaNs
                        sliding_scl_means.append(np.nan)
                        sliding_scr_means.append(np.nan)
                        sliding_scr_stds.append(np.nan)
                        sliding_scr_counts.append(np.nan)

            # Calculate statistics of sliding window metrics
            if len(sliding_scl_means) > 0:
                scl_mean_sliding_mean = float(np.nanmean(sliding_scl_means))
                scl_mean_sliding_std = float(np.nanstd(sliding_scl_means))
                scr_mean_sliding_mean = float(np.nanmean(sliding_scr_means))
                scr_mean_sliding_std = float(np.nanstd(sliding_scr_means))
                scr_std_sliding_mean = float(np.nanmean(sliding_scr_stds))
                scr_std_sliding_std = float(np.nanstd(sliding_scr_stds))
                scr_count_sliding_mean = float(np.nanmean(sliding_scr_counts))
                scr_count_sliding_std = float(np.nanstd(sliding_scr_counts))
            else:
                scl_mean_sliding_mean = np.nan
                scl_mean_sliding_std = np.nan
                scr_mean_sliding_mean = np.nan
                scr_mean_sliding_std = np.nan
                scr_std_sliding_mean = np.nan
                scr_std_sliding_std = np.nan
                scr_count_sliding_mean = np.nan
                scr_count_sliding_std = np.nan

            # Combine overall and sliding window metrics
            gsc_data = pd.DataFrame({
                'SCL_mean': [overall_scl_mean],
                'SCR_mean': [overall_scr_mean],
                'SCR_std': [overall_scr_std],
                'SCR_count': [overall_scr_count],
                'SCL_mean_sliding_mean': [scl_mean_sliding_mean],
                'SCL_mean_sliding_std': [scl_mean_sliding_std],
                'SCR_mean_sliding_mean': [scr_mean_sliding_mean],
                'SCR_mean_sliding_std': [scr_mean_sliding_std],
                'SCR_std_sliding_mean': [scr_std_sliding_mean],
                'SCR_std_sliding_std': [scr_std_sliding_std],
                'SCR_count_sliding_mean': [scr_count_sliding_mean],
                'SCR_count_sliding_std': [scr_count_sliding_std]
            })

            return gsc_data

        except Exception as e:
            logger.warning("EDA failed for %s [%.3f, %.3f]: %s", sub_id, start_t, end_t, e)
            return pd.DataFrame({
                'SCL_mean': [np.nan],
                'SCR_mean': [np.nan],
                'SCR_std': [np.nan],
                'SCR_count': [np.nan],
                'SCL_mean_sliding_mean': [np.nan],
                'SCL_mean_sliding_std': [np.nan],
                'SCR_mean_sliding_mean': [np.nan],
                'SCR_mean_sliding_std': [np.nan],
                'SCR_std_sliding_mean': [np.nan],
                'SCR_std_sliding_std': [np.nan],
                'SCR_count_sliding_mean': [np.nan],
                'SCR_count_sliding_std': [np.nan]
            })

    metric_frames = []
    completed = []
    for sub_id, events_data in events_by_subject.items():
        subject_start = perf_counter()
        signal = data_manager.scan_subject_signal(sub_id)
        try:
            schema = signal.collect_schema()
            columns = ["time", "gsc"] if "gsc" in schema else ["time"]
            samples = signal.select(columns).collect()
            if samples.is_empty():
                logger.warning("No samples for %s", sub_id)
                continue
            time_np = samples["time"].to_numpy()
            gsc = samples["gsc"].to_numpy() if "gsc" in samples.columns else None
            attributes = data_manager.load_subject_attributes(sub_id, include_quality=False) or {}
        except Exception:
            logger.exception("Could not load signal and attributes for %s", sub_id)
            continue

        sorted_time = np.all(np.isfinite(time_np)) and np.all(time_np[1:] >= time_np[:-1])
        def event_gsc(start, end):
            if sorted_time:
                left = np.searchsorted(time_np, start, side="left")
                right = np.searchsorted(time_np, end, side="right")
                return gsc[left:right]
            return gsc[(time_np >= start) & (time_np <= end)]

        rpeaks = [index for index in attributes.get("rpeaks", []) if 0 <= index < len(time_np)]
        rpeaks_timing = time_np[rpeaks]
        peak_positions = {index: position for position, index in reversed(list(enumerate(rpeaks)))}
        yellow_rpeaks_mapped = [peak_positions[yellow] for yellow in attributes.get("yellow_peaks", [])
                                if yellow in peak_positions]

        for event_data in events_data:
            start_time = event_data.get('start_time', 0.0)
            end_time = event_data.get('end_time', 0.0)
            event_name = event_data.get('Classification', event_data.get('event_name', 'Unknown'))

            original_duration = end_time - start_time
            if original_duration <= 0:
                continue

            # ECG block: use only rpeaks/time, no big signal arrays
            hrv_time = hrv_frequency = hrv_nonlinear = heart_rate = None
            duration = original_duration

            if len(rpeaks_timing) >= 3:
                # Event rpeaks window
                start_index_rpeaks = int(np.argmin(np.abs(rpeaks_timing - start_time)))
                end_index_rpeaks = int(np.argmin(np.abs(rpeaks_timing - end_time)))
                event_rpeaks_timing = rpeaks_timing[start_index_rpeaks:end_index_rpeaks]
                new_rpeaks_times = event_rpeaks_timing.copy().tolist()
                updated_duration = False

                # Remove artifact peaks and adjust
                for i in yellow_rpeaks_mapped:
                    if start_index_rpeaks <= i < end_index_rpeaks:
                        local_i = i - start_index_rpeaks
                        if 0 < local_i < len(event_rpeaks_timing) - 1:
                            updated_duration = True
                            prev_it = event_rpeaks_timing[local_i] - event_rpeaks_timing[local_i - 1]
                            next_it = event_rpeaks_timing[local_i + 1] - event_rpeaks_timing[local_i]
                            combined = prev_it + next_it
                            # remove local_i and its right neighbor
                            if local_i + 1 < len(new_rpeaks_times):
                                new_rpeaks_times.pop(local_i + 1)
                            if local_i < len(new_rpeaks_times):
                                new_rpeaks_times.pop(local_i)
                                for j in range(local_i, len(new_rpeaks_times)):
                                    new_rpeaks_times[j] -= combined

                if updated_duration:
                    # conservative recompute
                    duration = end_time - start_time

                if len(new_rpeaks_times) >= 3:
                    new_rpeaks_indices = (closest_indices(time_np, new_rpeaks_times) if sorted_time else
                                          np.array([np.argmin(np.abs(time_np - t)) for t in new_rpeaks_times]))
                    new_rpeaks_dict = {'ECG_R_Peaks': new_rpeaks_indices}
                    try:
                        hrv_time = nk.hrv_time(new_rpeaks_dict, sampling_rate=sampling_rate)
                        hrv_frequency = nk.hrv_frequency(
                            new_rpeaks_dict, sampling_rate=sampling_rate, normalize=False,
                            psd_method='welch', interpolation_rate=100
                        )
                        hrv_nonlinear = nk.hrv_nonlinear(new_rpeaks_dict, sampling_rate=sampling_rate)
                        rr = np.diff(new_rpeaks_times)
                        if rr.size > 0:
                            RR_mean = float(np.mean(rr))
                            RR_std = float(np.std(rr))
                            RR_min = float(np.min(rr))
                            RR_max = float(np.max(rr))
                            HR_mean = 60.0 / RR_mean
                            HR_std = 60.0 / RR_std if RR_std > 0 else 0.0
                            HR_min = 60.0 / RR_max
                            HR_max = 60.0 / RR_min
                            heart_rate = pd.DataFrame({
                                'HR_mean': [HR_mean], 'HR_std': [HR_std], 'HR_min': [HR_min], 'HR_max': [HR_max],
                                'RR_mean': [RR_mean], 'RR_std': [RR_std], 'RR_min': [RR_min], 'RR_max': [RR_max]
                            })
                    except (ValueError, RuntimeError) as e:
                        logger.warning("HRV failed for %s, %s: %s", sub_id, event_name, e)
                        hrv_time = hrv_frequency = hrv_nonlinear = heart_rate = None

            # GSC/EDA metrics computed with NeuroKit2
            if gsc is not None:
                gsc_data = compute_eda_metrics_in_window(event_gsc(start_time, end_time), sub_id, start_time, end_time)
            else:
                gsc_data = pd.DataFrame({
                    'SCL_mean': [np.nan],
                    'SCR_mean': [np.nan],
                    'SCR_std': [np.nan],
                    'SCR_count': [np.nan],
                    'SCL_mean_sliding_mean': [np.nan],
                    'SCL_mean_sliding_std': [np.nan],
                    'SCR_mean_sliding_mean': [np.nan],
                    'SCR_mean_sliding_std': [np.nan],
                    'SCR_std_sliding_mean': [np.nan],
                    'SCR_std_sliding_std': [np.nan],
                    'SCR_count_sliding_mean': [np.nan],
                    'SCR_count_sliding_std': [np.nan]
                })

            # Combine metrics
            if heart_rate is not None:
                metrics = pd.concat([heart_rate, hrv_time, hrv_frequency, hrv_nonlinear, gsc_data], axis=1)
            else:
                empty_ecg = pd.DataFrame({
                    'HR_mean': [np.nan], 'HR_std': [np.nan], 'HR_min': [np.nan], 'HR_max': [np.nan],
                    'RR_mean': [np.nan], 'RR_std': [np.nan], 'RR_min': [np.nan], 'RR_max': [np.nan]
                })
                metrics = pd.concat([empty_ecg, gsc_data], axis=1)

            # Metadata
            metrics['Subject'] = sub_id
            metrics['Event'] = event_name
            metrics['Original Duration'] = original_duration
            metrics['New Duration'] = duration
            metrics['duration_diff'] = original_duration - duration

            eda_columns = list(gsc_data.columns)
            ecg_columns = [column for column in metrics.columns
                           if column not in eda_columns and column not in {
                               "Subject", "Event", "Original Duration", "New Duration", "duration_diff"
                           }]
            column_order = ["Subject", "Event", "duration_diff", "Original Duration", "New Duration"]
            if rpeaks:
                column_order.extend(ecg_columns)
            if gsc is not None or not rpeaks:
                column_order.extend(eda_columns)
            metrics = metrics[column_order]

            metric_frames.append(pl.from_pandas(metrics))
            if heart_rate is not None or (gsc is not None and gsc_data["SCL_mean"].notna().any()):
                completed.append((sub_id, event_name))

        logger.info("Processed %s (%d events) in %.3fs", sub_id, len(events_data), perf_counter() - subject_start)

    if not metric_frames:
        logger.error("No results were generated for project %s", project)
        return 1

    results = pl.concat(metric_frames, how="diagonal_relaxed")
    results_path = data_manager.results_parquet_path
    if Path(results_path).exists():
        old = pl.read_parquet(results_path).drop("Index", strict=False)
        successful = pl.DataFrame(completed, schema=["Subject", "Event"], orient="row",
                                  schema_overrides={"Subject": pl.String, "Event": pl.String}).unique()
        old_keys = old.select("Subject", "Event").unique()
        new_only = results.join(old_keys, on=["Subject", "Event"], how="anti")
        refreshed = results.join(successful, on=["Subject", "Event"], how="semi").join(
            old_keys, on=["Subject", "Event"], how="semi"
        )
        unchanged = old.join(successful, on=["Subject", "Event"], how="anti")
        results = pl.concat([unchanged, refreshed, new_only], how="diagonal_relaxed")
    atomic_write(results_path, results.with_row_index("Index").write_parquet)

    if completed:
        done_keys = pl.DataFrame(completed, schema=keys, orient="row").unique().with_columns(done=pl.lit(True))
        queue = (queue.join(done_keys, on=keys, how="left")
                 .with_columns(pl.when((pl.col("project") == project) & pl.col("done").fill_null(False))
                               .then(pl.lit("done")).otherwise(pl.col("status")).alias("status"))
                 .drop("done"))
        atomic_write(queue_path, queue.write_csv)
    logger.info("Wrote %d result rows (%d queued events completed) in %.3fs to %s",
                len(metric_frames), len(completed), perf_counter() - started, results_path)
    return 0 if len(set(completed)) == queued.height else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main(args.project))
