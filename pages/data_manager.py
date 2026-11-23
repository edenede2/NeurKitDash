"""
Data Manager Module for RoeeProj

This module handles data storage and retrieval using Parquet files for signal data
and CSV files for attributes, replacing the old HDF5 format.
"""

import os
import json
import pandas as pd
import polars as pl
import numpy as np
import neurokit2 as nk
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict
from functools import lru_cache
from threading import RLock
from time import perf_counter, sleep

from pages.paths import DATA_PATH

logger = logging.getLogger(__name__)

_write_locks = defaultdict(RLock)
_SIGNAL_SUFFIX = ".parquet"
_PENDING_SUFFIX = ".parquet.pending"


def atomic_write(path, writer):
    temp_path = str(path) + ".tmp"
    try:
        writer(temp_path)
        replace_file(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def replace_file(source, path):
    for attempt in range(4):
        try:
            os.replace(source, path)
            return
        except PermissionError:
            if attempt == 3:
                raise
            sleep(0.05 * 2 ** attempt)


@lru_cache(maxsize=16)
def _parquet_subjects(path, mtime_ns, size):
    return tuple(pl.scan_parquet(path).select("subject_id").unique().collect().get_column("subject_id"))


def _json_numpy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class DataManager:
    """
    Central data management class that handles all data operations
    using Parquet and CSV formats instead of HDF5
    """

    def __init__(self, project_name: str, base_data_path: str = None):
        self.project_name = project_name
        self.base_data_path = os.fspath(base_data_path or DATA_PATH)

        # File paths
        self.signal_parquet_path = os.path.join(
            self.base_data_path, f"{project_name}_mindware_signals.parquet"
        )
        self.events_parquet_path = os.path.join(
            self.base_data_path, f"{project_name}_mindware_events.parquet"
        )
        self.attributes_csv_path = os.path.join(
            self.base_data_path, f"{project_name}_mindware_attributes.csv"
        )
        self.results_parquet_path = os.path.join(
            self.base_data_path, f"{project_name}_mindware_results.parquet"
        )

        # Per-subject signal directory — one small parquet per subject instead of
        # a single monolithic file that must be fully re-read on every save.
        self.signal_parquet_dir = os.path.join(
            self.base_data_path, f"{project_name}_signals"
        )

        # Ensure directories exist
        os.makedirs(self.base_data_path, exist_ok=True)
        os.makedirs(self.signal_parquet_dir, exist_ok=True)

    @property
    def write_lock(self):
        return _write_locks[os.path.normcase(os.path.abspath(self.signal_parquet_dir))]

    # ------------------------------------------------------------------
    # Helpers for per-subject signal access (used by downstream scripts)
    # ------------------------------------------------------------------

    def has_signal_data(self) -> bool:
        """Return True if any signal data exists (per-subject dir or legacy monolithic)."""
        if os.path.isdir(self.signal_parquet_dir):
            for f in os.listdir(self.signal_parquet_dir):
                if f.endswith('.parquet') and not f.endswith('.tmp'):
                    return True
        return os.path.exists(self.signal_parquet_path)

    def get_subject_signal_path(self, subject_id: str) -> Optional[str]:
        """Return the parquet file path for a subject's signal data, or None."""
        per_subj = os.path.join(self.signal_parquet_dir, f"{subject_id}.parquet")
        if os.path.exists(per_subj):
            return per_subj
        if os.path.exists(self.signal_parquet_path):
            return self.signal_parquet_path  # legacy fallback
        return None

    def scan_subject_signal(self, subject_id: str, columns: Optional[List[str]] = None) -> pl.LazyFrame:
        """Return a lazy frame for one subject's signal data.

        Reads from the per-subject file when available, falling back to the
        legacy monolithic parquet with a subject_id filter.
        """
        per_subj = os.path.join(self.signal_parquet_dir, f"{subject_id}.parquet")
        if os.path.exists(per_subj):
            lf = pl.scan_parquet(per_subj)
        elif os.path.exists(self.signal_parquet_path):
            lf = pl.scan_parquet(self.signal_parquet_path).filter(
                pl.col('subject_id') == subject_id
            )
        else:
            # Return an empty LazyFrame with expected columns
            cols = columns or ['subject_id', 'time', 'signal', 'motion']
            return pl.DataFrame({c: [] for c in cols}).lazy()
        if columns:
            lf = lf.select(columns)
        return lf

    def get_signal_schema(self, subject_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the parquet schema for signal data (column names and types).

        Checks per-subject file first, then legacy monolithic.
        """
        if subject_id:
            per_subj = os.path.join(self.signal_parquet_dir, f"{subject_id}.parquet")
            if os.path.exists(per_subj):
                return dict(pl.scan_parquet(per_subj).schema)
        # Fallback: monolithic
        if os.path.exists(self.signal_parquet_path):
            return dict(pl.scan_parquet(self.signal_parquet_path).schema)
        # Fallback: first file in per-subject dir
        if os.path.isdir(self.signal_parquet_dir):
            for f in os.listdir(self.signal_parquet_dir):
                if f.endswith('.parquet') and not f.endswith('.tmp'):
                    return dict(pl.scan_parquet(
                        os.path.join(self.signal_parquet_dir, f)
                    ).schema)
        return {}

    def save_subject_data(self, subject_id: str, signal_data: Dict[str, Any],
                          events_data: List[Dict], attributes: Dict[str, Any]) -> bool:
        with self.write_lock:
            try:
                events = self.stage_subject(subject_id, signal_data, events_data, attributes)
                return subject_id in self.commit_subjects({subject_id: events})
            except Exception:
                logger.exception("Could not save subject %s", subject_id)
                return False

    def stage_subject(self, subject_id, signal_data, events_data, attributes):
        """Stage one recording on disk, keeping only small event records in memory."""
        if not subject_id or subject_id != os.path.basename(subject_id):
            raise ValueError(f"Invalid subject ID: {subject_id!r}")
        if not len(signal_data.get("time", [])):
            raise ValueError("There are no signal samples to save")

        events = self._event_records(subject_id, events_data)
        metadata = dict(attributes)
        quality = metadata.get("quality")
        if quality is not None and len(quality) == len(signal_data["time"]):
            signal_data = {**signal_data, "quality": metadata.pop("quality")}

        pending_signal = self._subject_file(subject_id, _PENDING_SUFFIX)
        self._save_signal_data(subject_id, signal_data, pending_signal)
        atomic_write(self._subject_file(subject_id, ".json.pending"),
                     lambda path: self._write_metadata(path, metadata))
        return events

    def commit_subjects(self, staged):
        """Publish shared events once for a batch, then publish each signal."""
        if not staged:
            return set()
        event_records = [record for records in staged.values() for record in records]
        if event_records or os.path.exists(self.events_parquet_path):
            frames = []
            if os.path.exists(self.events_parquet_path):
                frames.append(pl.read_parquet(self.events_parquet_path)
                              .filter(~pl.col("subject_id").is_in(list(staged))))
            if event_records:
                frames.append(pl.DataFrame(event_records))
            events = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
            atomic_write(self.events_parquet_path, events.write_parquet)

        completed = set()
        for subject_id in staged:
            try:
                replace_file(self._subject_file(subject_id, ".json.pending"),
                             self._subject_file(subject_id, ".json"))
                replace_file(self._subject_file(subject_id, _PENDING_SUFFIX),
                             self._subject_file(subject_id, _SIGNAL_SUFFIX))
                completed.add(subject_id)
            except OSError:
                logger.exception("Could not publish subject %s", subject_id)
        return completed

    def _subject_file(self, subject_id, suffix):
        return os.path.join(self.signal_parquet_dir, subject_id + suffix)

    @staticmethod
    def _write_metadata(path, metadata):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, default=_json_numpy, separators=(",", ":"))

    def _save_signal_data(self, subject_id: str, signal_data: Dict[str, Any], path):
        """Save signal data to a per-subject Parquet file.

        Each subject gets its own file under signal_parquet_dir so that saving a new
        subject never requires reading back all previously-saved subjects.  This avoids
        the O(n²) memory growth that caused the 'memory allocation failed' crash when
        loading many subjects in a single session.
        """
        # Check if we have time data
        if 'time' not in signal_data or len(signal_data['time']) == 0:
            logger.warning("No time data for subject %s, skipping signal data save", subject_id)
            return

        time_arr = np.asarray(signal_data['time'], dtype=np.float64)
        time_length = len(time_arr)

        def prepare_array(values, expected_length, name):
            if len(values) == 0:
                return np.zeros(expected_length, dtype=np.float64)
            arr = np.asarray(values, dtype=np.float64)
            if len(arr) == expected_length:
                return arr
            logger.warning("%s length mismatch for subject %s: %d vs %d",
                           name, subject_id, len(arr), expected_length)
            if len(arr) > expected_length:
                return arr[:expected_length]
            padded = np.zeros(expected_length, dtype=np.float64)
            padded[:len(arr)] = arr
            return padded

        signal_df_data = {
            'subject_id': pl.repeat(subject_id, time_length, eager=True),
            'time': time_arr,
            'signal': prepare_array(signal_data.get('signal', []), time_length, 'Signal'),
            'motion': prepare_array(signal_data.get('motion', []), time_length, 'Motion'),
        }

        if 'gsc' in signal_data and len(signal_data['gsc']) > 0:
            gsc_values = signal_data['gsc']
            if len(gsc_values) == time_length:
                signal_df_data['gsc'] = np.asarray(gsc_values, dtype=np.float64)
                signal_df_data['eda_tonic'] = np.asarray(
                    signal_data.get('eda_tonic', np.zeros(time_length)), dtype=np.float64)
                signal_df_data['eda_phasic'] = np.asarray(
                    signal_data.get('eda_phasic', np.zeros(time_length)), dtype=np.float64)
            else:
                logger.warning("GSC length mismatch for subject %s, skipping GSC data", subject_id)

        if "quality" in signal_data:
            signal_df_data["quality"] = np.asarray(signal_data["quality"], dtype=np.float64)
        atomic_write(path, pl.DataFrame(signal_df_data).write_parquet)

    @staticmethod
    def _event_records(subject_id, events_data):
        events_list = []
        for event in events_data:
            event_record = {
                'subject_id': subject_id,
                'event_name': str(event.get('Classification', event.get('name', ''))),
                'start_time': float(event.get('Start_Time', 0)),
                'end_time': float(event.get('End_Time', 0)),
                'done': bool(event.get('done', False)),
                'artifacts_percentage': float(event.get('artifacts_percent', 0)),
                'corrected_rpeaks_percentage': float(event.get('corrected_rpeaks_percent', 0)),
                'avg_motion': float(event.get('avg_motion', 0)),
                'avg_quality': float(event.get('avg_quality', 0)),
                'duration_minutes': float((event.get('End_Time', 0) - event.get('Start_Time', 0)) / 60)
            }
            events_list.append(event_record)

        return events_list

    def load_subject_signal_data(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Load signal data for a specific subject.

        Checks the per-subject parquet file first (new format), then falls back to
        the legacy monolithic parquet for subjects saved before this change.
        """
        try:
            # New per-subject file format
            subject_parquet = os.path.join(self.signal_parquet_dir, f"{subject_id}.parquet")
            if os.path.exists(subject_parquet):
                df = pl.read_parquet(subject_parquet)
                if df.height > 0:
                    return df.to_pandas().to_dict('list')
                return None

            # Legacy: monolithic parquet (backward compat)
            if not os.path.exists(self.signal_parquet_path):
                return None
            subject_df = (
                pl.scan_parquet(self.signal_parquet_path)
                .filter(pl.col('subject_id') == subject_id)
                .collect()
            )
            if subject_df.height == 0:
                return None
            return subject_df.to_pandas().to_dict('list')

        except (IOError, ValueError, KeyError) as e:
            logger.error("Error loading signal data for subject %s: %s", subject_id, e)
            return None

    def load_subject_events_data(self, subject_id: str) -> Optional[List[Dict]]:
        """Load events data for a specific subject"""
        try:
            if not os.path.exists(self.events_parquet_path):
                return None

            subject_df = (pl.scan_parquet(self.events_parquet_path)
                          .filter(pl.col('subject_id') == subject_id).collect())

            if subject_df.height == 0:
                return None

            return subject_df.to_dicts()

        except (IOError, ValueError, KeyError) as e:
            logger.error("Error loading events data for subject %s: %s", subject_id, e)
            return None

    def load_subject_attributes(self, subject_id: str, *, include_quality=True) -> Optional[Dict[str, Any]]:
        """Load attributes for a specific subject"""
        metadata_path = self._subject_file(subject_id, ".json")
        signal_path = self._subject_file(subject_id, _SIGNAL_SUFFIX)
        if os.path.isfile(metadata_path) and os.path.isfile(signal_path):
            with open(metadata_path, encoding="utf-8") as file:
                attributes = json.load(file)
            if include_quality and "quality" in pl.read_parquet_schema(signal_path):
                attributes["quality"] = pl.read_parquet(signal_path, columns=["quality"])["quality"].to_list()
            return attributes
        try:
            if not os.path.exists(self.attributes_csv_path):
                return None
            try:
                lf = (
                    pl.scan_csv(self.attributes_csv_path, has_header=True, schema_overrides={"subject_id": pl.String})
                    .filter(pl.col('subject_id') == subject_id)
                    .select(['attributes_json'])
                )
                df = lf.collect(streaming=True)
                if df.height == 0:
                    return None
                attrs_json = df['attributes_json'][0]
                return json.loads(attrs_json)
            except Exception as e:
                logger.warning("Polars scan_csv failed, falling back to pandas chunks: %s", e)

            # Fallback: pandas chunked read (filter per chunk)
            usecols = ['subject_id', 'attributes_json']
            for chunk in pd.read_csv(self.attributes_csv_path, usecols=usecols, chunksize=128, dtype={"subject_id": str}):
                hit = chunk[chunk['subject_id'] == subject_id]
                if not hit.empty:
                    return json.loads(hit.iloc[0]['attributes_json'])
            return None

        except (IOError, ValueError, KeyError, MemoryError) as e:
            logger.error("Error loading attributes for subject %s: %s", subject_id, e)
            return None

    def get_all_subjects(self, use_cache: bool = True) -> List[str]:
        """List subjects across both storage formats, excluding interrupted imports."""
        files = os.listdir(self.signal_parquet_dir)
        completed = {name.removesuffix(_SIGNAL_SUFFIX) for name in files if name.endswith(_SIGNAL_SUFFIX)}
        pending = {name.removesuffix(_PENDING_SUFFIX) for name in files if name.endswith(_PENDING_SUFFIX)}
        subjects = set(completed)
        for path in (self.signal_parquet_path, self.events_parquet_path):
            if os.path.isfile(path):
                try:
                    stat = os.stat(path)
                    read_subjects = _parquet_subjects if use_cache else _parquet_subjects.__wrapped__
                    subjects.update(read_subjects(path, stat.st_mtime_ns, stat.st_size))
                except (OSError, pl.exceptions.PolarsError):
                    logger.exception("Unable to list subjects in %s", path)
        return sorted(subjects - (pending - completed) - {None})

    def get_events_by_type(self, event_name: str = None, subject_id: str = None) -> List[Dict]:
        """
        Get events filtered by event type and/or subject

        Args:
            event_name: Optional event name filter
            subject_id: Optional subject ID filter

        Returns:
            List of matching event dictionaries
        """
        try:
            if not os.path.exists(self.events_parquet_path):
                return []

            df = pl.read_parquet(self.events_parquet_path)

            # Apply filters
            if subject_id:
                df = df.filter(pl.col('subject_id') == subject_id)

            if event_name:
                df = df.filter(pl.col('event_name') == event_name)

            return df.to_pandas().to_dict('records')

        except Exception as e:
            logger.error("Error loading filtered events: %s", e)
            return []

    def get_event_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all events in the dataset

        Returns:
            Dictionary containing event summary statistics
        """
        try:
            if not os.path.exists(self.events_parquet_path):
                return {}

            df = pl.read_parquet(self.events_parquet_path)

            summary = {
                'total_events': df.height,
                'total_subjects': df['subject_id'].n_unique(),
                'event_types': df['event_name'].unique().to_list(),
                'completed_events': df.filter(pl.col('done') == True).height,
                'avg_duration_minutes': df['duration_minutes'].mean(),
                'total_duration_minutes': df['duration_minutes'].sum(),
            }

            # Per-event-type statistics
            event_stats = (
                df.group_by('event_name')
                .agg([
                    pl.count().alias('count'),
                    pl.col('duration_minutes').mean().alias('avg_duration'),
                    pl.col('avg_motion').mean().alias('avg_motion'),
                    pl.col('avg_quality').mean().alias('avg_quality'),
                    pl.col('done').sum().alias('completed_count')
                ])
                .to_pandas()
                .to_dict('records')
            )

            summary['by_event_type'] = {
                row['event_name']: {
                    'count': row['count'],
                    'avg_duration_minutes': row['avg_duration'],
                    'avg_motion': row['avg_motion'],
                    'avg_quality': row['avg_quality'],
                    'completed_count': row['completed_count'],
                    'completion_rate': row['completed_count'] / row['count'] if row['count'] > 0 else 0
                }
                for row in event_stats
            }

            return summary

        except Exception as e:
            logger.error("Error generating event summary: %s", e)
            return {}

    def update_event_status(self, subject_id: str, event_name: str, done: bool = True) -> bool:
        """
        Update the completion status of a specific event

        Args:
            subject_id: Subject identifier
            event_name: Event name to update
            done: Completion status

        Returns:
            Success status
        """
        try:
            if not os.path.exists(self.events_parquet_path):
                logger.error("Events file does not exist")
                return False

            df = pl.read_parquet(self.events_parquet_path)

            # Update the specific event
            updated_df = df.with_columns(
                pl.when(
                    (pl.col('subject_id') == subject_id) &
                    (pl.col('event_name') == event_name)
                )
                .then(done)
                .otherwise(pl.col('done'))
                .alias('done')
            )

            # Save with atomic write
            temp_path = self.events_parquet_path + '.tmp'
            try:
                updated_df.write_parquet(temp_path)
                if os.path.exists(self.events_parquet_path):
                    os.remove(self.events_parquet_path)
                os.rename(temp_path, self.events_parquet_path)

                logger.info("Updated event status for %s - %s: %s", subject_id, event_name, done)
                return True

            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e

        except Exception as e:
            logger.error("Error updating event status: %s", e)
            return False

    def process_raw_csv_data(self, csv_path: str, template_path: str) -> Dict[str, Any]:
        """
        Process raw CSV data file using project template and save to parquet/CSV format

        Args:
            csv_path: Path to the raw CSV data file
            template_path: Path to the project template JSON file

        Returns:
            Dictionary with processing results
        """
        try:
            # Load project configuration
            with open(template_path, 'r') as f:
                project_config = json.load(f)

            # Load raw CSV data
            raw_data_df = pd.read_csv(csv_path)
            logger.info("Loaded raw data from %s: %d rows", csv_path, len(raw_data_df))

            # Try to load events data from various sources
            events_df = None

            # Check if there's a separate events file
            events_csv_path = csv_path.replace('UpdatedData.csv', 'events.csv')
            if os.path.exists(events_csv_path):
                events_df = pd.read_csv(events_csv_path)
                logger.info("Loaded events from separate file: %d events", len(events_df))

            # Extract events from subject data (look for events columns in main data)
            if events_df is None:
                # Check for embedded events data
                events_cols = [col for col in raw_data_df.columns if 'Event' in col or 'event' in col]
                if events_cols:
                    logger.info("Found event columns in main data: %s", events_cols)
                    # Create events dataframe from embedded data
                    # This would need to be customized based on data format

            # For now, create empty events dataframe if none found
            if events_df is None:
                events_df = pd.DataFrame()
                logger.warning("No events data found, processing signals only")

            # Group data by subject and process each
            processed_subjects = []
            total_events = 0
            errors = []

            # Extract unique subjects from the data
            if 'subject_id' in raw_data_df.columns:
                subject_ids = raw_data_df['subject_id'].unique()
            elif 'Subject' in raw_data_df.columns:
                subject_ids = raw_data_df['Subject'].unique()
                raw_data_df = raw_data_df.rename(columns={'Subject': 'subject_id'})
            else:
                # Try to infer subjects from filename patterns or create single subject
                logger.warning("No subject ID column found, treating as single subject")
                subject_ids = ['unknown_subject']
                raw_data_df['subject_id'] = 'unknown_subject'

            logger.info("Found %d unique subjects to process", len(subject_ids))

            for subject_id in subject_ids:
                try:
                    logger.info("Processing subject: %s", subject_id)

                    # Filter data for this subject
                    subject_data_df = raw_data_df[raw_data_df['subject_id'] == subject_id].copy()

                    # Filter events for this subject if available
                    subject_events_df = pd.DataFrame()
                    if not events_df.empty:
                        if 'subject_id' in events_df.columns:
                            subject_events_df = events_df[events_df['subject_id'] == subject_id].copy()
                        else:
                            # Use all events for all subjects if no subject_id in events
                            subject_events_df = events_df.copy()

                    # Process the subject data
                    signal_data, events_data, attributes = process_raw_mindware_data(
                        subject_data_df, subject_events_df, project_config, subject_id
                    )

                    # Save processed data
                    success = self.save_subject_data(subject_id, signal_data, events_data, attributes)

                    if success:
                        processed_subjects.append(subject_id)
                        total_events += len(events_data)
                        logger.info("Successfully processed subject %s with %d events", subject_id, len(events_data))
                    else:
                        error_msg = f"Failed to save data for subject {subject_id}"
                        errors.append(error_msg)
                        logger.error(error_msg)

                except Exception as e:
                    error_msg = f"Error processing subject {subject_id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    continue

            result = {
                'success': len(processed_subjects) > 0,
                'processed_subjects': processed_subjects,
                'total_events': total_events,
                'errors': errors,
                'message': f"Processed {len(processed_subjects)} subjects with {total_events} total events"
            }

            logger.info("Data processing completed: %s", result['message'])
            return result

        except Exception as e:
            error_msg = f"Error processing raw CSV data: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'processed_subjects': [],
                'total_events': 0,
                'errors': [error_msg],
                'message': error_msg
            }

    def remove_subject(self, subject_id: str) -> bool:
        """Remove the subject's legacy and per-subject data."""
        try:
            with self.write_lock:
                for path in (self.signal_parquet_path, self.events_parquet_path):
                    if os.path.isfile(path):
                        original = pl.read_parquet(path)
                        remaining = original.filter(pl.col("subject_id") != subject_id)
                        if remaining.height != original.height:
                            atomic_write(path, remaining.write_parquet)
                if os.path.isfile(self.attributes_csv_path):
                    original = pd.read_csv(self.attributes_csv_path, dtype={"subject_id": str})
                    remaining = original[original["subject_id"] != subject_id]
                    if len(remaining) != len(original):
                        atomic_write(self.attributes_csv_path, lambda path: remaining.to_csv(path, index=False))
                for suffix in (_SIGNAL_SUFFIX, _PENDING_SUFFIX, ".json", ".json.pending"):
                    path = self._subject_file(subject_id, suffix)
                    if os.path.isfile(path):
                        os.remove(path)
            return True
        except Exception:
            logger.exception("Could not remove subject %s", subject_id)
            return False

    def migrate_from_hdf5(self, hdf5_path: str, project_config: Dict = None) -> bool:
        """
        Migrate data from existing HDF5 files to Parquet/CSV format with event processing

        Args:
            hdf5_path: Path to the HDF5 file to migrate
            project_config: Optional project configuration for event processing

        Returns:
            bool: Success status
        """
        try:
            import h5py

            # Try to load project config if not provided
            if project_config is None:
                template_path = os.path.join(
                    os.path.dirname(__file__), "templates", f"{self.project_name}_template.json"
                )
                if os.path.exists(template_path):
                    with open(template_path, 'r') as f:
                        project_config = json.load(f)
                        logger.info("Loaded project config from template: %s", template_path)
                else:
                    logger.warning("No project config found, migrating without event processing")

            with h5py.File(hdf5_path, 'r') as hdf_file:
                for subject_id in hdf_file.keys():
                    logger.info("Migrating subject %s with event processing", subject_id)

                    # Load raw signal data
                    raw_data_df = None
                    raw_events_df = None

                    if 'data' in hdf_file[subject_id]:
                        # Load the raw data as DataFrame for processing
                        raw_data_df = pd.read_hdf(hdf5_path, key=f'{subject_id}/data')
                        logger.info("Loaded raw data for %s: %d samples", subject_id, len(raw_data_df))

                    # Try to load events data
                    if 'events' in hdf_file[subject_id]:
                        try:
                            raw_events_df = pd.read_hdf(hdf5_path, key=f'{subject_id}/events')
                            logger.info("Loaded events data for %s: %d events", subject_id, len(raw_events_df))
                        except Exception as e:
                            logger.warning("Could not load events for %s: %s", subject_id, e)

                    # Check if we have events data in the data attributes
                    if raw_events_df is None and 'events_data' in hdf_file[subject_id]['data'].attrs:
                        try:
                            events_json = hdf_file[subject_id]['data'].attrs['events_data']
                            if isinstance(events_json, bytes):
                                events_json = events_json.decode('utf-8')
                            events_dict = json.loads(events_json)

                            # Convert to DataFrame format expected by event processing
                            events_list = []
                            for event_data in events_dict.values():
                                if 'Event' in event_data and 'Time' in event_data:
                                    events_list.append({
                                        'Event': event_data['Event'],
                                        'Time': event_data['Time']
                                    })

                            if events_list:
                                raw_events_df = pd.DataFrame(events_list)
                                logger.info("Converted events from JSON for %s: %d events", subject_id, len(raw_events_df))
                        except Exception as e:
                            logger.warning("Could not parse events JSON for %s: %s", subject_id, e)

                    # Process data with event processing if we have the necessary components
                    if raw_data_df is not None and project_config is not None:
                        if raw_events_df is not None:
                            # Use the new event processing system
                            signal_data, events_data, attributes = process_raw_mindware_data(
                                raw_data_df, raw_events_df, project_config, subject_id
                            )
                            logger.info("Processed %s with %d events using new system", subject_id, len(events_data))
                        else:
                            # Process without events (legacy migration)
                            logger.warning("No events data for %s, processing signals only", subject_id)
                            signal_data, events_data, attributes = process_raw_mindware_data(
                                raw_data_df, pd.DataFrame(), project_config, subject_id
                            )
                    else:
                        # Fallback to legacy migration
                        logger.warning("Using legacy migration for %s (no project config)", subject_id)
                        signal_data = {
                            'time': raw_data_df['time'].to_list() if 'time' in raw_data_df.columns else [],
                            'signal': raw_data_df.get('signal', []).to_list() if 'signal' in raw_data_df.columns else [],
                            'motion': raw_data_df.get('motion', []).to_list() if 'motion' in raw_data_df.columns else [],
                        }

                        if 'gsc' in raw_data_df.columns:
                            signal_data['gsc'] = raw_data_df['gsc'].to_list()
                            signal_data['eda_tonic'] = raw_data_df.get('eda_tonic', []).to_list()
                            signal_data['eda_phasic'] = raw_data_df.get('eda_phasic', []).to_list()

                        events_data = []
                        attributes = {}

                        # Load legacy attributes
                        if 'rpeaks' in hdf_file[subject_id]:
                            rpeaks_df = pd.read_hdf(hdf5_path, key=f'{subject_id}/rpeaks')
                            attributes['rpeaks'] = rpeaks_df[0].to_list()

                            if 'signal_attributes' in hdf_file[subject_id]['rpeaks'].attrs:
                                signal_attrs = hdf_file[subject_id]['rpeaks'].attrs['signal_attributes']
                                if isinstance(signal_attrs, bytes):
                                    signal_attrs = signal_attrs.decode('utf-8')
                                attributes.update(json.loads(signal_attrs))

                    # Save migrated data
                    self.save_subject_data(subject_id, signal_data, events_data, attributes)

            logger.info("Successfully migrated data from %s with event processing", hdf5_path)
            return True

        except (IOError, ValueError, KeyError) as e:
            logger.error("Error migrating from HDF5 %s: %s", hdf5_path, e)
            return False


def get_data_manager(project_name: str) -> DataManager:
    """Factory function to get a DataManager instance"""
    return DataManager(project_name)


def process_raw_mindware_data(data_df: pd.DataFrame, events_df: pd.DataFrame,
                            project_config: Dict, subject_id: str) -> Tuple[Dict, List, Dict]:
    """
    Process raw mindware data and return structured data for storage - OPTIMIZED

    Args:
        data_df: Raw signal dataframe
        events_df: Raw events dataframe with markers
        project_config: Project configuration from template
        subject_id: Subject identifier

    Returns:
        Tuple of (signal_data, events_data, attributes)
    """
    start_time = perf_counter()

    sampling_rate = project_config['parameters']['sampling_rate']
    channels = project_config['channel']
    events_config = project_config.get('events', {})

    # OPTIMIZED: Simplified safe_tolist - numpy arrays can be stored directly
    def safe_tolist(val):
        """Convert numpy arrays to lists efficiently"""
        if val is None:
            return None
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (list, tuple)):
            return list(val)
        elif hasattr(val, 'tolist'):
            return val.tolist()
        else:
            return val

    # Reduce debug output - only log essentials
    logger.debug("Processing subject %s: %d rows, %d cols", subject_id, data_df.shape[0], data_df.shape[1])

    # Extract signal data with versatile case-insensitive matching
    bio_cols = data_df.filter(regex='(?i)(bio|ecg|cardiac|heart)')
    bio_in_channels = any(channel.lower() == 'bio' for channel in channels)
    signal = bio_cols.values.flatten() if bio_in_channels and not bio_cols.empty else np.array([])

    # Time extraction
    time_cols = data_df.filter(regex='(?i)(time|timestamp)')
    if not time_cols.empty:
        time = time_cols.iloc[:, 0].values.flatten()
    else:
        time = np.arange(len(data_df)) / sampling_rate
        logger.warning("No time column found for %s, creating synthetic time array", subject_id)

    # Extract motion data with proper error handling - OPTIMIZED
    try:
        x_axis_cols = data_df.filter(regex='(?i)(x.axis|x_axis|x-axis|accel.*x)')
        y_axis_cols = data_df.filter(regex='(?i)(y.axis|y_axis|y-axis|accel.*y)')
        z_axis_cols = data_df.filter(regex='(?i)(z.axis|z_axis|z-axis|accel.*z)')

        if not x_axis_cols.empty and not y_axis_cols.empty:
            x_axis = x_axis_cols.values.flatten()
            y_axis = y_axis_cols.values.flatten()

            if len(x_axis) == len(y_axis) and len(x_axis) > 0:
                if not z_axis_cols.empty and len(z_axis_cols.values.flatten()) == len(x_axis):
                    z_axis = z_axis_cols.values.flatten()
                    motion = np.sqrt(x_axis**2 + y_axis**2 + z_axis**2) * 0.01
                else:
                    motion = np.sqrt(x_axis**2 + y_axis**2) * 0.01
            else:
                motion = np.zeros(len(time))
        else:
            motion = np.zeros(len(time))

    except Exception:
        motion = np.zeros(len(time))

    # Ensure motion and time have compatible lengths - OPTIMIZED
    if len(motion) != len(time):
        if len(motion) > len(time):
            ratio = len(motion) // len(time)
            if ratio > 0:
                motion = motion[::ratio][:len(time)]
        else:
            padded_motion = np.zeros(len(time))
            padded_motion[:len(motion)] = motion
            motion = padded_motion

    # Extract GSC data if available - OPTIMIZED
    gsc_signal = np.array([])
    gsc_cleaned = np.array([])
    gsc_tonic = np.array([])
    gsc_phasic = np.array([])

    gsc_in_channels = any(channel.lower() == 'gsc' for channel in channels)
    if gsc_in_channels:
        gsc_cols = data_df.filter(regex='(?i)(gsc|eda|skin.conductance|galvanic)')
        if not gsc_cols.empty:
            gsc_signal = gsc_cols.values.flatten()

            try:
                gsc_cleaned = nk.eda_clean(gsc_signal, sampling_rate=sampling_rate)
                gsc_decomposed = nk.eda_phasic(gsc_cleaned, sampling_rate=sampling_rate, method='highpass')
                gsc_tonic = gsc_decomposed['EDA_Tonic'].values
                gsc_phasic = gsc_decomposed['EDA_Phasic'].values
            except Exception:
                gsc_cleaned = gsc_signal
                gsc_tonic = np.zeros_like(gsc_signal)
                gsc_phasic = np.zeros_like(gsc_signal)

    # OPTIMIZED: Store numpy arrays directly - polars/parquet handle them efficiently
    signal_data = {
        'time': time,
        'signal': signal if len(signal) > 0 else np.array([]),
        'motion': motion,
    }

    # Add GSC data if available
    if len(gsc_cleaned) > 0:
        signal_data.update({
            'gsc': gsc_cleaned,
            'eda_tonic': gsc_tonic,
            'eda_phasic': gsc_phasic,
        })

    attributes = {}

    # Process ECG if signal exists - OPTIMIZED logging
    if len(signal) > 0:
        try:
            cleaned_signal = nk.ecg_clean(signal, sampling_rate=sampling_rate)

            _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=sampling_rate, correct_artifacts=True)

            rpeaks_indices = rpeaks.get('ECG_R_Peaks', [])
            if hasattr(rpeaks_indices, 'tolist'):
                rpeaks_indices = rpeaks_indices.tolist()

            if len(rpeaks_indices) > 0:
                quality = nk.ecg_quality(cleaned_signal, sampling_rate=sampling_rate)
                quality_arr = np.asarray(quality)
                quality_rpeaks = quality_arr[rpeaks_indices]

                signal_data['signal'] = cleaned_signal

                # OPTIMIZED: Use numpy operations instead of list comprehensions
                yellow_mask = quality_rpeaks < 0.6
                yellow_peaks = np.array(rpeaks_indices)[yellow_mask].tolist() if np.any(yellow_mask) else []

                attributes = {
                    'rpeaks': rpeaks_indices,
                    'yellow_peaks': yellow_peaks,
                    'ectopic_beats': safe_tolist(rpeaks.get('ECG_fixpeaks_ectopic', [])),
                    'longshort_beats': safe_tolist(rpeaks.get('ECG_fixpeaks_longshort', [])),
                    'false_negatives': safe_tolist(rpeaks.get('ECG_fixpeaks_missed', [])),
                    'false_positives': safe_tolist(rpeaks.get('ECG_fixpeaks_extra', [])),
                    'quality': quality_arr,
                }
                logger.debug("ECG processed for %s: %d R-peaks", subject_id, len(rpeaks_indices))
            else:
                signal_data['signal'] = cleaned_signal
                attributes = {}

        except Exception as e:
            logger.warning("ECG processing failed for %s: %s", subject_id, e)
            signal_data['signal'] = signal
            attributes = {}

    # Process GSC attributes if GSC data exists
    if len(gsc_cleaned) > 0:
        try:
            _, gsc_peaks = nk.eda_peaks(gsc_cleaned, sampling_rate=sampling_rate)

            attributes['gsc_attributes'] = {
                'SCR_Onsets': safe_tolist(gsc_peaks.get('SCR_Onsets', [])),
                'SCR_Peaks': safe_tolist(gsc_peaks.get('SCR_Peaks', [])),
                'SCR_Height': safe_tolist(gsc_peaks.get('SCR_Height', [])),
                'SCR_Amplitude': safe_tolist(gsc_peaks.get('SCR_Amplitude', [])),
                'SCR_RiseTime': safe_tolist(gsc_peaks.get('SCR_RiseTime', [])),
                'SCR_Recovery': safe_tolist(gsc_peaks.get('SCR_Recovery', [])),
                'SCR_RecoveryTime': safe_tolist(gsc_peaks.get('SCR_RecoveryTime', []))
            }
        except Exception:
            pass  # GSC peaks processing is optional

    # Process events using the project configuration
    events_data = []
    if events_df is not None and not events_df.empty and events_config:
        events_data = _process_events_from_config(
            events_df, events_config, time, signal_data, sampling_rate, subject_id
        )

    elapsed = perf_counter() - start_time
    logger.debug("Subject %s processed in %.2fs", subject_id, elapsed)

    return signal_data, events_data, attributes


def _process_events_from_config(events_df: pd.DataFrame, events_config: Dict,
                              time_array: np.ndarray, signal_data: Dict,
                              sampling_rate: int, subject_id: str) -> List[Dict]:
    """
    Process events based on project configuration

    Args:
        events_df: Raw events dataframe with markers
        events_config: Events configuration from project template
        time_array: Time array from signal data
        signal_data: Processed signal data dictionary
        sampling_rate: Sampling rate
        subject_id: Subject identifier

    Returns:
        List of processed event dictionaries
    """
    processed_events = []

    # Create marker lookup dictionary from events dataframe - OPTIMIZED
    marker_times = {}

    # Collect all raw markers using vectorized operations where possible
    if 'Event' in events_df.columns and 'Time' in events_df.columns:
        for _, row in events_df.iterrows():
            marker_times[str(row['Event']).strip()] = float(row['Time'])
    elif 'marker' in events_df.columns and 'time' in events_df.columns:
        for _, row in events_df.iterrows():
            marker_times[str(row['marker']).strip()] = float(row['time'])
    elif 'Name' in events_df.columns and 'Time' in events_df.columns:
        for _, row in events_df.iterrows():
            marker_times[str(row['Name']).strip()] = float(row['Time'])
    elif 'Event Type' in events_df.columns and 'Time' in events_df.columns:
        for _, row in events_df.iterrows():
            marker_times[str(row['Event Type']).strip()] = float(row['Time'])
    else:
        logger.warning("Events dataframe doesn't have expected columns: %s", list(events_df.columns))
        return processed_events

    # Create keyboard-to-marker conversions for fibro project
    keyboard_conversions = {
        'Keyboard 1': 'F1',
        'Keyboard 2': 'F2',
        'Keyboard 3': 'F3',
        'Keyboard 4': 'F4',
        'Keyboard 5': 'F5'
    }

    # Create UDP block conversions based on template blocks_shorts
    udp_block_conversions = {
        'learning_block1': 'lb1',
        'learning_block2': 'lb2',
        'learning_block3': 'lb3',
        'learning_block4': 'lb4',
        'reversal_block1': 'rb1',
        'reversal_block2': 'rb2',
        'reversal_block3': 'rb3',
        'experiment_finish': 'ef'
    }

    # Add converted markers to the lookup - OPTIMIZED (no debug prints)
    for kb_marker, std_marker in keyboard_conversions.items():
        if kb_marker in marker_times:
            marker_times[std_marker] = marker_times[kb_marker]

    # Add UDP block conversions to the lookup
    for udp_marker, std_marker in udp_block_conversions.items():
        if udp_marker in marker_times:
            marker_times[std_marker] = marker_times[udp_marker]
        udp_prefixed = f"UDP:{udp_marker}"
        if udp_prefixed in marker_times:
            marker_times[std_marker] = marker_times[udp_prefixed]

    # Generate synthetic block markers if they don't exist but we have the main session markers
    if 'F2' in marker_times and 'F3' in marker_times:
        session_start = marker_times['F2']
        session_end = marker_times['F3']
        session_duration = session_end - session_start

        # Check if we're missing any lb/rb markers and generate them
        missing_markers = []
        for short_name in ['lb1', 'lb2', 'lb3', 'lb4', 'rb1', 'rb2', 'rb3']:
            if short_name not in marker_times:
                missing_markers.append(short_name)

        if missing_markers:
            pass  # Markers will be generated if needed

            # # Generate learning block markers (first half of session)
            # learning_duration = session_duration * 0.4  # 40% for learning
            # learning_start = session_start + (session_duration * 0.05)  # Start 5% into session

            # if 'lb1' not in marker_times:
            #     marker_times['lb1'] = learning_start
            #     print(f"      [DEBUG] Generated synthetic marker 'lb1' at time {learning_start:.2f}")

            # if 'lb2' not in marker_times:
            #     marker_times['lb2'] = learning_start + (learning_duration * 0.33)
            #     print(f"      [DEBUG] Generated synthetic marker 'lb2' at time {marker_times['lb2']:.2f}")

            # if 'lb3' not in marker_times:
            #     marker_times['lb3'] = learning_start + (learning_duration * 0.66)
            #     print(f"      [DEBUG] Generated synthetic marker 'lb3' at time {marker_times['lb3']:.2f}")

            # if 'lb4' not in marker_times:
            #     marker_times['lb4'] = learning_start + learning_duration
            #     print(f"      [DEBUG] Generated synthetic marker 'lb4' at time {marker_times['lb4']:.2f}")

            # # Generate reversal block markers (second half of session)
            # reversal_duration = session_duration * 0.4  # 40% for reversal
            # reversal_start = learning_start + learning_duration + (session_duration * 0.05)  # Gap between learning and reversal

            # if 'rb1' not in marker_times:
            #     marker_times['rb1'] = reversal_start
            #     print(f"      [DEBUG] Generated synthetic marker 'rb1' at time {reversal_start:.2f}")

            # if 'rb2' not in marker_times:
            #     marker_times['rb2'] = reversal_start + (reversal_duration * 0.5)
            #     print(f"      [DEBUG] Generated synthetic marker 'rb2' at time {marker_times['rb2']:.2f}")

            # if 'rb3' not in marker_times:
            #     marker_times['rb3'] = reversal_start + reversal_duration
            #     print(f"      [DEBUG] Generated synthetic marker 'rb3' at time {marker_times['rb3']:.2f}")

    # Process each event in the configuration
    sorted_time = bool(np.all(np.isfinite(time_array)) and np.all(time_array[1:] > time_array[:-1]))
    for event_key, event_config in events_config.items():
        try:
            event_name = event_config.get('name', event_key)
            start_marker = event_config.get('start')
            end_marker = event_config.get('end')
            step_marker = event_config.get('step', None)  # Optional
            length_seconds = event_config.get('length', None)  # Optional

            # Find start and end times
            start_time = marker_times.get(start_marker)
            end_time = marker_times.get(end_marker)
            if start_time is None or end_time is None:
                logger.debug("Missing markers %s / %s for %s, event %s",
                             start_marker, end_marker, subject_id, event_name)
                continue
            if step_marker:
                start_time += float(step_marker)
            if length_seconds:
                end_time = min(end_time, start_time + float(length_seconds))

            if start_time >= end_time:
                print(f"      [WARNING] Invalid time range for event '{event_name}': {start_time} >= {end_time}")
                continue

            # Convert times to sample indices
            start_idx = _time_to_index(start_time, time_array, sorted_time)
            end_idx = _time_to_index(end_time, time_array, sorted_time)

            if start_idx is None or end_idx is None:
                print(f"      [WARNING] Could not convert times to indices for event '{event_name}'")
                continue

            # Extract event segment data
            event_signal_data = _extract_event_segment(signal_data, start_idx, end_idx)

            # Calculate event metrics
            event_metrics = _calculate_event_metrics(
                event_signal_data, sampling_rate, start_time, end_time
            )

            # Create event record
            event_record = {
                'Classification': event_name,
                'name': event_name,
                'Start_Time': start_time,
                'End_Time': end_time,
                'start_marker': start_marker,
                'end_marker': end_marker,
                'duration_seconds': end_time - start_time,
                'start_index': start_idx,
                'end_index': end_idx,
                'done': False,  # Default to not processed
                **event_metrics
            }

            # Add optional group information
            if 'group' in event_config:
                event_record['group'] = event_config['group']

            # Handle windowed events
            if 'window' in event_config:
                window_config = event_config['window']
                window_length = window_config.get('length', 4)  # seconds
                window_step = window_config.get('step', 1)  # seconds

                windowed_events = _create_windowed_events(
                    event_record, event_signal_data, window_length, window_step, sampling_rate
                )
                processed_events.extend(windowed_events)
            else:
                processed_events.append(event_record)

        except Exception as e:
            logger.warning("Failed to process event '%s': %s", event_key, e)
            continue

    return processed_events


def _time_to_index(target_time: float, time_array: np.ndarray, sorted_time=False) -> Optional[int]:
    """Convert time to the nearest sample index"""
    if len(time_array) == 0:
        return None

    if sorted_time:
        right = min(int(np.searchsorted(time_array, target_time)), len(time_array) - 1)
        left = max(right - 1, 0)
        closest_idx = min((left, right), key=lambda index: abs(time_array[index] - target_time))
    else:
        closest_idx = int(np.argmin(np.abs(time_array - target_time)))
    if abs(time_array[closest_idx] - target_time) > 1.0:
        return None
    return closest_idx


def _extract_event_segment(signal_data: Dict, start_idx: int, end_idx: int) -> Dict:
    """Extract signal data segment for an event"""
    event_data = {}

    for key, values in signal_data.items():
        if isinstance(values, list) and len(values) > 0:
            # Ensure indices are within bounds
            start_safe = max(0, min(start_idx, len(values) - 1))
            end_safe = max(start_safe + 1, min(end_idx, len(values)))

            event_data[key] = values[start_safe:end_safe]
        else:
            event_data[key] = []

    return event_data


def _calculate_event_metrics(event_data: Dict, sampling_rate: int,
                           start_time: float, end_time: float) -> Dict:
    """Calculate metrics for an event segment"""
    metrics = {
        'artifacts_percent': 0.0,
        'corrected_rpeaks_percent': 0.0,
        'avg_motion': 0.0,
        'avg_quality': 0.0,
        'duration_minutes': (end_time - start_time) / 60.0
    }

    try:
        # Calculate motion metrics
        if 'motion' in event_data and len(event_data['motion']) > 0:
            motion_values = np.array(event_data['motion'])
            metrics['avg_motion'] = float(np.mean(motion_values))

        # Calculate ECG quality metrics
        if 'signal' in event_data and len(event_data['signal']) > 0:
            signal_values = np.array(event_data['signal'])

            # Simple quality estimation based on signal variability
            if len(signal_values) > 10:
                signal_std = np.std(signal_values)
                signal_mean = np.abs(np.mean(signal_values))
                if signal_mean > 0:
                    cv = signal_std / signal_mean
                    # Quality score based on coefficient of variation (inverse relationship)
                    metrics['avg_quality'] = float(max(0, min(1, 1 - (cv / 2))))
                else:
                    metrics['avg_quality'] = 0.0

            # Artifact detection (simple threshold-based)
            if len(signal_values) > 0:
                signal_abs = np.abs(signal_values)
                threshold = np.percentile(signal_abs, 95) * 2  # Artifacts are extreme outliers
                artifacts = np.sum(signal_abs > threshold)
                metrics['artifacts_percent'] = float((artifacts / len(signal_values)) * 100)

        # Placeholder for R-peaks correction percentage
        # This would be calculated based on actual R-peaks correction algorithms
        metrics['corrected_rpeaks_percent'] = 0.0

    except Exception as e:
        print(f"      [WARNING] Error calculating event metrics: {e}")

    return metrics


def _create_windowed_events(base_event: Dict, event_data: Dict,
                          window_length: float, window_step: float,
                          sampling_rate: int) -> List[Dict]:
    """Create windowed sub-events from a larger event"""
    windowed_events = []

    event_duration = base_event['duration_seconds']
    if event_duration < window_length:
        # Event is shorter than window, return as-is
        return [base_event]

    num_windows = int((event_duration - window_length) / window_step) + 1

    for i in range(num_windows):
        window_start_offset = i * window_step
        window_end_offset = window_start_offset + window_length

        # Calculate absolute times
        window_start_time = base_event['Start_Time'] + window_start_offset
        window_end_time = base_event['Start_Time'] + window_end_offset

        # Calculate indices within the event data
        start_samples = int(window_start_offset * sampling_rate)
        end_samples = int(window_end_offset * sampling_rate)

        # Extract windowed data
        windowed_data = {}
        for key, values in event_data.items():
            if isinstance(values, list) and len(values) > 0:
                start_safe = max(0, min(start_samples, len(values) - 1))
                end_safe = max(start_safe + 1, min(end_samples, len(values)))
                windowed_data[key] = values[start_safe:end_safe]
            else:
                windowed_data[key] = []

        # Calculate windowed metrics
        windowed_metrics = _calculate_event_metrics(
            windowed_data, sampling_rate, window_start_time, window_end_time
        )

        # Create windowed event record
        windowed_event = base_event.copy()
        windowed_event.update({
            'Classification': f"{base_event['Classification']}_window_{i+1}",
            'name': f"{base_event['name']}_window_{i+1}",
            'Start_Time': window_start_time,
            'End_Time': window_end_time,
            'duration_seconds': window_length,
            'window_index': i + 1,
            'parent_event': base_event['name'],
            **windowed_metrics
        })

        windowed_events.append(windowed_event)

    return windowed_events
