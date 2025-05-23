import h5py
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import json
import re
import sys
import neurokit2 as nk
import polars as pl
import warnings
import time as ti
warnings.filterwarnings('ignore')




def main(param):
    print('This is a message from the function main')
    project = param

    

    fibro_hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware.hdf5')

    

    if not os.path.exists(fibro_hdf_file):

        print(f'File {fibro_hdf_file} not found', file=sys.stderr)
        ti.sleep(20)
        return

    with h5py.File(fibro_hdf_file, 'r') as file:
        subject_list = [group for group in file.keys() if re.search(r'_(\d{3})$', group)]
        file.close()
    # subject_list = [group for group in file.keys() if re.search(r'^sub_', group)]
    # file.close()
    total_subjects = len(subject_list)
    results_list = {}
    print(f'Total subjects: {total_subjects}')


    # print('Loading yellow rpeaks', file=sys.stderr)

    metrics_pl_df = pl.DataFrame()

    tqdm_subjects = tqdm(subject_list, file=sys.stdout, desc='Subjects')


    for subject in tqdm_subjects:
        tqdm_subjects.set_description(f'Subjects: {subject}')
        
        current_subject_index = subject_list.index(subject)
        group = subject_list[current_subject_index]

        fibro_hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
        fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware.hdf5')

        if not os.path.exists(fibro_hdf_file):
            print(f'File {fibro_hdf_file} not found', file=sys.stderr)
            ti.sleep(20)
            return

        file = h5py.File(fibro_hdf_file, 'r+')
        events_data = json.loads(file[group]['data'].attrs['events_data'])
        file.close()
        # print(f'subject found: {group}')
        sub_id = group
        # print('Loading yellow rpeaks')

        ecg = False
        gsc = False

        file = h5py.File(fibro_hdf_file, 'r+')
        if 'rpeaks' not in file[sub_id].keys():
            ecg = False
            # print(f'No rpeaks for {sub_id}')
        else:
            ecg = True
            yellow_rpeaks = json.loads(file[sub_id]['rpeaks'].attrs['signal_attributes'])['yellow_peaks']
        
        if 'gsc_attributes' not in file[sub_id]['data'].attrs.keys():
            gsc = False
            # print(f'No gsc for {sub_id}')
        else:
            gsc = True
            gsc_attributes = json.loads(file[sub_id]['data'].attrs['gsc_attributes'])
        
        file.close()



        def find_closest_indices(array, values):
            array = np.asarray(array)
            indices = np.searchsorted(array, values)
            indices = np.clip(indices, 1, len(array) - 1)
            left = array[indices - 1]
            right = array[indices]
            indices -= values - left < right - values
            return indices

        dataset = pd.read_hdf(fibro_hdf_file, key=sub_id + '/data')
        # print('Datasets loaded')

        dataset_df = pd.DataFrame(dataset)
        
        time = list(dataset_df['time'])

        # print('Loading datasets')
        if ecg:
            rpeaks = pd.read_hdf(fibro_hdf_file, key=sub_id + '/rpeaks')
            rpeaks_df = pd.DataFrame(rpeaks)
            rpeaks = [int(i) for i in rpeaks_df[0]]
            rpeaks_time = [time[i] for i in rpeaks]
            yellow_rpeaks_mapped = [rpeaks.index(i) for i in yellow_rpeaks if i in rpeaks]


        if gsc:
            gsc_signal = list(dataset_df['gsc'])
            


        

        

        cont = 0
        for event in events_data.keys():
            event_data = events_data[event]
            start_time = event_data['Start_Time']
            end_time = event_data['End_Time']
            event_name = event_data['Classification']

            original_duration = end_time - start_time

            time = np.array(dataset_df['time'])

            if ecg:
                rpeaks_timing = np.array([time[i] for i in rpeaks])

                differences_start_time_rpeaks = np.abs(rpeaks_timing - start_time)
                min_index_start_rpeaks = np.argmin(differences_start_time_rpeaks)
                closest_start_time_rpeaks = rpeaks_timing[min_index_start_rpeaks]
                start_index_rpeaks = np.where(rpeaks_timing == closest_start_time_rpeaks)[0][0]

                differences_end_time_rpeaks = np.abs(rpeaks_timing - end_time)
                min_index_end_rpeaks = np.argmin(differences_end_time_rpeaks)
                closest_end_time_rpeaks = rpeaks_timing[min_index_end_rpeaks]
                end_index_rpeaks = np.where(rpeaks_timing == closest_end_time_rpeaks)[0][0]

                if start_index_rpeaks == end_index_rpeaks:
                    continue

                rpeaks_timing = rpeaks_timing[start_index_rpeaks:end_index_rpeaks]
                new_rpeaks_times = (rpeaks_timing.copy()).tolist()
                new_rpeaks_indices_b = rpeaks[start_index_rpeaks:end_index_rpeaks]
                updated_duration = False

                for i in yellow_rpeaks_mapped:
                    if i != 0 or i != len(rpeaks_timing) - 1:
                        if i > start_index_rpeaks and i < end_index_rpeaks:
                            i = i - start_index_rpeaks
                            if i -1 >= 0 and i + 1 < len(rpeaks_timing):
                                updated_duration = True
                                previous_interval = rpeaks_timing[i] - rpeaks_timing[i - 1]
                                next_interval = rpeaks_timing[i + 1] - rpeaks_timing[i]
                                combined_interval = previous_interval + next_interval

                                duration = original_duration - combined_interval

                                if i + 1 < len(new_rpeaks_times):
                                    new_rpeaks_times.pop(i + 1)
                                    new_rpeaks_indices_b.pop(i + 1)

                                if i < len(new_rpeaks_times):
                                    new_rpeaks_indices_b.pop(i)
                                    new_rpeaks_times.pop(i)

                                    for j in range(i, len(new_rpeaks_times)):
                                        new_rpeaks_times[j] -= combined_interval
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                
                if len(yellow_rpeaks_mapped) == 0:
                    duration = original_duration


                if not updated_duration:
                    duration = original_duration

                new_rpeaks_indices = find_closest_indices(time, new_rpeaks_times)
                new_rpeaks_dict = {
                    'ECG_R_Peaks' : new_rpeaks_indices
                }

                try:
                    hrv_time = nk.hrv_time(new_rpeaks_dict, sampling_rate=500)
                    hrv_frequency = nk.hrv_frequency(new_rpeaks_dict, sampling_rate=500, normalize=False,
                                                        psd_method='welch', interpolation_rate=100)
                    hrv_nonlinear = nk.hrv_nonlinear(new_rpeaks_dict, sampling_rate=500)
                except ValueError as e:
                    # print(f'Error in {sub_id} - {event_name}: {e}')
                    continue

                RR_mean = np.mean(np.diff(new_rpeaks_times))
                RR_std = np.std(np.diff(new_rpeaks_times))
                RR_min = np.min(np.diff(new_rpeaks_times))
                RR_max = np.max(np.diff(new_rpeaks_times))
                HR_mean = 60 / RR_mean
                HR_std = 60 / RR_std
                HR_min = 60 / RR_max
                HR_max = 60 / RR_min

                heart_rate = pd.DataFrame({
                    'HR_mean': HR_mean,
                    'HR_std': HR_std,
                    'HR_min': HR_min,
                    'HR_max': HR_max,
                    'RR_mean': RR_mean,
                    'RR_std': RR_std,
                    'RR_min': RR_min,
                    'RR_max': RR_max
                }, index=[0])
            
            else:
                hrv_time = None
                hrv_frequency = None
                hrv_nonlinear = None
                heart_rate = None
                duration = original_duration

            if gsc:
                gsc_data = pd.DataFrame({
                    'GSC_mean': np.mean(gsc_signal),
                }, index=[0])
            else:
                gsc_data = pd.DataFrame({
                    'GSC_mean': np.nan,
                }, index=[0])
            
            metrics = pd.concat([heart_rate, hrv_time, hrv_frequency, hrv_nonlinear, gsc_data], axis=1)
            


            metrics['Subject'] = sub_id 
            metrics['Event'] = event_name
            metrics['Original Duration'] = original_duration
            metrics['New Duration'] = duration

            
            # Sort the columns
            if gsc and ecg:
                heart_rate_columns = [col for col in heart_rate.columns if col in metrics.columns]
                hrv_time_columns = [col for col in hrv_time.columns if col in metrics.columns]
                hrv_frequency_columns = [col for col in hrv_frequency.columns if col in metrics.columns]
                hrv_nonlinear_columns = [col for col in hrv_nonlinear.columns if col in metrics.columns]
                gsc_columns = [col for col in gsc_data.columns if col in metrics.columns]
            elif ecg:
                heart_rate_columns = [col for col in heart_rate.columns if col in metrics.columns]
                hrv_time_columns = [col for col in hrv_time.columns if col in metrics.columns]
                hrv_frequency_columns = [col for col in hrv_frequency.columns if col in metrics.columns]
                hrv_nonlinear_columns = [col for col in hrv_nonlinear.columns if col in metrics.columns]
                gsc_columns = []
            elif gsc:
                heart_rate_columns = []
                hrv_time_columns = []
                hrv_frequency_columns = []
                hrv_nonlinear_columns = []
                gsc_columns = [col for col in gsc_data.columns if col in metrics.columns]
            
            metrics['duration_diff'] = metrics['Original Duration'] - metrics['New Duration']

            if hrv_time is None and gsc_columns == []:
                metrics = metrics[['Subject', 'Event', 'duration_diff','Original Duration', 'New Duration']]
            else:
                metrics = metrics[['Subject', 'Event', 'duration_diff','Original Duration', 'New Duration'] + heart_rate_columns + hrv_time_columns + hrv_frequency_columns + hrv_nonlinear_columns+ gsc_columns]

            


            # Add metrics to results list dictionary
            if sub_id not in results_list:
                results_list[sub_id] = {}

            results_list[sub_id][cont] = metrics.to_dict('records')
            metrics = pl.DataFrame(metrics)
            metrics_pl_df = pl.concat([metrics_pl_df, metrics], how='diagonal')
            cont += 1
        


    fibro_hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware.hdf5')

    if not os.path.exists(fibro_hdf_file):
        print(f'File {fibro_hdf_file} not found', file=sys.stderr)
        ti.sleep(20)
        return

    metrics_pl_df = metrics_pl_df.with_row_index(name='Index')

    old_results_parquet = fibro_hdf_file.replace('.hdf5', '_results.parquet')
    
    if not os.path.exists(old_results_parquet):
        if 'results' in file.keys():
            old_results = pd.read_hdf(fibro_hdf_file, key='/results')
            old_results = pl.DataFrame(old_results)
        else:
            old_results = None

        if old_results is not None:
            old_results = (
                old_results
                .select(
                    pl.all().name.suffix('_old')
                )
                .join(
                    metrics_pl_df,
                    left_on=['Subject_old', 'Event_old'],
                    right_on=['Subject', 'Event'],
                    how='left')

            )
            old_columns = [col for col in old_results.columns if col.endswith('_old')]

            for col in old_columns:
                new_col = col.replace('_old', '')
                if new_col in old_results.columns:
                    old_results = (
                        old_results
                        .with_columns(
                            pl.col(new_col).fill_null(pl.col(col))
                        )
                    )
           
        else:
            old_results = metrics_pl_df
    else:
        old_results = pl.read_parquet(old_results_parquet)

        print('Old results loaded')
        
    results_df = pl.concat([old_results, metrics_pl_df], how='diagonal')    

    
    results_df.write_parquet(old_results_parquet)


if __name__ == '__main__':

    try:
        param = sys.argv[1]
    except IndexError:
        param = 'fibro'


    main(param)