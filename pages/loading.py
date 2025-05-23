from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
import os
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import pickle as pkl
import h5py
import numpy as np
import webview
# import dash_core_components as dcc
from flask import Flask
import neurokit2 as nk
import base64
import json
import io
import re
from pymongo import MongoClient
import dash

dash.register_page(__name__, name='Loading', order =2)

pages = {}

for page in os.listdir(r"C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates"):
    page_name = page.split("_")[0]
    page_value = page.split("_")[0]

    pages[page_name] = page_value


layout = html.Div(
    [
        html.Div("Subject Data Uploading", style={"fontSize": 40, 'textAlign': 'center'}),
        html.Hr(),
        html.Div("In this page you can upload the data of the subjects from the Google Drive folders or by uploading single data and events txt files.", style={"fontSize": 20, 'textAlign': 'center'}),
        html.Hr(),
        html.Div("Select Project", style={"fontSize": 20, 'textAlign': 'center'}),

        dcc.Dropdown(
            id='project-dropdown-loading',
            options=[
                {'label': key, 'value': value} for key, value in pages.items()
            ],
            style={"color": "black"},
            value='fibro'
        ),

        html.Div(id='project-markdown-loading', style={"fontSize": 20, 'textAlign': 'center'}),
        
        # Radio button for overwriting the data or not
        html.Div(
            dcc.RadioItems(
                id='overwrite-data',
                options=[
                    {'label': 'Append data', 'value': 'append'},
                    {'label': 'Skip data', 'value': 'skip'}
                ],
                value='append',
                labelStyle={'display': 'inline-block'}
            )
        ),

        dbc.Container(
            [
            html.Div("Upload the data file", style={"fontSize": 20, 'textAlign': 'center'}),
            
            html.Hr(),

            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
                )
            ,
            

            html.Div("Upload the events file", style={"fontSize": 20, 'textAlign': 'center'}),
            dcc.Upload(
                id='upload-event',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),

            dcc.Loading(
                [dcc.ConfirmDialog(
                id='confirm',
                message='',
                )],

                overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white", "filter": "blur(2px)"},
                custom_spinner=html.H2(["Uploading subject data...", dbc.Spinner(color="primary")])
                )

            # dbc.Button("Submit", id="submit", color="primary", className="mr-1", style={"margin": "10px"}),

            
            ]
        ),
        dbc.Container(
            [
                html.Div("Check for updates in the Google Drive folders", style={"fontSize": 20, 'textAlign': 'center'}),
                html.Hr(),
                dbc.Button("Check for updates", id="check-updates", color="primary", className="mr-1", style={"margin": "10px"}),
                dcc.Loading(
                    [dcc.ConfirmDialog(
                        id='update-confirm',
                        message='',
                    )],
                    overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white", "filter": "blur(2px)"},
                    custom_spinner=html.H2(["Checking for updates...", dbc.Spinner(color="primary")])
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Subjects Found"),
                        dbc.ModalBody(
                            [
                                dcc.Checklist(id='subjects-checklist'),
                                dbc.Button("Load Selected Subjects", id='load-selected-subjects', color="primary", className="mr-1", style={"margin": "10px"}),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-modal", className="ml-auto")
                        ),
                    ],
                    id="modal",
                    is_open=False,
                ),
                dcc.ConfirmDialog(
                    id='no-new-subjects-dialog',
                    message='No new subjects found.',
                )
            ]
            
        ),

        dcc.Store(id='uploaded-data-store'),
        dcc.Store(id='uploaded-event-store'),
        dcc.Store(id='subjects-to-load-store')
    ]
)

@callback(
    [Output('project-markdown-loading', 'children'),
    Output('confirm', 'message'),    
    Output('confirm', 'displayed'),
    Output('upload-data', 'contents'),
    Output('upload-event', 'contents')],
    [Input('overwrite-data', 'value'),
    Input('project-dropdown-loading', 'value'),
    Input('upload-data', 'contents'),
    Input('upload-event', 'contents')],
    [State('upload-data', 'filename'),
    State('upload-event', 'filename')],
    prevent_initial_call=True
)
def display_confirm_dialog(mode, project, contents_data, contents_events, filename_data, filename_events):

    if not contents_data or not contents_events:
        raise PreventUpdate

    if contents_data and contents_events:
        print(f"Mode: {mode}")

        events_file_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
        events_file_path = os.path.join(events_file_path, f'{project}_mindware.parquet')

        if os.path.exists(events_file_path):
            events_full_df = pd.read_parquet(events_file_path)
            print(f"Events file loaded successfully")
        else:
            events_full_df = pd.DataFrame()
            print(f"Events file not found")


        def detect_events(event_list, events_df, new_events_data, skip_indices, blocks_shorts=None):
            for event_key, event in event_list:
                start_flag = event['start']
                end_flag = event['end']
                classification = event['name']
                events_df['Event Type'] = [e.split(":")[1] if ":" in e else e for e in events_df['Event Type']]
                if blocks_shorts is not None:
                    events_df['Name'] = [blocks_shorts[e] if e in blocks_shorts.keys() else e for e in events_df['Name']]
                    events_df['Event Type'] = [events_df.loc[idx, 'Name'] if re.search('UDP', e) else e for idx, e in enumerate(events_df['Event Type'])]
                start_indices = events_df[(events_df['Event Type'] == start_flag) & (~events_df.index.isin(skip_indices))].index.tolist()
                end_indices = events_df[(events_df['Event Type'] == end_flag) & (~events_df.index.isin(skip_indices))].index.tolist()
                start_indices.sort()
                end_indices.sort()
                for start_idx in start_indices:
                    for end_idx in end_indices:
                        if start_idx < end_idx:
                            start_time = events_df.loc[start_idx, 'Time']
                            end_time = events_df.loc[end_idx, 'Time']
                            if not any((e['Start_Flag'] == start_flag and e['End_Flag'] == end_flag and
                                        e['Start_Time'] == start_time and e['End_Time'] == end_time) for e in new_events_data):
                                new_events_data.append({
                                    'Classification': classification,
                                    'Start_Flag': start_flag,
                                    'End_Flag': end_flag,
                                    'Start_Time': start_time,
                                    'End_Time': end_time
                                })
                            break
        
        def mark_special_events(events_df, events_mapping, special_type):
            special_events = [event for event in events_mapping.items() if event[1].get(special_type)]
            special_flags = set()
            for _, event in special_events:
                start_flag = event['start']
                end_flag = event['end']
                special_flags.add((start_flag, end_flag))
            marked_indices = []
            flagged_events_count = {}
            for start_flag, end_flag in special_flags:
                start_indices = events_df[events_df['Event Type'] == start_flag].index.tolist()
                end_indices = events_df[events_df['Event Type'] == end_flag].index.tolist()
                start_indices.sort()
                end_indices.sort()
                pairs_count = min(len(start_indices), len(end_indices))
                flagged_events_count[(start_flag, end_flag)] = pairs_count
                for start_idx in start_indices:
                    for end_idx in end_indices:
                        if start_idx < end_idx:
                            marked_indices.append((start_idx, end_idx))
                            break
            adjusted_indices = []
            for (start_flag, end_flag), count in flagged_events_count.items():
                mapping_count = sum([1 for event in events_mapping.items() if event[1].get(special_type) and event[1]['start'] == start_flag and event[1]['end'] == end_flag])
                regular_event_count = count - mapping_count
                if regular_event_count < 0:
                    return []
                if regular_event_count > 0:
                    events_mapping_order = [event[0] for event in events_mapping.items() if event[1]['start'] == start_flag and event[1]['end'] == end_flag]
                    events_mapping_order = [event.split("_")[1] for event in events_mapping_order]
                    regular_event_number = [event[0] for event in events_mapping.items() if event[1]['start'] == start_flag and event[1]['end'] == end_flag and not event[1].get(special_type)]
                    regular_event_number = [event.split("_")[1] for event in regular_event_number]
                    events_mask = [1 if event in regular_event_number else 0 for event in events_mapping_order]
                    for idx, (start_idx, end_idx) in enumerate(marked_indices):
                        if events_mask[idx] == 0:
                            adjusted_indices.append(start_idx)
                            adjusted_indices.append(end_idx)
            return adjusted_indices
        
        if project:
            with open(f'pages/templates/{project}_template.json') as f:
                json_data = json.load(f)

                sampling_rate = json_data['parameters']['sampling_rate']
                events_mapping = json_data['events']
                channels = json_data['channel']
                if json_data.get('blocks_shorts'):
                    blocks_shorts = json_data['blocks_shorts']
                else:
                    blocks_shorts = None

                # close the file
                f.close()

            print("Project template loaded successfully")


            try:
                loaded_data_message = f'{filename_data} has been successfully loaded'
                loaded_events_message = f'{filename_events} has been successfully loaded'

                data_content_type, data_content_string = contents_data.split(',')
                # print(f"Data content type: {data_content_type}")
                events_content_type, events_content_string = contents_events.split(',')
                decoded_data = base64.b64decode(data_content_string)
                decoded_events = base64.b64decode(events_content_string)

                signal_df = pd.read_csv(io.StringIO(decoded_data.decode('utf-8')), sep='\t', skiprows=1)
                print("Data file loaded successfully")
                events_df = pd.read_csv(io.StringIO(decoded_events.decode('utf-8')), sep='\t')
                print("Events file loaded successfully")

                data_df = signal_df

                events = events_mapping

                signal = data_df.filter(regex='Bio').values.flatten()
                time = data_df['Time (s)'].values.flatten()

                x_axis = data_df.filter(regex='X Axis').values.flatten()
                y_axis = data_df.filter(regex='Y Axis').values.flatten()

                try:
                    z_axis = data_df.filter(regex='Z Axis').values.flatten()
                    x_axis = [x**2 for x in x_axis]
                    y_axis = [y**2 for y in y_axis]
                    z_axis = [z**2 for z in z_axis]
                    motion = np.sqrt(x_axis + y_axis + z_axis)
                    motion = [m * 0.01 for m in motion]
                    # Downsample the motion data to the length of the signal every 2 samples
                    if len(motion) / len(signal) == 3:
                        motion = motion[::3]
                    elif len(motion) / len(signal) == 2:
                        motion = motion[::2]

                    print(f"Motion data loaded successfully")
                except KeyError:
                    x_axis = [x**2 for x in x_axis]
                    y_axis = [y**2 for y in y_axis]

                    motion = np.sqrt(x_axis + y_axis)   
                    motion = [m * 0.01 for m in motion]
                    # Downsample the motion data to the length of the signal every 2 samples
                    if len(motion) / len(signal) == 3:
                        motion = motion[::3]
                    elif len(motion) / len(signal) == 2:
                        motion = motion[::2]
                    print(f"Motion data loaded successfully")


                events_df['Event Type'] = [x.split(':')[1] for x in events_df['Event Type']]

                binah_marked_indices = mark_special_events(events_df, events, 'Binah')
                print(f"Binah marked indices: {binah_marked_indices}")
                korro_marked_indices = mark_special_events(events_df, events, 'Korro')
                print(f"Korro marked indices: {korro_marked_indices}")

                new_events_data = []

                non_special_events = [event for event in events.items() if not event[1].get('Binah') and not event[1].get('Korro')]
                print(f"Non special events: {non_special_events}")

                if binah_marked_indices or korro_marked_indices:
                    print(f"Start detecting special events")
                    detect_events(non_special_events, events_df, new_events_data, binah_marked_indices + korro_marked_indices,blocks_shorts)

                    binah_events = [event for event in events.items() if event[1].get('Binah')]
                    korro_events = [event for event in events.items() if event[1].get('Korro')]
                    detect_events(binah_events, events_df, new_events_data, [], blocks_shorts)
                    detect_events(korro_events, events_df, new_events_data, [], blocks_shorts)
                else:
                    detect_events(events.items(), events_df, new_events_data, [], blocks_shorts)

                
                new_events_df = pd.DataFrame(new_events_data)
                print(f"New events data: {new_events_df}")
                events_data = new_events_df.to_dict(orient='records')



                subject_folder = filename_data.split('.')[0]

                




                events_columns = [{'name': i, 'id': i, 'editable': True} for i in events_df.columns]
                [print(f'Column: {column}') for column in events_columns]

                if signal.size == 0:
                    print(f"No ecg signal found for subject {subject_folder}")

                    events_df['avg_motion'] = np.nan



                    # convert the events data to a dictionary
                    events_data = {event['Classification']: event for event in events_data}

                    for key, value in events_data.items():

                        print(f"Key: {key}, Value: {value}")
                        if 'Start_Time' not in value or 'End_Time' not in value:
                            continue
                        start_time = value['Start_Time']
                        end_time = value['End_Time']

                        time = np.array(time)

                        differences_start_time = np.abs(time - start_time)
                        min_index_start = np.argmin(differences_start_time)
                        closest_start_time = time[min_index_start]

                        start_index = np.where(time == closest_start_time)[0][0]

                        differences_end_time = np.abs(time - end_time)
                        min_index_end = np.argmin(differences_end_time)
                        closest_end_time = time[min_index_end]

                        end_index = np.where(time == closest_end_time)[0][0]

                        print(f"Start index: {start_index}, End index: {end_index}")

                        average_motion = round(np.mean(motion[start_index:end_index]), 2)
                        print(f"Average motion: {average_motion}")
                        events_data[key]['avg_motion'] = average_motion
                        print(f"Row {key} calculated successfully")

                
                else:

                    cleaned_signal = nk.ecg_clean(signal, sampling_rate=500)
                    print(f"Signal cleaned successfully")

                    info, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=500, correct_artifacts=True)
                    rpeaks_indices = rpeaks['ECG_R_Peaks'].tolist()

                    ectopic_beats = rpeaks['ECG_fixpeaks_ectopic']
                    longshort_beats = rpeaks['ECG_fixpeaks_longshort']
                    false_negatives = rpeaks['ECG_fixpeaks_missed']
                    false_positives = rpeaks['ECG_fixpeaks_extra']

                    quality = nk.ecg_quality(cleaned_signal, sampling_rate=500, rpeaks=rpeaks_indices)
                    print(f"Quality calculated successfully")

                    quality_rpeaks = quality[rpeaks_indices]

                    peaks_color = ['yellow' if quality_rpeak<0.6 else 'red' for quality_rpeak in quality_rpeaks]

                    # get the indices of the rpeaks that are not of good quality
                    rpeaks_indices_bad_quality = [rpeaks_indices[i] for i in range(len(quality_rpeaks)) if quality_rpeaks[i] < 0.6]

                    peaks_color_binary = [1 if quality_rpeak<0.6 else 0 for quality_rpeak in quality_rpeaks]

                    columns_to_add = ['avg_quality', 'avg_motion', 'corrected_rpeaks_percent']

                    for column in columns_to_add:
                        events_df[column] = np.nan

                    # convert the events data to a dictionary
                    events_data = {event['Classification']: event for event in events_data}

                    for key, value in events_data.items():

                        print(f"Key: {key}, Value: {value}")
                        if 'Start_Time' not in value or 'End_Time' not in value:
                            continue
                        start_time = value['Start_Time']
                        end_time = value['End_Time']

                        time = np.array(time)

                        differences_start_time = np.abs(time - start_time)
                        min_index_start = np.argmin(differences_start_time)
                        closest_start_time = time[min_index_start]

                        start_index = np.where(time == closest_start_time)[0][0]

                        differences_end_time = np.abs(time - end_time)
                        min_index_end = np.argmin(differences_end_time)
                        closest_end_time = time[min_index_end]

                        end_index = np.where(time == closest_end_time)[0][0]

                        print(f"Start index: {start_index}, End index: {end_index}")

                        rpeaks_time = [time[rpeak] for rpeak in rpeaks_indices]

                        rpeaks_time = np.array(rpeaks_time)

                        differences_start_time_rpeaks = np.abs(rpeaks_time - start_time)
                        min_index_start_rpeaks = np.argmin(differences_start_time_rpeaks)
                        closest_start_time_rpeaks = rpeaks_time[min_index_start_rpeaks]

                        start_index_rpeaks = np.where(rpeaks_time == closest_start_time_rpeaks)[0][0]

                        differences_end_time_rpeaks = np.abs(rpeaks_time - end_time)
                        min_index_end_rpeaks = np.argmin(differences_end_time_rpeaks)
                        closest_end_time_rpeaks = rpeaks_time[min_index_end_rpeaks]

                        end_index_rpeaks = np.where(rpeaks_time == closest_end_time_rpeaks)[0][0]

                        quality_value = round(np.mean(quality_rpeaks[start_index_rpeaks:end_index_rpeaks]), 2)
                        print(f"Quality value: {quality_value}")

                        average_motion = round(np.mean(motion[start_index:end_index]), 2)
                        print(f"Average motion: {average_motion}")

                        rpeaks_indices_range = [start_index_rpeaks, end_index_rpeaks]

                        ectopic_beats_value = len([peak for peak in ectopic_beats if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
                        longshort_beats_value = len([peak for peak in longshort_beats if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
                        false_negatives_value = len([peak for peak in false_negatives if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
                        false_positives_value = len([peak for peak in false_positives if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
                        
                        artifacts = len([peak for peak in peaks_color_binary[start_index_rpeaks:end_index_rpeaks] if peak == 1])

                        if artifacts > 0 and len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks]) != 0:
                            artifacts_percent = round((artifacts / len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks])) * 100, 2)
                        else:
                            artifacts_percent = 0.00


                        corrected_rpeaks = ectopic_beats_value + longshort_beats_value + false_negatives_value + false_positives_value

                        if corrected_rpeaks > 0:
                            print(f"!!!!!!!!!!!!!!!!!!!!!Currected rpeaks: {corrected_rpeaks}!!!!!!!!!!!!!!!!!!!!!!!!!!")

                        if len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks]) != 0:
                            corrected_rpeaks_percent = round((corrected_rpeaks / len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks])) * 100, 2)
                        else:
                            corrected_rpeaks_percent = 0

                        print(f"Corrected rpeaks: {corrected_rpeaks_percent}")

                        events_data[key]['avg_quality'] = quality_value
                        events_data[key]['artifacts_percent'] = artifacts_percent
                        events_data[key]['avg_motion'] = average_motion
                        events_data[key]['corrected_rpeaks_percent'] = corrected_rpeaks_percent
                        print(f"Row {key} calculated successfully")

                if 'GSC' in json_data['channel'] and data_df.filter(regex='GSC').values.any():
                    gsc = data_df.filter(regex='GSC').values.flatten()
                    
                    gsc_cleaned = nk.eda_clean(gsc, sampling_rate=500)
                    print(f"GSC cleaned successfully")

                    gsc_phasic = nk.eda_phasic(gsc_cleaned, sampling_rate=500, method='highpass')
                    print(f"GSC phasic calculated successfully")

                    _, gsc_peaks = nk.eda_peaks(gsc_cleaned, sampling_rate=500)
                    print(f"GSC peaks calculated successfully")

                    if signal.size != 0:


                        stored_signal = {
                            'signal': signal,
                            'gsc': gsc,
                            'eda_tonic': gsc_phasic['EDA_Tonic'].to_list(),
                            'eda_phasic': gsc_phasic['EDA_Phasic'].to_list(),
                            'time': time,
                            'motion': motion
                        }

                        signal_attributes = {
                            'rpeaks': rpeaks_indices,
                            'yellow_peaks': rpeaks_indices_bad_quality,
                            'ectopic_beats': ectopic_beats,
                            'longshort_beats': longshort_beats,
                            'false_negatives': false_negatives,
                            'false_positives': false_positives
                        }

                        gsc_attributes = {
                            'SCR_Onsets': gsc_peaks['SCR_Onsets'].tolist(),
                            'SCR_Peaks': gsc_peaks['SCR_Peaks'].tolist(),
                            'SCR_Height': gsc_peaks['SCR_Height'].tolist(),
                            'SCR_Amplitude': gsc_peaks['SCR_Amplitude'].tolist(),
                            'SCR_RiseTime': gsc_peaks['SCR_RiseTime'].tolist(),
                            'SCR_Recovery': gsc_peaks['SCR_Recovery'].tolist(),
                            'SCR_RecoveryTime': gsc_peaks['SCR_RecoveryTime'].tolist()
                        }

                        # signal_attributes.update(gsc_attributes)

                    else:

                        stored_signal = {
                            'gsc': gsc,
                            'eda_tonic': gsc_phasic['EDA_Tonic'].to_list(),
                            'eda_phasic': gsc_phasic['EDA_Phasic'].to_list(),
                            'time': time,
                            'motion': motion
                        }

                        gsc_attributes = {
                            'SCR_Onsets': gsc_peaks['SCR_Onsets'].tolist(),
                            'SCR_Peaks': gsc_peaks['SCR_Peaks'].tolist(),
                            'SCR_Height': gsc_peaks['SCR_Height'].tolist(),
                            'SCR_Amplitude': gsc_peaks['SCR_Amplitude'].tolist(),
                            'SCR_RiseTime': gsc_peaks['SCR_RiseTime'].tolist(),
                            'SCR_Recovery': gsc_peaks['SCR_Recovery'].tolist(),
                            'SCR_RecoveryTime': gsc_peaks['SCR_RecoveryTime'].tolist()
                        }

                        signal_attributes = {
                        }
                else:
                    stored_signal = {
                        'signal': signal,
                        'time': time,
                        'motion': motion
                    }

                    gsc_attributes = {
                    }

                    signal_attributes = {
                        'rpeaks': rpeaks_indices,
                        'yellow_peaks': rpeaks_indices_bad_quality,
                        'ectopic_beats': ectopic_beats,
                        'longshort_beats': longshort_beats,
                        'false_negatives': false_negatives,
                        'false_positives': false_positives
                    }
                for key, value in signal_attributes.items():
                    print(f"Key: {key}, Length: {len(value)}, Type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"Key: {key}, Shape: {value.shape}")

                for key, value in stored_signal.items():
                    print(f"Key: {key}, Length: {len(value)}, Type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"Key: {key}, Shape: {value.shape}")

                signal_dataset = pd.DataFrame(stored_signal)
                # subject_folder = filename_data.split('_')[0] + filename_data.split('_')[1]
                subject_folder = filename_data.split('_')[0] + '_' + filename_data.split('_')[1]
                subject_folder_name = '/' + subject_folder + '/data'
                print(f"Subject folder name: {subject_folder_name}")
                subject_group = f'{subject_folder}/data'
                rpeaks_group = f'{subject_folder}/rpeaks'


                file_name = f'pages/data/mindwareData/{project}_mindware.hdf5'

                    
                if not os.path.exists(file_name):
                    # Create a new HDF5 file for the project
                    signal_dataset.to_hdf(file_name, key=subject_group, mode='w', format='fixed', complevel=9, complib='blosc')

                    if signal_attributes != {}: 
                        rpeaks_dataset = pd.DataFrame(rpeaks_indices)

                        rpeaks_dataset.to_hdf(file_name, key=rpeaks_group, mode='w', format='fixed', complevel=9, complib='blosc')

                    with pd.HDFStore(file_name, mode='a') as store:
                        data_group = store[f'{subject_folder}/data']

                        if signal_attributes != {}:
                            signal_attributes.pop('rpeaks')
                            store.get_storer(f'{rpeaks_group}').attrs.signal_attributes = json.dumps(signal_attributes)
                    
                        if gsc_attributes != {}:
                            store.get_storer(f'{subject_group}').gsc_attributes = json.dumps(gsc_attributes)

                        store.get_storer(f'{subject_group}').attrs.events_data = json.dumps(events_data)

                        store.close()

                        message, df = save_events_data(file_name, subject_folder, project, 'overwrite')

                        if df is not None:
                                events_full_df = pd.concat([events_full_df, df], ignore_index=True)
                                events_full_df.to_parquet(events_file_path, index=False)
                
                else:
                    

                    if mode == 'overwrite':
                        print(f"Mode: {mode}")
                        
                        # Search for the subject group in the HDF5 file
                        with pd.HDFStore(file_name, mode='a') as store:
                            print(f"{subject_folder_name}")
                            print(store.keys())
                            # Check if the subject group already exists in the HDF5 file if it does, delete it
                            if subject_folder_name in store.keys():
                                del store[subject_folder]
                                print(f"Subject group {subject_group} removed successfully")
                            else:
                                print(f"Subject group {subject_group} does not exist in the HDF5 file")
                            store.close()

                        signal_dataset.to_hdf(file_name, key=subject_group, mode='a', format='fixed', complevel=9, complib='blosc')

                        if signal_attributes != {}:
                            rpeaks_dataset = pd.DataFrame(rpeaks_indices)

                            rpeaks_dataset.to_hdf(file_name, key=rpeaks_group, mode='a', format='fixed', complevel=9, complib='blosc')

                        with pd.HDFStore(file_name, mode='a') as store:
                        
                            data_group = store[f'{subject_folder}/data']

                        
                            # store.get_storer(f'{subject_group}').attrs.signal = json.dumps(stored_signal)
                        
                            
                        
                            if signal_attributes != {}:
                                # store.get_storer(f'{subject_group}').rpeaks = json.dumps(signal_attributes['rpeaks'])
                                
                                signal_attributes.pop('rpeaks')
                                store.get_storer(f'{rpeaks_group}').attrs.signal_attributes = json.dumps(signal_attributes)
                        
                            if gsc_attributes != {}:
                                store.get_storer(f'{subject_group}').gsc_attributes = json.dumps(gsc_attributes)

                            store.get_storer(f'{subject_group}').attrs.events_data = json.dumps(events_data)

                            store.close()

                            message, df = save_events_data(file_name, subject_folder,project, 'overwrite')
                    
                            if df is not None:
                                events_full_df = pd.concat([events_full_df, df], ignore_index=True)
                                events_full_df.to_parquet(events_file_path, index=False)
                            

                    elif mode == 'append':
                        print(f"Mode: {mode}")


                        signal_dataset.to_hdf(file_name, key=subject_group, mode='a', format='fixed', complevel=9, complib='blosc')

                        if signal_attributes != {}:
                            rpeaks_dataset = pd.DataFrame(rpeaks_indices)

                            rpeaks_dataset.to_hdf(file_name, key=rpeaks_group, mode='a', format='fixed', complevel=9, complib='blosc')

                        # Save the stored signal to the group of the subject in the HDF5 file
                        with pd.HDFStore(file_name, mode='a') as store:
                        
                            data_group = store[f'{subject_folder}/data']

                        
                            # store.get_storer(f'{subject_group}').attrs.signal = json.dumps(stored_signal)
                        
                            
                        
                            if signal_attributes != {}:
                                # store.get_storer(f'{subject_group}').rpeaks = json.dumps(signal_attributes['rpeaks'])
                                signal_attributes.pop('rpeaks')
                                store.get_storer(f'{rpeaks_group}').attrs.signal_attributes = json.dumps(signal_attributes)
                        
                            if gsc_attributes != {}:
                                store.get_storer(f'{subject_group}').gsc_attributes = json.dumps(gsc_attributes)

                            store.get_storer(f'{subject_group}').attrs.events_data = json.dumps(events_data)

                            store.close()

                            message, df = save_events_data(file_name, subject_folder,project, 'append')

                            if df is not None:
                                events_full_df = pd.concat([events_full_df, df], ignore_index=True)
                                events_full_df.to_parquet(events_file_path, index=False)
                    
                    elif mode == 'skip':
                        print(f"Mode: {mode}")
                        print(file_name)
                        print(subject_group)
                        subject_group = f'/{subject_group}'
                        # Check if the subject group already exists in the HDF5 file
                        with pd.HDFStore(file_name, mode='r') as store:
                            print('Checking if subject group exists in the HDF5 file')
                            if subject_group in store.keys():
                                message = f'Data for {subject_folder} already exists in {project}_mindware.hdf5'
                                store.close()
                                print('Subject group already exists in the HDF5 file')
                                return json_data,  message, True, None, None
                            else:
                                store.close()
                        
                        signal_dataset.to_hdf(file_name, key=subject_group, mode='a', format='fixed', complevel=9, complib='blosc')
                        # Save the stored signal to the group of the subject in the HDF5 file
                        print('Data saved successfully')
                        if signal_attributes != {}:
                            print('Saving rpeaks')
                            rpeaks_dataset = pd.DataFrame(rpeaks_indices)

                            rpeaks_dataset.to_hdf(file_name, key=rpeaks_group, mode='a', format='fixed', complevel=9, complib='blosc')
                            
                            
                        with pd.HDFStore(file_name, mode='a') as store:
                            print('Saving signal attributes')
                            data_group = store[f'{subject_folder}/data']

                        
                            # store.get_storer(f'{subject_group}').attrs.signal = json.dumps(stored_signal)
                        
                            
                        
                            if signal_attributes != {}:
                                # store.get_storer(f'{subject_group}').rpeaks = json.dumps(signal_attributes['rpeaks'])
                                signal_attributes.pop('rpeaks')
                                store.get_storer(f'{rpeaks_indices}').attrs.signal_attributes = json.dumps(signal_attributes)
                                print('Signal attributes saved successfully')
                            if gsc_attributes != {}:
                                store.get_storer(f'{subject_group}').gsc_attributes = json.dumps(gsc_attributes)

                            store.get_storer(f'{subject_group}').attrs.events_data = json.dumps(events_data)
                            store.close()
                            message, df = save_events_data(file_name, subject_folder, project, 'skip')

                            if df is not None:
                                events_full_df = pd.concat([events_full_df, df], ignore_index=True)
                                events_full_df.to_parquet(events_file_path, index=False)


                
                    
            except Exception as e:
                message = f'Error saving data for {filename_data}'
                print(e)
                return json_data, message, True, None, None
            
            print(f"Data saved successfully to {project}_mindware.hdf5")
            return json_data, message, True, None, None


        else:
            return "Please select a project", "", False, None, None
        
    # else:
    #     return "Please upload both data and events files", "", False, None, None
    







@callback(
    Output('subjects-checklist', 'options'),
    Output('modal', 'is_open'),
    Output('no-new-subjects-dialog', 'displayed'),
    Input('check-updates', 'n_clicks'),
    State('project-dropdown-loading', 'value'),
    prevent_initial_call=True
)
def check_google_drive(n_clicks, project):
    if n_clicks is None:
        raise PreventUpdate
    
    # Function to check Google Drive folder for new subjects
    new_subjects = check_new_subjects_in_drive(project)

    print(f'New subjects: {new_subjects}')
    
    if not new_subjects:
        return [], False, True
    
    subject_options = [{'label': subject, 'value': subject} for subject in new_subjects]
    return subject_options, True, False





def check_new_subjects_in_drive(project):
        
    json_file_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates'
    json_file_name = f'{project}_template.json'

    print(f'JSON: {json_file_name}')

    if not os.path.exists(os.path.join(json_file_path, json_file_name)):
        return []
    
    json_data = json.load(open(os.path.join(json_file_path, json_file_name)))
    
    project_folder = json_data['input']['path']
    fibro_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    fibro_file = os.path.join(fibro_file, f'{project}_mindware.hdf5')

    print(f'Project folder: {project_folder}')

    if not os.path.exists(fibro_file):
        print(f'HDF5 file not found: {fibro_file}')
        
    


    if os.path.exists(fibro_file):
        print(f"Reading HDF5 file: {fibro_file}")
        file = h5py.File(fibro_file, 'r')
        subjects_in_hdf5 = [group for group in file.keys() if re.search(r'_(\d{3})$', group)]
        file.close()
    else:
        print(f"HDF5 file not found: {fibro_file}")
        subjects_in_hdf5 = []
        
    if os.path.exists(fibro_file.replace('.hdf5', '.parquet')):
        print(f"Reading parquet file: {fibro_file.replace('.hdf5', '.parquet')}")
        events_data_subjects_parquet = pd.read_parquet(fibro_file.replace('.hdf5', '.parquet'))

        events_data_subjects_parquet = events_data_subjects_parquet['Id'].values
        print(f"Events data subjects: {events_data_subjects_parquet}")
        print(f"Subjects in HDF5: {subjects_in_hdf5}")
        # drop None values from the list
        events_data_subjects_parquet = [sub for sub in events_data_subjects_parquet if sub is not None]
        subjects_in_hdf5 = [sub for sub in subjects_in_hdf5 if sub is not None]
        events_data_subjects = np.intersect1d(events_data_subjects_parquet, subjects_in_hdf5)
    else:
        print(f"Parquet file not found: {fibro_file.replace('.hdf5', '.parquet')}")
        events_data_subjects = []

    # with pd.HDFStore(fibro_file, mode='r') as store:
    #     if '/events_data' in store.keys():
    #         try:
    #             events_data_subjects = store['events_data']
    #             if isinstance(events_data_subjects, pd.DataFrame) and 'Id' in events_data_subjects.columns:
    #                 events_data_subjects = np.intersect1d(events_data_subjects['Id'].values, subjects_in_hdf5)
    #             else:
    #                 if fibro_file.replace('.hdf5', '.parquet') in os.listdir('pages/data'):
    #                     events_data_subjects_parquet = pd.read_parquet(fibro_file.replace('.hdf5', '.parquet'))
    #                     events_data_subjects_parquet = events_data_subjects_parquet['Id'].values
    #                     events_data_subjects = np.intersect1d(events_data_subjects_parquet, subjects_in_hdf5)
    #                 else:
    #                     events_data_subjects = []
    #                     subjects_in_hdf5 = []
                    
    #         except (KeyError, TypeError, ValueError) as e:
    #             print(f"Error reading 'events_data': {e}")
    #             if fibro_file.replace('.hdf5', '.parquet') in os.listdir('pages/data'):
    #                 events_data_subjects_parquet = pd.read_parquet(fibro_file.replace('.hdf5', '.parquet'))
    #                 events_data_subjects_parquet = events_data_subjects_parquet['Id'].tolist()
    #                 events_data_subjects = np.intersect1d(events_data_subjects_parquet, subjects_in_hdf5)
    #             else:
    #                 events_data_subjects = []
    #                 subjects_in_hdf5 = []
    #     else:
    #         if fibro_file.replace('.hdf5', '.parquet') in os.listdir('pages/data'):
    #             events_data_subjects_parquet = pd.read_parquet(fibro_file.replace('.hdf5', '.parquet'))
    #             events_data_subjects_parquet = events_data_subjects_parquet['Id'].tolist()
    #             events_data_subjects = np.intersect1d(events_data_subjects_parquet, subjects_in_hdf5)
    #         else:
    #             events_data_subjects = []
    #             subjects_in_hdf5 = []

    #     store.close()
    
    

    subjects_in_hdf5 = [sub for sub in subjects_in_hdf5 if sub in events_data_subjects]

    new_subjects = []

    # Search for subjects in the Google Drive folder
    for subject in os.listdir(project_folder):
        print(f'Subject: {subject}')
        if re.search(r'_(\d{3})$', subject):
            hrv_pattern = r'^HRV'
            mindware_pattern = r'^MINDWARE'

            if subject in subjects_in_hdf5:
                print(f'Subject {subject} already exists in the HDF5 file')
                continue

            for folder in os.listdir(os.path.join(project_folder, subject)):
                print(f'Folder: {folder}')

                if re.search(hrv_pattern, folder) or re.search(mindware_pattern, folder):
                    print(f'Folder found for {subject}')
                    HRV_folder_path = os.path.join(project_folder, subject, folder)
                    data_file, events_file = None, None

                    for data_file_name in os.listdir(HRV_folder_path):
                        print(f'Data file: {data_file_name}')
                        if data_file_name.endswith('data.txt'):
                            data_file_path = os.path.join(HRV_folder_path, data_file_name)
                            try:
                                data_df = pd.read_csv(data_file_path, sep='\t', skiprows=1)
                                data_file = data_file_path
                                print(f'Data file loaded successfully')
                            except pd.errors.EmptyDataError:
                                print(f'Empty data file for {subject}')
                                continue
                        if data_file_name.endswith('events.txt'):
                            events_file_path = os.path.join(HRV_folder_path, data_file_name)
                            events_df = pd.read_csv(events_file_path, sep='\t')
                            events_file = events_file_path

                    if data_file and events_file and subject not in subjects_in_hdf5:
                        new_subjects.append(subject)
                        break

    return new_subjects



@callback(
    Output('update-confirm', 'message'),
    Output('update-confirm', 'displayed'),
    Input('load-selected-subjects', 'n_clicks'),
    State('subjects-checklist', 'value'),
    State('project-dropdown-loading', 'value'),
    prevent_initial_call=True
)
def load_selected_subjects(n_clicks, selected_subjects, project):
    if n_clicks is None:
        raise PreventUpdate

    if not selected_subjects:
        return "No subjects selected for loading.", True
    
    # Process the selected subjects
    message = process_selected_subjects(selected_subjects, project)
    return message, True

def process_selected_subjects(selected_subjects, project):
    
    json_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates'
    json_file = os.path.join(json_file, f'{project}_template.json')

    if not os.path.exists(json_file):
        return "Please select a project"
    
    json_data = json.load(open(json_file))
    project_folder = json_data['input']['path']
    fibro_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    fibro_file = os.path.join(fibro_file, f'{project}_mindware.hdf5')
    
    events_df = pd.DataFrame()

    for subject in selected_subjects:
        subject_folder = os.path.join(project_folder, subject)
        for folder in os.listdir(subject_folder):
            if re.search(r'^(HRV|MINDWARE)', folder):
                HRV_folder_path = os.path.join(subject_folder, folder)
                data_file, events_file = None, None

                for data_file_name in os.listdir(HRV_folder_path):
                    if data_file_name.endswith('data.txt'):
                        data_file_path = os.path.join(HRV_folder_path, data_file_name)
                        try:
                            data_df = pd.read_csv(data_file_path, sep='\t', skiprows=1)
                            data_file = data_file_path
                        except pd.errors.EmptyDataError:
                            continue
                    if data_file_name.endswith('events.txt'):
                        events_file_path = os.path.join(HRV_folder_path, data_file_name)
                        events_df = pd.read_csv(events_file_path, sep='\t')
                        events_file = events_file_path

                if data_file and events_file:
                    # Process and save to HDF5 file
                    message, df = save_to_hdf5(fibro_file, subject, data_df, events_df, project)
                    
                    events_df = pd.concat([events_df, df], ignore_index=True)
                    print(message)
    

    save_parquet_file(fibro_file, selected_subjects, project, events_df)
    
    return "Selected subjects have been successfully loaded."

def detect_events(event_list, events_df, new_events_data, skip_indices, blocks_shorts=None):
    for event_key, event in event_list:
        start_flag = event['start']
        end_flag = event['end']
        classification = event['name']
        events_df['Event Type'] = [e.split(":")[1] if ":" in e else e for e in events_df['Event Type']]
        if blocks_shorts is not None:
            events_df['Name'] = [blocks_shorts[e] if e in blocks_shorts.keys() else e for e in events_df['Name']]
            events_df['Event Type'] = [events_df.loc[idx, 'Name'] if re.search('UDP', e) else e for idx, e in enumerate(events_df['Event Type'])]
        start_indices = events_df[(events_df['Event Type'] == start_flag) & (~events_df.index.isin(skip_indices))].index.tolist()
        end_indices = events_df[(events_df['Event Type'] == end_flag) & (~events_df.index.isin(skip_indices))].index.tolist()
        start_indices.sort()
        end_indices.sort()
        for start_idx in start_indices:
            for end_idx in end_indices:
                if start_idx < end_idx:
                    start_time = events_df.loc[start_idx, 'Time']
                    end_time = events_df.loc[end_idx, 'Time']
                    if not any((e['Start_Flag'] == start_flag and e['End_Flag'] == end_flag and
                                e['Start_Time'] == start_time and e['End_Time'] == end_time) for e in new_events_data):
                        new_events_data.append({
                            'Classification': classification,
                            'Start_Flag': start_flag,
                            'End_Flag': end_flag,
                            'Start_Time': start_time,
                            'End_Time': end_time
                        })
                    break
        
def mark_special_events(events_df, events_mapping, special_type):
    special_events = [event for event in events_mapping.items() if event[1].get(special_type)]
    special_flags = set()
    for _, event in special_events:
        start_flag = event['start']
        end_flag = event['end']
        special_flags.add((start_flag, end_flag))
    marked_indices = []
    flagged_events_count = {}
    for start_flag, end_flag in special_flags:
        start_indices = events_df[events_df['Event Type'] == start_flag].index.tolist()
        end_indices = events_df[events_df['Event Type'] == end_flag].index.tolist()
        start_indices.sort()
        end_indices.sort()
        pairs_count = min(len(start_indices), len(end_indices))
        flagged_events_count[(start_flag, end_flag)] = pairs_count
        for start_idx in start_indices:
            for end_idx in end_indices:
                if start_idx < end_idx:
                    marked_indices.append((start_idx, end_idx))
                    break
    adjusted_indices = []
    for (start_flag, end_flag), count in flagged_events_count.items():
        mapping_count = sum([1 for event in events_mapping.items() if event[1].get(special_type) and event[1]['start'] == start_flag and event[1]['end'] == end_flag])
        regular_event_count = count - mapping_count
        if regular_event_count < 0:
            return []
        if regular_event_count > 0:
            events_mapping_order = [event[0] for event in events_mapping.items() if event[1]['start'] == start_flag and event[1]['end'] == end_flag]
            events_mapping_order = [event.split("_")[1] for event in events_mapping_order]
            regular_event_number = [event[0] for event in events_mapping.items() if event[1]['start'] == start_flag and event[1]['end'] == end_flag and not event[1].get(special_type)]
            regular_event_number = [event.split("_")[1] for event in regular_event_number]
            events_mask = [1 if event in regular_event_number else 0 for event in events_mapping_order]
            for idx, (start_idx, end_idx) in enumerate(marked_indices):
                if events_mask[idx] == 0:
                    adjusted_indices.append(start_idx)
                    adjusted_indices.append(end_idx)
    return adjusted_indices

def save_to_hdf5(fibro_file, subject, data_df, events_df, project):

    templates_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates'
    project_template = f'{project}_template.json'
    with open(os.path.join(templates_path, project_template)) as f:
        json_data = json.load(f)

        events = json_data['events']
        sampling_rate = json_data['parameters']['sampling_rate']
        channels = json_data['channel']
        if json_data.get('blocks_shorts'):
            blocks_shorts = json_data['blocks_shorts']
        else:
            blocks_shorts = None

        f.close()

    signal = data_df.filter(regex='Bio').values.flatten()
    time = data_df['Time (s)'].values.flatten()

    x_axis = data_df.filter(regex='X Axis').values.flatten()
    y_axis = data_df.filter(regex='Y Axis').values.flatten()

    try:
        z_axis = data_df.filter(regex='Z Axis').values.flatten()
        x_axis = [x**2 for x in x_axis]
        y_axis = [y**2 for y in y_axis]
        z_axis = [z**2 for z in z_axis]
        motion = np.sqrt(x_axis + y_axis + z_axis)
        motion = [m * 0.01 for m in motion]
        if len(motion) / len(time) == 3:
            motion = motion[::3]
        elif len(motion) / len(time) == 2:
            motion = motion[::2]


        
        print(f"Motion data loaded successfully")
    except KeyError:
        x_axis = [x**2 for x in x_axis]
        y_axis = [y**2 for y in y_axis]

        motion = np.sqrt(x_axis + y_axis)   
        motion = [m * 0.01 for m in motion]
        motion = motion[::2]
        print(f"Motion data loaded successfully")

    events_df['Event Type'] = [x.split(':')[1] for x in events_df['Event Type']]
    binah_marked_indices = mark_special_events(events_df, events, 'Binah')
    korro_marked_indices = mark_special_events(events_df, events, 'Korro')

    new_events_data = []

    non_special_events = [event for event in events.items() if not event[1].get('Binah') and not event[1].get('Korro')]
    if binah_marked_indices or korro_marked_indices:
        detect_events(non_special_events, events_df, new_events_data, binah_marked_indices + korro_marked_indices, blocks_shorts)

        binah_events = [event for event in events.items() if event[1].get('Binah')]
        korro_events = [event for event in events.items() if event[1].get('Korro')]
        detect_events(binah_events, events_df, new_events_data, [], blocks_shorts)
        detect_events(korro_events, events_df, new_events_data, [], blocks_shorts)
    else:
        detect_events(events.items(), events_df, new_events_data, [], blocks_shorts)

    new_events_df = pd.DataFrame(new_events_data)
    events_data = new_events_df.to_dict(orient='records')

    events_columns = [{'name': i, 'id': i, 'editable': True} for i in events_df.columns]

    if signal.size == 0:
        print(f"No ecg signal found for subject {subject}")

        events_df['avg_motion'] = np.nan

        events_data = {event['Classification']: event for event in events_data}

        for key, value in events_data.items():
            if 'Start_Time' not in value or 'End_Time' not in value:
                continue
            start_time = value['Start_Time']
            end_time = value['End_Time']

            time = np.array(time)

            differences_start_time = np.abs(time - start_time)
            min_index_start = np.argmin(differences_start_time)
            closest_start_time = time[min_index_start]

            start_index = np.where(time == closest_start_time)[0][0]

            differences_end_time = np.abs(time - end_time)
            min_index_end = np.argmin(differences_end_time)
            closest_end_time = time[min_index_end]

            end_index = np.where(time == closest_end_time)[0][0]

            average_motion = round(np.mean(motion[start_index:end_index]), 2)
            events_data[key]['avg_motion'] = average_motion


    else:
        cleaned_signal = nk.ecg_clean(signal, sampling_rate=500)

        info, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=500, correct_artifacts=True)
        rpeaks_indices = rpeaks['ECG_R_Peaks'].tolist()

        ectopic_beats = rpeaks['ECG_fixpeaks_ectopic']
        longshort_beats = rpeaks['ECG_fixpeaks_longshort']
        false_negatives = rpeaks['ECG_fixpeaks_missed']
        false_positives = rpeaks['ECG_fixpeaks_extra']

        quality = nk.ecg_quality(cleaned_signal, sampling_rate=500)

        quality_rpeaks = quality[rpeaks_indices]

        peaks_color = ['yellow' if quality_rpeak<0.6 else 'red' for quality_rpeak in quality_rpeaks]

        rpeaks_indices_bad_quality = [rpeaks_indices[i] for i in range(len(quality_rpeaks)) if quality_rpeaks[i] < 0.6]

        peaks_color_binary = [1 if quality_rpeak<0.6 else 0 for quality_rpeak in quality_rpeaks]

        columns_to_add = ['avg_quality', 'avg_motion', 'corrected_rpeaks_percent']

        for column in columns_to_add:
            events_df[column] = np.nan

        events_data = {event['Classification']: event for event in events_data}

        for key, value in events_data.items():

            if 'Start_Time' not in value or 'End_Time' not in value:
                continue
            start_time = value['Start_Time']
            end_time = value['End_Time']

            time = np.array(time)

            differences_start_time = np.abs(time - start_time)
            min_index_start = np.argmin(differences_start_time)
            closest_start_time = time[min_index_start]

            start_index = np.where(time == closest_start_time)[0][0]

            differences_end_time = np.abs(time - end_time)
            min_index_end = np.argmin(differences_end_time)
            closest_end_time = time[min_index_end]

            end_index = np.where(time == closest_end_time)[0][0]

            rpeaks_time = [time[rpeak] for rpeak in rpeaks_indices]

            rpeaks_time = np.array(rpeaks_time)

            differences_start_time_rpeaks = np.abs(rpeaks_time - start_time)
            min_index_start_rpeaks = np.argmin(differences_start_time_rpeaks)
            closest_start_time_rpeaks = rpeaks_time[min_index_start_rpeaks]

            start_index_rpeaks = np.where(rpeaks_time == closest_start_time_rpeaks)[0][0]

            differences_end_time_rpeaks = np.abs(rpeaks_time - end_time)
            min_index_end_rpeaks = np.argmin(differences_end_time_rpeaks)
            closest_end_time_rpeaks = rpeaks_time[min_index_end_rpeaks]

            end_index_rpeaks = np.where(rpeaks_time == closest_end_time_rpeaks)[0][0]

            quality_value = round(np.mean(quality_rpeaks[start_index_rpeaks:end_index_rpeaks]), 2)

            average_motion = round(np.mean(motion[start_index:end_index]), 2)

            rpeaks_indices_range = [start_index_rpeaks, end_index_rpeaks]

            ectopic_beats_value = len([peak for peak in ectopic_beats if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
            longshort_beats_value = len([peak for peak in longshort_beats if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
            false_negatives_value = len([peak for peak in false_negatives if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
            false_positives_value = len([peak for peak in false_positives if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])

            artifacts = len([peak for peak in peaks_color_binary[start_index_rpeaks:end_index_rpeaks] if peak == 1])

            if artifacts > 0 and len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks]) != 0:
                artifacts_percent = round((artifacts / len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks])) * 100, 2)
            else:
                artifacts_percent = 0.00

            corrected_rpeaks = ectopic_beats_value + longshort_beats_value + false_negatives_value + false_positives_value

            if corrected_rpeaks > 0:
                print(f"!!!!!!!!!!!!!!!!!!!!!Currected rpeaks: {corrected_rpeaks}!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks]) != 0:
                corrected_rpeaks_percent = round((corrected_rpeaks / len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks])) * 100, 2)
            else:
                corrected_rpeaks_percent = 0

            events_data[key]['avg_quality'] = quality_value
            events_data[key]['artifacts_percent'] = artifacts_percent
            events_data[key]['avg_motion'] = average_motion
            events_data[key]['corrected_rpeaks_percent'] = corrected_rpeaks_percent

    if 'GSC' in json_data['channel'] and data_df.filter(regex='GSC').values.any():
        gsc = data_df.filter(regex='GSC').values.flatten()
        
        gsc_cleaned = nk.eda_clean(gsc, sampling_rate=500)
        print(f"GSC cleaned successfully")

        gsc_phasic = nk.eda_phasic(gsc_cleaned, sampling_rate=500, method='highpass')
        print(f"GSC phasic calculated successfully")

        _, gsc_peaks = nk.eda_peaks(gsc_cleaned, sampling_rate=500)
        print(f"GSC peaks calculated successfully")

        if signal.size != 0:


            stored_signal = {
                'signal': signal,
                'gsc': gsc,
                'eda_tonic': gsc_phasic['EDA_Tonic'].to_list(),
                'eda_phasic': gsc_phasic['EDA_Phasic'].to_list(),
                'time': time,
                'motion': motion
            }

            signal_attributes = {
                'rpeaks': rpeaks_indices,
                'yellow_peaks': rpeaks_indices_bad_quality,
                'ectopic_beats': ectopic_beats,
                'longshort_beats': longshort_beats,
                'false_negatives': false_negatives,
                'false_positives': false_positives
            }

            gsc_attributes = {
                'SCR_Onsets': gsc_peaks['SCR_Onsets'].tolist(),
                'SCR_Peaks': gsc_peaks['SCR_Peaks'].tolist(),
                'SCR_Height': gsc_peaks['SCR_Height'].tolist(),
                'SCR_Amplitude': gsc_peaks['SCR_Amplitude'].tolist(),
                'SCR_RiseTime': gsc_peaks['SCR_RiseTime'].tolist(),
                'SCR_Recovery': gsc_peaks['SCR_Recovery'].tolist(),
                'SCR_RecoveryTime': gsc_peaks['SCR_RecoveryTime'].tolist()
            }

            # signal_attributes.update(gsc_attributes)

        else:

            stored_signal = {
                'gsc': gsc,
                'eda_tonic': gsc_phasic['EDA_Tonic'].to_list(),
                'eda_phasic': gsc_phasic['EDA_Phasic'].to_list(),
                'time': time,
                'motion': motion
            }

            gsc_attributes = {
                'SCR_Onsets': gsc_peaks['SCR_Onsets'].tolist(),
                'SCR_Peaks': gsc_peaks['SCR_Peaks'].tolist(),
                'SCR_Height': gsc_peaks['SCR_Height'].tolist(),
                'SCR_Amplitude': gsc_peaks['SCR_Amplitude'].tolist(),
                'SCR_RiseTime': gsc_peaks['SCR_RiseTime'].tolist(),
                'SCR_Recovery': gsc_peaks['SCR_Recovery'].tolist(),
                'SCR_RecoveryTime': gsc_peaks['SCR_RecoveryTime'].tolist()
            }

            signal_attributes = {
            }
    else:
        stored_signal = {
            'signal': signal,
            'time': time,
            'motion': motion
        }

        gsc_attributes = {
        }

        signal_attributes = {
            'rpeaks': rpeaks_indices,
            'yellow_peaks': rpeaks_indices_bad_quality,
            'ectopic_beats': ectopic_beats,
            'longshort_beats': longshort_beats,
            'false_negatives': false_negatives,
            'false_positives': false_positives
        }
    for key, value in signal_attributes.items():
        print(f"Key: {key}, Length: {len(value)}, Type: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"Key: {key}, Shape: {value.shape}")

    for key, value in stored_signal.items():
        print(f"Key: {key}, Length: {len(value)}, Type: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"Key: {key}, Shape: {value.shape}")

    signal_dataset = pd.DataFrame(stored_signal)
    subject_folder_name = '/' + subject + '/data'
    subject_group = f'{subject}/data'
    rpeaks_group = f'{subject}/rpeaks'

    file_name = f'pages/data/mindwareData/{project}_mindware.hdf5'

    with pd.HDFStore(file_name, mode='a') as store:
        # Check if the subject group already exists in the HDF5 file
        if subject_folder_name in store.keys():
            del store[subject]
            print(f"Subject group {subject_group} removed successfully")
            store.close()
        else:
            print(f"Subject group {subject_group} does not exist in the HDF5 file")
            store.close()

    signal_dataset.to_hdf(file_name, key=subject_group, mode='a', format='fixed', complevel=9, complib='blosc')

    if signal_attributes != {}:
        rpeaks_dataset = pd.DataFrame(rpeaks_indices)

        rpeaks_dataset.to_hdf(file_name, key=rpeaks_group, mode='a', format='fixed', complevel=9, complib='blosc')

    with pd.HDFStore(file_name, mode='a') as store:

        data_group = store[f'{subject}/data']

        if signal_attributes != {}:
            signal_attributes.pop('rpeaks')
            store.get_storer(f'{rpeaks_group}').attrs.signal_attributes = json.dumps(signal_attributes)

        if gsc_attributes != {}:
            store.get_storer(f'{subject_group}').gsc_attributes = json.dumps(gsc_attributes)

        store.get_storer(f'{subject_group}').attrs.events_data = json.dumps(events_data)

        store.close()

    finish_message, df = save_events_data(file_name, subject, project)

    return finish_message, df
    


def save_parquet_file(file_name, selected_subjects, project, events_df):
    
    events_df.to_parquet(file_name.replace('.hdf5', '.parquet'), index=False)
    print(f"events_data dataset saved successfully in {file_name}")
        

def save_events_data(file_name, subject, project, mode='append'):
    


    # if 'events_data' not in fibro_hdf.keys():
    #     if file_name.replace('.hdf5', '.parquet') in os.listdir('pages/data'):
    #         events_df = pd.read_parquet(file_name.replace('.hdf5', '.parquet'))

    #     else:
    #         print(f"Creating events_data dataset in {file_name}")
    #         events_df = pd.DataFrame(columns=['Id', 'event_name', 'done', 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality', 'duration(minutes)'])
    #         first_row = {'Id': 'Id', 'event_name': 'event_name', 'done': False, 'artifacts_percentage_%': 0.0, 'corrected_rpeaks_%': 0.0, 'avg_motion': 0.0, 'avg_quality': 0.0, 'duration(minutes)': 0.0}
    #         events_df = pd.concat([events_df, pd.DataFrame([first_row])])
    #         fibro_hdf.close()
    #         print(f"events_data dataset created successfully in {file_name}")
    #         # done_bool = events_df['done']
    #         # done_str = events_df['done'].astype(str)
    #         # events_df['done'] = done_str
    #         # events_df.to_hdf(file_name, key='events_data', mode='a', format='table')
    #         # events_df_parquet = events_df.copy()
    #         # events_df_parquet['done'] = done_bool
    #         events_df.to_parquet(file_name.replace('.hdf5', '.parquet'), index=False)
    #         print(f"events_data dataset saved successfully in {file_name}")
    # else:
    #     try:
    #         events_df = pd.read_hdf(file_name, key='events_data')
    #     except:
    #         events_df = pd.DataFrame(columns=['Id', 'event_name', 'done', 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality', 'duration(minutes)'])    
    # print(f"events_data dataset loaded successfully from {file_name}")

    # if mode == 'overwrite':
    #     events_df = events_df[events_df['Id'] != subject]
    
    events_df = pd.DataFrame(columns=['Id', 'event_name', 'done', 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality', 'duration(minutes)'])

    fibro_hdf = h5py.File(file_name, 'r+')


    for group in fibro_hdf.keys():
        if mode == 'skip':
            if group == subject:
                break
            else:
                continue
        for key in fibro_hdf[group].keys():
            if 'events_data' in fibro_hdf[group][key].attrs.keys():
                for event, value in json.loads(fibro_hdf[group][key].attrs['events_data'].decode('utf-8')).items():
                    
                    sub_id = group
                    event_name = value['Classification']
                    if 'artifacts_percent' in value.keys():
                        artifacts_percentage = value['artifacts_percent']
                    else:
                        artifacts_percentage = 0

                    if 'corrected_rpeaks_percent' in value.keys():
                        corrected_rpeaks = value['corrected_rpeaks_percent']
                    else:
                        corrected_rpeaks = 0

                    if 'avg_motion' in value.keys():
                        avg_motion = value['avg_motion']
                    else:
                        avg_motion = 0

                    if 'avg_quality' in value.keys():
                        avg_quality = value['avg_quality']
                    else:
                        avg_quality = 0
                    start_time = value['Start_Time']
                    end_time = value['End_Time']
                    duration = end_time - start_time
                    duration = duration/60
                    new_row = {'Id': sub_id, 'event_name': event_name, 'done': False, 'artifacts_percentage_%': artifacts_percentage, 'corrected_rpeaks_%': corrected_rpeaks, 'avg_motion': avg_motion, 'avg_quality': avg_quality, 'duration(minutes)': duration}
                    if 'done' not in events_df.columns:
                        events_df['done'] = False
                    
                    events_df = pd.concat([events_df, pd.DataFrame([new_row])])

                event_name = 'Total'
                filtered_df = events_df[events_df['Id'] == group]

                # Convert the 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality' and 'duration(minutes)' columns to float
                filtered_df['artifacts_percentage_%'] = filtered_df['artifacts_percentage_%'].astype(float)
                filtered_df['corrected_rpeaks_%'] = filtered_df['corrected_rpeaks_%'].astype(float)
                filtered_df['avg_motion'] = filtered_df['avg_motion'].astype(float)
                filtered_df['avg_quality'] = filtered_df['avg_quality'].astype(float)
                filtered_df['duration(minutes)'] = filtered_df['duration(minutes)'].astype(float)
                if 'done' not in filtered_df.columns:
                    filtered_df['done'] = False
                
                filtered_df['done'] = False

                aggregated_values = filtered_df.agg({
                    'artifacts_percentage_%': 'mean',
                    'corrected_rpeaks_%': 'mean',
                    'avg_motion': 'mean',
                    'avg_quality': 'mean',
                    'duration(minutes)': 'sum'
                }).to_dict()
                

                # round the values
                aggregated_values['artifacts_percentage_%'] = round(aggregated_values['artifacts_percentage_%'], 2)
                aggregated_values['corrected_rpeaks_%'] = round(aggregated_values['corrected_rpeaks_%'], 2)
                aggregated_values['avg_motion'] = round(aggregated_values['avg_motion'], 2)
                aggregated_values['avg_quality'] = round(aggregated_values['avg_quality'], 2)
                aggregated_values['duration(minutes)'] = round(aggregated_values['duration(minutes)'], 2)

                new_row_mean = {'Id': sub_id, 'event_name': event_name, 'done': False, 'artifacts_percentage_%': aggregated_values['artifacts_percentage_%'], 'corrected_rpeaks_%': aggregated_values['corrected_rpeaks_%'], 'avg_motion': aggregated_values['avg_motion'], 'avg_quality': aggregated_values['avg_quality'], 'duration(minutes)': aggregated_values['duration(minutes)']}
                
                if 'done' not in events_df.columns:
                    events_df['done'] = False

                events_df = events_df.dropna(subset=['Id'])

                events_df = pd.concat([events_df, pd.DataFrame([new_row_mean])])
                
                break
            break

    fibro_hdf.close()

    # events_df = events_df.reset_index(drop=True)

    # # Add the events_df as a new dataset in the hdf file
    # # Convert the 'duration' column to string
    # events_df['duration(minutes)'] = events_df['duration(minutes)'].astype(str)

    
    # # Remove the 'events_data' dataset if it already exists
    # with pd.HDFStore(file_name, mode='a') as store:
    #     if 'events_data' in store.keys():
    #         store.remove('events_data')
    #     store.close()

    # # events_df['done'] = str(events_df['done'])
    # # Now try saving to HDF5 again
    # # events_df.to_hdf(file_name, key='events_data', mode='a', format='table')
    # # events_df_parquet = events_df.copy()
    # # events_df_parquet['done'] = events_df_parquet['done'].astype(bool)
    # events_df.to_parquet(file_name.replace('.hdf5', '.parquet'), index=False)


    

    return f'Data for {subject} saved successfully to {project}_mindware.hdf5', events_df



@callback(
    Output("modal", "is_open", allow_duplicate=True),
    [Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


