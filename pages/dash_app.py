from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objs as go
import pandas as pd
import h5py
import numpy as np
import webview
import os
# import dash_core_components as dcc
from flask import Flask
import neurokit2 as nk
import base64
import json
import io
import re
import dash
import polars as pl
import datetime

dash.register_page(__name__, name="Signal Processing Page", path="/", order = 6)

pages = {}

for page in os.listdir(r"C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates"):
    page_name = page.split("_")[0]
    page_value = page.split("_")[0]

    pages[page_name] = page_value




layout = html.Div([
    dbc.Container(
        [
            dbc.Row(
                [
                    html.Div("Signal Processing Page", style={"fontSize": 50, 'textAlign': 'center'}),
                ]
            ),
            dbc.Row(
                [
                    html.Div("In this page you can select a project and a subject to view the signals and process them.", style={"fontSize": 20, 'textAlign': 'center'}),
                ]
            ),
            dbc.Row(
                [
                dbc.Col([
                    html.Div("Select Project", style={"fontSize": 20, 'textAlign': 'right'}),
                ]),
                dbc.Col([
                    dcc.Dropdown(
                        id='project-selection-dropdown',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={"width": "60%", 'color': 'black', 'margin': 'auto', 'textAlign': 'left'},
                        value='fibro'
                    ),
                ]),
                dbc.Col([
                    dbc.Button(
                        id='submit-project-button',
                        n_clicks=0,
                        children='Submit',
                        style={'margin': 'auto', 'textAlign': 'left'}
                    ),
                ]),
                ]),
            ]),
    dbc.Container(
        [
            dcc.Loading(
                [html.Div(id='subjects-table-container', children=[])],
                overlay_style={"visibility": "visible", 
                                 "opacity": 0.5,
                                 "background": "white"},
                custom_spinner=html.H2(["Loading subjects table...", dbc.Spinner(color="primary")]),
            ),
        ]),   
    dbc.Container(
        [
            dcc.Loading(
                [html.Div(id='plot-container', children=[])],
                overlay_style={"visibility": "visible", 
                                 "opacity": 0.5,
                                 "background": "white"},
                custom_spinner=html.H2(["Loading plot...", dbc.Spinner(color="primary")]),
            ),
            html.Hr(),
            html.Div('When you are done with the plot, please click the "Save Changes" button to save the changes.', style={'color': 'red', 'fontSize': 12})
        ]
    ), 
    dbc.Container(
        [
            dcc.Loading(
                [html.Div(id='click-data-container', children=[])],
                overlay_style={"visibility": "visible", 
                                 "opacity": 0.5,
                                 "background": "white"},
                custom_spinner=html.H2(["Loading click data...", dbc.Spinner(color="primary")]),
            ),
        ]
    ),  

    dcc.Store(id='signal-processing-store', data = [], storage_type='session'),     
    dcc.Store(id='click-data-history-store', data=[], storage_type='session')
])

@callback(
    Output('subjects-table-container', 'children'),
    Input('submit-project-button', 'n_clicks'),
    State('project-selection-dropdown', 'value'),
)
def update_subjects_table(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    if n_clicks is None:
        raise PreventUpdate
    
    
    fibro_hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'

    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware.parquet')

    print(f'fibro_hdf_file: {fibro_hdf_file}')

    if not os.path.exists(fibro_hdf_file):
        return html.Div("Error loading data. Please load subjects to the app and try again.", style={"color": "red", "fontSize": 20})


    print(f'Loading fibro subjects table...')

    try:
        # with pd.HDFStore(fibro_hdf_file, 'r') as store:
        #     subjects = pd.read_hdf(store, key='events_data')

        #     # try:
        #     #     fibro_events_stats = json.loads(store.get_storer('events_data').attrs.description.decode('utf-8'))
        #     # except:
        #     #     fibro_events_stats = json.loads(store.get_storer('events_data').attrs.description)
            
        #     store.close()
        subjects = pl.read_parquet(fibro_hdf_file.replace('.hdf5', '.parquet'))
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return html.Div("Error loading data. Please check the file and try again.", style={"color": "red", "fontSize": 20})

    print(f'Loaded fibro subjects file')


    



    subjects = (
        subjects
        .drop_nulls(subset=['Id'])
        .with_row_index(name='Index')
    )


    

    

    columns_def = [
        {
            'headerName': 'Index',
            'field': 'Index',
            'checkboxSelection': True,
        },
        {
            'headerName': 'Id',
            'field': 'Id',
        },
        {
            'headerName': 'event_name',
            'field': 'event_name',
            'type': 'leftAligned',
        },
        {
            'headerName': 'done',
            'field': 'done',
            'cellRenderer': 'agCheckboxCellRenderer',
            'cellRendererParams': {'disabled': True}
        },
        {
            'headerName': 'artifacts_percentage_%',
            'field': 'artifacts_percentage_%',
            'type': 'rightAligned',
        },
        {
            'headerName': 'corrected_rpeaks_%',
            'field': 'corrected_rpeaks_%',
            'type': 'rightAligned',
        },
        {
            'headerName': 'avg_motion',
            'field': 'avg_motion',
            'type': 'rightAligned',
        },
        {
            'headerName': 'avg_quality',
            'field': 'avg_quality',
            'type': 'rightAligned',
        },
        {
            'headerName': 'duration(minutes)',
            'field': 'duration(minutes)',
            'type': 'rightAligned',
        }
    ]


    subjects = subjects.to_pandas()
        
    print(f'Finished loading fibro subjects table')

    subjects_table = dag.AgGrid(
        id={'type': 'subjects-AgGrid', 'index': n_clicks},
        columnDefs=columns_def,
        rowData=subjects.to_dict('records'),
        defaultColDef={'resizable': True, 'sortable': True, 'filter': True},
        columnSize='responsiveSizeToFit',
        # getRowStyle=getRowStyle,
        dashGridOptions={'pagination': True, 'paginationPageSize': 30, 'undoRedoCellEditing': True, 'rowSelection': 'single'},
    )

    # columns_dropdown = dcc.Dropdown(
    #     id={'type': 'columns-dropdown', 'index': n_clicks},
    #     options=[
    #         {'label': 'artifacts_percentage_%', 'value': 'artifacts_percentage_%'},
    #         {'label': 'corrected_rpeaks_%', 'value': 'corrected_rpeaks_%'},
    #         {'label': 'avg_motion', 'value': 'avg_motion'},
    #         {'label': 'avg_quality', 'value': 'avg_quality'},
    #         {'label': 'duration(minutes)', 'value': 'duration(minutes)'}
    #     ],
    #     style={"width": "60%", 'color': 'black', 'margin': 'auto', 'textAlign': 'left'},
    # )

    
    

    # std_slider = dcc.Slider(
    #     id={'type': 'std-slider', 'index': n_clicks},
    #     min=0,
    #     max=6,
    #     step=0.1,
    #     value=1,
    #     marks={i: str(i) for i in range(7)},
    #     tooltip={'placement': 'bottom'}
    # )

    


    gsc_or_ecg = dbc.RadioItems(
        id={'type': 'gsc-or-ecg-radio', 'index': n_clicks},
        options=[
            {'label': 'GSC', 'value': 'gsc'},
            {'label': 'ECG', 'value': 'ecg'}
        ],
        value='gsc',
        inline=True
    )

    print(f'Finished loading fibro subjects table')
    return html.Div([
        subjects_table,
        gsc_or_ecg
    ])
    



@callback(
    Output('plot-container', 'children'),
    Input({'type': 'subjects-AgGrid', 'index': ALL}, 'selectedRows'),
    Input({'type': 'gsc-or-ecg-radio', 'index': ALL}, 'value'),
    State('project-selection-dropdown', 'value'),
    )
def update_plot(selectedRows, gsc_or_ecg, project):
    print(f'selectedRows: {selectedRows}')
    if not selectedRows:
        return {}
    
    for row in selectedRows[0]:
        gsc_or_ecg = gsc_or_ecg[0]
        print(gsc_or_ecg)
        sub_id = row['Id']
        print(f'sub_id: {sub_id}')
        event_name = row['event_name']
        print(f'event_name: {event_name}')

        hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'

        hdf_file = os.path.join(hdf_file, f'{project}_mindware.hdf5')

        if not os.path.exists(hdf_file):
            return html.Div("Error loading data. Please load subjects to the app and try again.", style={"color": "red", "fontSize": 20})

        # try:
        print(f'Loading data for subject {sub_id}...')
        fibro_file = h5py.File(hdf_file, 'r')
        print(f'Loaded data for subject {sub_id}')
        fibro_events_data = json.loads(fibro_file[sub_id]['data'].attrs['events_data'].decode('utf-8'))
        
        try:
            print(f'fibro_events_data: {fibro_events_data}')
            gsc_attributes = json.loads(fibro_file[sub_id]['data'].attrs['gsc_attributes'].decode('utf-8'))

            print(f'gsc_attributes: {gsc_attributes}')
        except Exception as e:
            print(f'Error loading attributes: {e}')
            gsc_attributes = {}
            
        
        event_start_time = fibro_events_data[event_name]['Start_Time']
        event_end_time = fibro_events_data[event_name]['End_Time']


        print(f'event_start_time: {event_start_time}')
        print(f'event_end_time: {event_end_time}')
        
        try:
            try:
                signal_attributes = json.loads(fibro_file[sub_id]['rpeaks'].attrs['signal_attributes'].decode('utf-8'))
            except Exception as e:
                signal_attributes = json.loads(fibro_file[sub_id]['rpeaks'].attrs['signal_attributes'])
        except Exception as e:
            print(f'Error loading signal attributes: {e}')
            signal_attributes = {}

        print(f'signal_attributes: {signal_attributes}')
        fibro_file.close()
        print(f'Closed file')

        
        origin_signals = pd.read_hdf(hdf_file, key=f'{sub_id}/data')


        # Standardize the ecg, gsc and motion signals
        if gsc_or_ecg == 'ecg':
            origin_signals['signal'] = (origin_signals['signal'] - origin_signals['signal'].mean()) / origin_signals['signal'].std()
            origin_ecg = origin_signals['signal'].to_list()
            origin_gsc = []
        elif gsc_or_ecg == 'gsc':
            origin_signals['gsc'] = (origin_signals['gsc'] - origin_signals['gsc'].mean()) / origin_signals['gsc'].std()
            origin_gsc = origin_signals['gsc'].to_list()
            origin_ecg = []
        origin_signals['motion'] = (origin_signals['motion'] - origin_signals['motion'].mean()) / origin_signals['motion'].std()


        
        

        original_time = origin_signals['time'].to_list()

        print(f'signals: {origin_signals.head()}')

        try:
            rpeaks_index_df = pd.read_hdf(hdf_file, key=f'{sub_id}/rpeaks')
        
            print(f'rpeaks_index_df: {rpeaks_index_df.head()}')
        except Exception as e:
            print(f'Error loading rpeaks index: {e}')
        
        # Slice by 'Time' column
        signals = origin_signals[(origin_signals['time'] >= event_start_time) & (origin_signals['time'] <= event_end_time)]

        print(f'signals: {signals.head()}')

        # Plotting
        if 'gsc' in signals.columns and 'signal' in signals.columns:
            try:
                if gsc_or_ecg == 'gsc':
                    print(f'Plotting GSC or ECG signals...')
                    gsc_signal = signals['gsc'].to_list()
                    eda_tonic = signals['eda_tonic'].to_list()
                    eda_phasic = signals['eda_phasic'].to_list()

                    print(f'gsc_signal: {gsc_signal[:5]}')
                time = signals['time'].to_list()
                motion = signals['motion'].to_list()
    

                if gsc_or_ecg == 'ecg':
                    print(f'time: {time[:5]}')
                    ecg_signal = signals['signal'].to_list()

                    ecg_quality = nk.ecg_quality(origin_ecg, sampling_rate=500)

                

                    rpeaks = rpeaks_index_df[0].values.tolist()

                # rpeaks_quality = ecg_quality[rpeaks]

                    rpeaks_time = [original_time[i] for i in rpeaks]

                    rpeaks_time = [i for i in rpeaks_time if i >= event_start_time and i <= event_end_time]
                    rpeaks_index = [i for i in range(len(original_time)) if original_time[i] in rpeaks_time]
                    rpeaks_quality = [ecg_quality[i] for i in rpeaks_index]


                    print(f'rpeaks_quality: {rpeaks_quality}')
                    print(f'rpeaks_time: {rpeaks_time}')
                    yellow_peaks_time = signal_attributes['yellow_peaks']
                    yellow_peaks_time = [original_time[i] for i in yellow_peaks_time]
                    yellow_peaks_time = [i for i in yellow_peaks_time if i >= event_start_time and i <= event_end_time]
                    yellow_peaks_index = [i for i in range(len(original_time)) if original_time[i] in yellow_peaks_time]

                    print(f'yellow_peaks_time: {yellow_peaks_time}')
                    ectopic_beats_time = signal_attributes['ectopic_beats']
                    print(f'ectopic_beats_time: {ectopic_beats_time}')
                    ectopic_beats_time = [original_time[i] for i in ectopic_beats_time]
                    ectopic_beats_time = [i for i in ectopic_beats_time if i >= event_start_time and i <= event_end_time]
                    ectopic_beats_index = [i for i in range(len(original_time)) if original_time[i] in ectopic_beats_time]

                    print(f'ectopic_beats_time: {ectopic_beats_time}')
                    long_short_beats_time = signal_attributes['longshort_beats']
                    print(f'long_short_beats_time: {long_short_beats_time}')
                    long_short_beats_time = [original_time[i] for i in long_short_beats_time]
                    long_short_beats_time = [i for i in long_short_beats_time if i >= event_start_time and i <= event_end_time]
                    long_short_beats_index = [i for i in range(len(original_time)) if original_time[i] in long_short_beats_time]


                    print(f'long_short_beats_time: {long_short_beats_time}')
                    false_negative_beats_time = signal_attributes['false_negatives']
                    print(f'false_negative_beats_time: {false_negative_beats_time}')
                    false_negative_beats_time = [original_time[i] for i in false_negative_beats_time]
                    false_negative_beats_time = [i for i in false_negative_beats_time if i >= event_start_time and i <= event_end_time]
                    false_negative_beats_index = [i for i in range(len(original_time)) if original_time[i] in false_negative_beats_time]


                    print(f'false_negative_beats_time: {false_negative_beats_time}')
                    false_positive_beats_time = signal_attributes['false_positives']
                    print(f'false_positive_beats_time: {false_positive_beats_time}')
                    false_positive_beats_time = [original_time[i] for i in false_positive_beats_time]
                    false_positive_beats_time = [i for i in false_positive_beats_time if i >= event_start_time and i <= event_end_time]
                    false_positive_beats_index = [i for i in range(len(original_time)) if original_time[i] in false_positive_beats_time]

                if gsc_attributes != {}:
                    scr_onset = gsc_attributes['SCR_Onsets']
                    scr_onset = [original_time[i] for i in scr_onset]
                    scr_onsets_time = [i for i in scr_onset if i >= event_start_time and i <= event_end_time]
                    scr_onsets_index = [i for i in range(len(original_time)) if original_time[i] in scr_onsets_time]
                    print(f'scr_onset: {scr_onset}')


                    print(f'scr_onsets_time: {scr_onsets_time}')
                    scr_peak = gsc_attributes['SCR_Peaks']
                    scr_peak = [original_time[i] for i in scr_peak]
                    scr_peaks_time = [i for i in scr_peak if i >= event_start_time and i <= event_end_time]
                    scr_peak_index = [i for i in range(len(original_time)) if original_time[i] in scr_peaks_time]

                    print(f'scr_peak: {scr_peak}')

                    print(f'scr_peaks_time: {scr_peaks_time}')

                    scr_height = gsc_attributes['SCR_Height']

                    print(f'scr_height: {scr_height}')
                    scr_amplitude = gsc_attributes['SCR_Amplitude']

                    print(f'scr_amplitude: {scr_amplitude}')
                    scr_risetime = gsc_attributes['SCR_RiseTime']

                    print(f'scr_risetime: {scr_risetime}')
                    scr_recovery = gsc_attributes['SCR_Recovery']
            
                    scr_recovery = [int(x) for x in scr_recovery if not np.isnan(x)]

                    scr_recovery = [original_time[i] for i in scr_recovery]

                    scr_recovery_time = [i for i in scr_recovery if i >= event_start_time and i <= event_end_time]

                    scr_recovery_index = [i for i in range(len(original_time)) if original_time[i] in scr_recovery_time]

                    print(f'scr_recovery: {scr_recovery}')

                    scr_recoverytime = gsc_attributes['SCR_RecoveryTime']

                    print(f'scr_recoverytime: {scr_recoverytime}')
                
                else:
                    scr_onsets_time = []
                    scr_peaks_time = []
                    scr_recovery_time = []
                    scr_onsets_index = []
                    scr_peak_index = []
                    scr_recovery_index = []
                

                # graph = html.Div(
                #     'Plotting GSC or ECG signals...',
                #     style={'color': 'green', 'fontSize': 20}
                # )

                if gsc_or_ecg == 'gsc':
                    # if gsc_attributes == {}:
                    #     return html.Div("No GSC data available for this subject.", style={"color": "red", "fontSize": 20})
                    print(f'Plotting GSC signal...')
                    
                    graph = dcc.Graph(
                                id={
                                    'type': 'gsc-plot',
                                    'index': 1
                                },
                                figure={
                                    'data': [
                                        go.Scatter(
                                            x=time,
                                            y=[x*1 for x in motion],
                                            mode='lines',
                                            name='Motion',
                                            line=dict(color='black'),
                                            hovertext=[f'Time: {time[i]}<br>Motion: {motion[i]}' for i in range(len(time))]
                                        ),
                                        go.Scatter(
                                            x=time,
                                            y=gsc_signal,
                                            mode='lines',
                                            name='GSC',
                                            line=dict(color='blue'),
                                            hovertext=[f'Time: {time[i]}<br>GSC: {gsc_signal[i]}' for i in range(len(time))]
                                        ),
                                        go.Scatter(
                                            x=time,
                                            y=eda_tonic,
                                            mode='lines',
                                            name='EDA Tonic',
                                            line=dict(color='green'),
                                            hovertext=[f'Time: {time[i]}<br>EDA Tonic: {eda_tonic[i]}' for i in range(len(time))]
                                        ),
                                        go.Scatter(
                                            x=time,
                                            y=eda_phasic,
                                            mode='lines',
                                            name='EDA Phasic',
                                            line=dict(color='red'),
                                            hovertext=[f'Time: {time[i]}<br>EDA Phasic: {eda_phasic[i]}' for i in range(len(time))]
                                        ),
                                        go.Scatter(
                                            x = scr_onsets_time,
                                            y = [origin_gsc[i] for i in scr_onsets_index],
                                            mode = 'markers',
                                            name = 'SCR Onsets',
                                            marker = dict(symbol='x', color='green'),
                                            hovertext=[f'Time: {scr_onsets_time[i]}<br>SCR Onset: {gsc_signal[i]}' for i in range(len(scr_onsets_time))]
                                        ),
                                        go.Scatter(
                                            x = scr_peaks_time,
                                            y = [origin_gsc[i] for i in scr_peak_index],
                                            mode = 'markers',
                                            name = 'SCR Peaks',
                                            marker = dict(symbol='x', color='red'),
                                            hovertext=[f'Time: {scr_peaks_time[i]}<br>SCR Peak: {gsc_signal[i]}' for i in range(len(scr_peaks_time))]
                                        ),
                                        go.Scatter(
                                            x = scr_recovery_time,
                                            y = [origin_gsc[i] for i in scr_recovery_index],
                                            mode = 'markers',
                                            name = 'SCR Recovery',
                                            marker = dict(symbol='x', color='yellow'),
                                            hovertext=[f'Time: {scr_recovery_time[i]}<br>SCR Recovery: {gsc_signal[i]}' for i in range(len(scr_recovery_time))]
                                        ),
                                    ],
                                    'layout': {
                                        'title': 'GSC Signal',
                                        'xaxis': {'title': 'Time'},
                                        'yaxis': {'title': 'GSC'},
                                        'hovermode': 'closest'
                                    }
                                }
                            )
                    
                    print(f'Graph created...')
                    
                    
                    graph_layout = html.Div([
                        
                        graph,
                        dbc.Row([
                            dbc.Col(
                                dbc.Container(
                                    [
                                        dbc.Button(
                                        'Save Changes',
                                        id={
                                            'type': 'save-changes-button',
                                            'index': 1
                                        },
                                        n_clicks=0,
                                        style={'margin': 'auto', 'textAlign': 'center'}
                                        ),
                                        dcc.Loading(
                                            [dcc.ConfirmDialog(
                                                id={
                                                    'type': 'save-changes-confirm',
                                                    'index': 1
                                                    },
                                                    message='',
                                            )],
                                            overlay_style={"visibility": "visible", 
                                            "opacity": 0.5,
                                            "background": "white"},
                                            custom_spinner=html.H2(["Saving changes...", dbc.Spinner(color="primary")]),
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("Are you sure you want to save the changes?"),
                                                dbc.ModalBody([
                                                    html.Div('Please make sure the changes are correct before saving.'),
                                                    dbc.Button('Save', id={
                                                        'type': 'save-changes-confirm-button',
                                                        'index': 1
                                                    }, style={'margin': 'auto', 'textAlign': 'center'})
                                            ]),
                                                dbc.ModalFooter(
                                                    dbc.Button('Close', id={
                                                        'type': 'save-changes-close-button',
                                                        'index': 1
                                                    })
                                                ),
                                            ],
                                            id={
                                                'type': 'save-changes-modal',
                                                'index': 1
                                            },
                                            is_open=False    
                                        ),
                                        dcc.ConfirmDialog(
                                            id={
                                                'type': 'no-changes-confirm',
                                                'index': 1
                                            },
                                            message='No changes were made.'
                                        )
                                        
                                    ]),                        
                            ),
                                dbc.Col(
                                    html.Div(
                                        id={
                                            'type': 'changes-ecg',
                                            'index': 1
                                        },
                                        children=[]
                                    
                                    ),
                                ),
                                dbc.Col(
                                    html.Div(
                                        id={
                                            'type': 'changes-gsc',
                                            'index': 1
                                        },
                                        children=[]
                                    )
                                )]
                        )
                    ]
                )
                    
                    return graph_layout
            


                elif gsc_or_ecg == 'ecg':
                    print(f'Plotting ECG signal...')

                    if signal_attributes == {}:
                        return html.Div("No ECG data available for this subject.", style={"color": "red", "fontSize": 20})

                    graph = dcc.Graph(
                        id={
                            'type': 'ecg-plot',
                            'index': 1
                        },
                        figure={
                            'data': [
                                go.Scatter(
                                    x=time,
                                    y=[x*1 for x in motion],
                                    mode='lines',
                                    name='Motion',
                                    line=dict(color='black'),
                                    hovertext=[f'Time: {time[i]}<br>Motion: {motion[i]}' for i in range(len(time))]
                                ),
                                go.Scatter(
                                    x=time,
                                    y=ecg_signal,
                                    mode='lines',
                                    name='ECG',
                                    line=dict(color='blue'),
                                    hovertext=[f'Time: {time[i]}<br>ECG: {ecg_signal[i]}' for i in range(len(time))]
                                ),
                                go.Scatter(
                                    x=rpeaks_time,
                                    y=[origin_ecg[i] for i in rpeaks_index],
                                    mode='markers',
                                    name='R-Peaks',
                                    # marker color is yellow if the peak is a yellow peak, if not - red
                                    marker=dict(color=['yellow' if i in signal_attributes['yellow_peaks'] else 'red' for i in rpeaks_index]),
                                    hovertext=[f'Time: {rpeaks_time[i]}<br>R-Peak: {ecg_signal[i]} <br>Quality: {rpeaks_quality[i]}' for i in range(len(rpeaks_time))]
                                ),
                                go.Scatter(
                                    x=ectopic_beats_time,
                                    y=[origin_ecg[i] for i in ectopic_beats_index],
                                    mode='markers',
                                    name='Ectopic Beats',
                                    marker=dict(symbol='x', color='black'),
                                    hovertext=[f'Time: {rpeaks_time[i]}<br>Ectopic Beat: {ecg_signal[i]}' for i in range(len(rpeaks_time))]
                                ),
                                go.Scatter(
                                    x=long_short_beats_time,
                                    y=[origin_ecg[i] for i in long_short_beats_index],
                                    mode='markers',
                                    name='Long/Short Beats',
                                    marker=dict(symbol='x', color='green'),
                                    hovertext=[f'Time: {rpeaks_time[i]}<br>Long/Short Beat: {ecg_signal[i]}' for i in range(len(rpeaks_time))]
                                ),
                                go.Scatter(
                                    x=false_negative_beats_time,
                                    y=[origin_ecg[i] for i in false_negative_beats_index],
                                    mode='markers',
                                    name='False Negative Beats',
                                    marker=dict(symbol='x', color='red'),
                                    hovertext=[f'Time: {rpeaks_time[i]}<br>False Negative Beat: {ecg_signal[i]}' for i in range(len(rpeaks_time))]
                                ),
                                go.Scatter(
                                    x=false_positive_beats_time,
                                    y=[origin_ecg[i] for i in false_positive_beats_index],
                                    mode='markers',
                                    name='False Positive Beats',
                                    marker=dict(symbol='x', color='yellow'),
                                    hovertext=[f'Time: {rpeaks_time[i]}<br>False Positive Beat: {ecg_signal[i]}' for i in range(len(rpeaks_time))]
                                ),
                                
                            ],

                            'layout': {
                                'title': 'ECG Signal',
                                'xaxis': {'title': 'Time'},
                                'yaxis': {'title': 'ECG'},
                                'hovermode': 'closest'
                            }
                        }

                    )

                    graph_layout = html.Div([
                        
                        graph,
                        dbc.Row([
                            dbc.Col(
                                dbc.Container(
                                    [
                                        dbc.Button(
                                        'Save Changes',
                                        id={
                                            'type': 'save-changes-button',
                                            'index': 1
                                        },
                                        n_clicks=0,
                                        style={'margin': 'auto', 'textAlign': 'center'}
                                        ),
                                        dcc.Loading(
                                            [dcc.ConfirmDialog(
                                                id={
                                                    'type': 'save-changes-confirm',
                                                    'index': 1
                                                    },
                                                    message='',
                                            )],
                                            overlay_style={"visibility": "visible", 
                                            "opacity": 0.5,
                                            "background": "white"},
                                            custom_spinner=html.H2(["Saving changes...", dbc.Spinner(color="primary")]),
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("Are you sure you want to save the changes?"),
                                                dbc.ModalBody([
                                                    html.Div('Please make sure the changes are correct before saving.'),
                                                    dbc.Button('Save', id={
                                                        'type': 'save-changes-confirm-button',
                                                        'index': 1
                                                    }, style={'margin': 'auto', 'textAlign': 'center'})
                                            ]),
                                                dbc.ModalFooter(
                                                    dbc.Button('Close', id={
                                                        'type': 'save-changes-close-button',
                                                        'index': 1
                                                    })
                                                ),
                                            ],
                                            id={
                                                'type': 'save-changes-modal',
                                                'index': 1
                                            },
                                            is_open=False    
                                        ),
                                        dcc.ConfirmDialog(
                                            id={
                                                'type': 'no-changes-confirm',
                                                'index': 1
                                            },
                                            message='No changes were made.'
                                        )
                                        
                                    ]),                        
                            ),
                                dbc.Col(
                                    html.Div(
                                        id={
                                            'type': 'changes-ecg',
                                            'index': 1
                                        },
                                        children=[]
                                    
                                    ),
                                ),
                                dbc.Col(
                                    html.Div(
                                        id={
                                            'type': 'changes-gsc',
                                            'index': 1
                                        },
                                        children=[]
                                    )
                                )]
                            ),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Slider(0.0,
                                                1.0,
                                                0.01,
                                                marks = {i: str(round(i,2)) for i in np.arange(0.0, 1.0, 0.05)},
                                                value=0.5,
                                                id={
                                                    'type': 'slider-ecg',
                                                    'index': 1
                                                }),
                                ),
                            ]),
                            dbc.Row([
                                html.Div(
                                    id={
                                        'type': 'slider-output-ecg',
                                        'index': 1
                                    }
                                )
                            ])
                    ]
                )

                    return graph_layout
            except Exception as e:
                print(f'Error plotting signals: {e}')
                return html.Div("Error plotting signals, check if there is data in the subject's data folder.", style={"color": "red", "fontSize": 20})


        
        elif 'signal' not in signals.columns and 'gsc' in signals.columns:
            try:
                if gsc_or_ecg == 'gsc':
                    gsc_signal = signals['gsc'].to_list()
                    eda_tonic = signals['eda_tonic'].to_list()
                    eda_phasic = signals['eda_phasic'].to_list()

                print(f'gsc_signal: {gsc_signal[:5]}')
                time = signals['time'].to_list()
                motion = signals['motion'].to_list()

                if gsc_or_ecg == 'gsc':

                    scr_onset = gsc_attributes['SCR_Onsets']
                    scr_onset = [original_time[i] for i in scr_onset]
                    scr_onsets_time = [i for i in scr_onset if i >= event_start_time and i <= event_end_time]
                    scr_onsets_index = [i for i in range(len(original_time)) if original_time[i] in scr_onsets_time]
                    print(f'scr_onset: {scr_onset}')


                    print(f'scr_onsets_time: {scr_onsets_time}')
                    scr_peak = gsc_attributes['SCR_Peaks']
                    scr_peak = [original_time[i] for i in scr_peak]
                    scr_peaks_time = [i for i in scr_peak if i >= event_start_time and i <= event_end_time]
                    scr_peak_index = [i for i in range(len(original_time)) if original_time[i] in scr_peaks_time]

                    print(f'scr_peak: {scr_peak}')

                    print(f'scr_peaks_time: {scr_peaks_time}')

                    scr_height = gsc_attributes['SCR_Height']

                    print(f'scr_height: {scr_height}')
                    scr_amplitude = gsc_attributes['SCR_Amplitude']

                    print(f'scr_amplitude: {scr_amplitude}')
                    scr_risetime = gsc_attributes['SCR_RiseTime']

                    print(f'scr_risetime: {scr_risetime}')
                    scr_recovery = gsc_attributes['SCR_Recovery']
                    
                    scr_recovery = [int(x) for x in scr_recovery]

                    scr_recovery = [original_time[i] for i in scr_recovery]

                    scr_recovery_time = [i for i in scr_recovery if i >= event_start_time and i <= event_end_time]

                    scr_recovery_index = [i for i in range(len(original_time)) if original_time[i] in scr_recovery_time]

                    print(f'scr_recovery: {scr_recovery}')

                    scr_recoverytime = gsc_attributes['SCR_RecoveryTime']

                    print(f'scr_recoverytime: {scr_recoverytime}')

                if gsc_or_ecg == 'gsc':
                        if gsc_attributes == {}:
                            return html.Div("No GSC data available for this subject.", style={"color": "red", "fontSize": 20})
                        
                        graph = dcc.Graph(
                                id={
                                    'type': 'gsc-plot',
                                    'index': 1
                                },
                                figure={
                                    'data': [
                                        go.Scatter(
                                            x=time,
                                            y=[x*0.1 for x in motion],
                                            mode='lines',
                                            name='Motion',
                                            line=dict(color='black'),
                                            hovertext=[f'Time: {time[i]}<br>Motion: {motion[i]}' for i in range(len(time))]
                                        ),
                                        go.Scatter(
                                            x=time,
                                            y=gsc_signal,
                                            mode='lines',
                                            name='GSC',
                                            line=dict(color='blue'),
                                            hovertext=[f'Time: {time[i]}<br>GSC: {gsc_signal[i]}' for i in range(len(time))]
                                        ),
                                        go.Scatter(
                                            x=time,
                                            y=eda_tonic,
                                            mode='lines',
                                            name='EDA Tonic',
                                            line=dict(color='green'),
                                            hovertext=[f'Time: {time[i]}<br>EDA Tonic: {eda_tonic[i]}' for i in range(len(time))]
                                        ),
                                        go.Scatter(
                                            x=time,
                                            y=eda_phasic,
                                            mode='lines',
                                            name='EDA Phasic',
                                            line=dict(color='red'),
                                            hovertext=[f'Time: {time[i]}<br>EDA Phasic: {eda_phasic[i]}' for i in range(len(time))]
                                        ),
                                        
                                        go.Scatter(
                                            x = scr_onsets_time,
                                            y = [origin_gsc[i] for i in scr_onsets_index],
                                            mode = 'markers',
                                            name = 'SCR Onsets',
                                            marker = dict(symbol='x', color='green'),
                                            hovertext=[f'Time: {scr_onsets_time[i]}<br>SCR Onset: {gsc_signal[i]}' for i in range(len(scr_onsets_time))]
                                        ),
                                        go.Scatter(
                                            x = scr_peaks_time,
                                            y = [origin_gsc[i] for i in scr_peak_index],
                                            mode = 'markers',
                                            name = 'SCR Peaks',
                                            marker = dict(symbol='x', color='red'),
                                            hovertext=[f'Time: {scr_peaks_time[i]}<br>SCR Peak: {gsc_signal[i]}' for i in range(len(scr_peaks_time))]
                                        ),
                                        go.Scatter(
                                            x = scr_recovery_time,
                                            y = [origin_gsc[i] for i in scr_recovery_index],
                                            mode = 'markers',
                                            name = 'SCR Recovery',
                                            marker = dict(symbol='x', color='yellow'),
                                            hovertext=[f'Time: {scr_recovery_time[i]}<br>SCR Recovery: {gsc_signal[i]}' for i in range(len(scr_recovery_time))]
                                        ),
                                    ],
                                    'layout': {
                                        'title': 'GSC Signal',
                                        'xaxis': {'title': 'Time'},
                                        'yaxis': {'title': 'GSC'},
                                        'hovermode': 'closest'
                                    }
                                }
                            )
                        
                elif gsc_or_ecg == 'ecg':
                    return html.Div("No ECG signal to plot", style={"color": "red", "fontSize": 20})
                
                graph_layout = html.Div([
                        
                        graph,
                        dbc.Row([
                            dbc.Col(
                                dbc.Container(
                                    [
                                        dbc.Button(
                                        'Save Changes',
                                        id={
                                            'type': 'save-changes-button',
                                            'index': 1
                                        },
                                        n_clicks=0,
                                        style={'margin': 'auto', 'textAlign': 'center'}
                                        ),
                                        dcc.Loading(
                                            [dcc.ConfirmDialog(
                                                id={
                                                    'type': 'save-changes-confirm',
                                                    'index': 1
                                                    },
                                                    message='',
                                            )],
                                            overlay_style={"visibility": "visible", 
                                            "opacity": 0.5,
                                            "background": "white"},
                                            custom_spinner=html.H2(["Saving changes...", dbc.Spinner(color="primary")]),
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("Are you sure you want to save the changes?"),
                                                dbc.ModalBody([
                                                    html.Div('Please make sure the changes are correct before saving.'),
                                                    dbc.Button('Save', id={
                                                        'type': 'save-changes-confirm-button',
                                                        'index': 1
                                                    }, style={'margin': 'auto', 'textAlign': 'center'})
                                            ]),
                                                dbc.ModalFooter(
                                                    dbc.Button('Close', id={
                                                        'type': 'save-changes-close-button',
                                                        'index': 1
                                                    })
                                                ),
                                            ],
                                            id={
                                                'type': 'save-changes-modal',
                                                'index': 1
                                            },
                                            is_open=False    
                                        ),
                                        dcc.ConfirmDialog(
                                            id={
                                                'type': 'no-changes-confirm',
                                                'index': 1
                                            },
                                            message='No changes were made.'
                                        )
                                        
                                    ]),                        
                            ),
                                dbc.Col(
                                    html.Div(
                                        id={
                                            'type': 'changes-ecg',
                                            'index': 1
                                        },
                                        children=[]
                                    
                                    ),
                                ),
                                dbc.Col(
                                    html.Div(
                                        id={
                                            'type': 'changes-gsc',
                                            'index': 1
                                        },
                                        children=[]
                                    )
                                )]
                        )
                    ]
                )
                    
                return graph_layout
            except Exception as e:
                print(f'Error plotting signals: {e}')
                return html.Div("Error plotting signals, check if there is data in the subject's data folder.", style={"color": "red", "fontSize": 20})
        
        elif 'signal' in signals.columns and 'gsc' not in signals.columns:
        
            time = signals['time'].to_list()

            motion = signals['motion'].to_list()
            
            if gsc_or_ecg == 'ecg':
                print(f'time: {time[:5]}')
                ecg_signal = signals['signal'].to_list()

                ecg_quality = nk.ecg_quality(origin_ecg, sampling_rate=500)

            

                rpeaks = rpeaks_index_df[0].values.tolist()

                rpeaks_time = [original_time[i] for i in rpeaks]

                rpeaks_time = [i for i in rpeaks_time if i >= event_start_time and i <= event_end_time]
                rpeaks_index = [i for i in range(len(original_time)) if original_time[i] in rpeaks_time]
                rpeaks_quality = [ecg_quality[i] for i in rpeaks_index]


                print(f'rpeaks_quality: {rpeaks_quality}')
                print(f'rpeaks_time: {rpeaks_time}')
                yellow_peaks_time = signal_attributes['yellow_peaks']
                yellow_peaks_time = [original_time[i] for i in yellow_peaks_time]
                yellow_peaks_time = [i for i in yellow_peaks_time if i >= event_start_time and i <= event_end_time]
                yellow_peaks_index = [i for i in range(len(original_time)) if original_time[i] in yellow_peaks_time]

                print(f'yellow_peaks_time: {yellow_peaks_time}')
                ectopic_beats_time = signal_attributes['ectopic_beats']
                print(f'ectopic_beats_time: {ectopic_beats_time}')
                ectopic_beats_time = [original_time[i] for i in ectopic_beats_time]
                ectopic_beats_time = [i for i in ectopic_beats_time if i >= event_start_time and i <= event_end_time]
                ectopic_beats_index = [i for i in range(len(original_time)) if original_time[i] in ectopic_beats_time]

                print(f'ectopic_beats_time: {ectopic_beats_time}')
                long_short_beats_time = signal_attributes['longshort_beats']
                print(f'long_short_beats_time: {long_short_beats_time}')
                long_short_beats_time = [original_time[i] for i in long_short_beats_time]
                long_short_beats_time = [i for i in long_short_beats_time if i >= event_start_time and i <= event_end_time]
                long_short_beats_index = [i for i in range(len(original_time)) if original_time[i] in long_short_beats_time]


                print(f'long_short_beats_time: {long_short_beats_time}')
                false_negative_beats_time = signal_attributes['false_negatives']
                print(f'false_negative_beats_time: {false_negative_beats_time}')
                false_negative_beats_time = [original_time[i] for i in false_negative_beats_time]
                false_negative_beats_time = [i for i in false_negative_beats_time if i >= event_start_time and i <= event_end_time]
                false_negative_beats_index = [i for i in range(len(original_time)) if original_time[i] in false_negative_beats_time]


                print(f'false_negative_beats_time: {false_negative_beats_time}')
                false_positive_beats_time = signal_attributes['false_positives']
                print(f'false_positive_beats_time: {false_positive_beats_time}')
                false_positive_beats_time = [original_time[i] for i in false_positive_beats_time]
                false_positive_beats_time = [i for i in false_positive_beats_time if i >= event_start_time and i <= event_end_time]
                false_positive_beats_index = [i for i in range(len(original_time)) if original_time[i] in false_positive_beats_time]

            if gsc_or_ecg == 'gsc':
                return html.Div("No GSC signal to plot", style={"color": "red", "fontSize": 20})
            
            elif gsc_or_ecg == 'ecg':
                if signal_attributes == {}:
                    return html.Div("No ECG data available for this subject.", style={"color": "red", "fontSize": 20})
                graph = dcc.Graph(
                    id={
                        'type': 'ecg-plot',
                        'index': 1
                    },
                    figure={
                        'data': [
                            go.Scatter(
                                x=time,
                                y=[x*1 for x in motion],
                                mode='lines',
                                name='Motion',
                                line=dict(color='black'),
                                hovertext=[f'Time: {time[i]}<br>Motion: {motion[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=time,
                                y=ecg_signal,
                                mode='lines',
                                name='ECG',
                                line=dict(color='blue'),
                                hovertext=[f'Time: {time[i]}<br>ECG: {ecg_signal[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=rpeaks_time,
                                y=[origin_ecg[i] for i in rpeaks_index],
                                mode='markers',
                                name='R-Peaks',
                                # marker color is yellow if the peak is a yellow peak, if not - red
                                marker=dict(color=['yellow' if i in signal_attributes['yellow_peaks'] else 'red' for i in rpeaks_index]),
                                hovertext=[f'Time: {rpeaks_time[i]}<br>R-Peak: {origin_ecg[i]} <br>Quality: {rpeaks_quality[i]}' for i in range(len(rpeaks_index))]
                            ),
                            go.Scatter(
                                x=ectopic_beats_time,
                                y=[origin_ecg[i] for i in signal_attributes['ectopic_beats']],
                                mode='markers',
                                name='Ectopic Beats',
                                marker=dict(symbol='x', color='black'),
                                hovertext=[f'Time: {rpeaks_time[i]}<br>Ectopic Beat: {origin_ecg[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=long_short_beats_time,
                                y=[origin_ecg[i] for i in signal_attributes['longshort_beats']],
                                mode='markers',
                                name='Long/Short Beats',
                                marker=dict(symbol='x', color='green'),
                                hovertext=[f'Time: {rpeaks_time[i]}<br>Long/Short Beat: {origin_ecg[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=false_negative_beats_time,
                                y=[origin_ecg[i] for i in signal_attributes['false_negatives']],
                                mode='markers',
                                name='False Negative Beats',
                                marker=dict(symbol='x', color='red'),
                                hovertext=[f'Time: {rpeaks_time[i]}<br>False Negative Beat: {origin_ecg[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=false_positive_beats_time,
                                y=[origin_ecg[i] for i in signal_attributes['false_positives']],
                                mode='markers',
                                name='False Positive Beats',
                                marker=dict(symbol='x', color='yellow'),
                                hovertext=[f'Time: {rpeaks_time[i]}<br>False Positive Beat: {origin_ecg[i]}' for i in range(len(rpeaks_time))]
                            ),
                            
                        ],

                        'layout': {
                            'title': 'ECG Signal',
                            'xaxis': {'title': 'Time'},
                            'yaxis': {'title': 'ECG'},
                            'hovermode': 'closest'
                        }
                    }

                )

                graph_layout = html.Div([
                        
                        graph,
                        
                        dbc.Row([
                        dbc.Col(
                            dbc.Container(
                                [
                                    dbc.Button(
                                    'Save Changes',
                                    id={
                                        'type': 'save-changes-button',
                                        'index': 1
                                    },
                                    n_clicks=0,
                                    style={'margin': 'auto', 'textAlign': 'center'}
                                    ),
                                    dcc.Loading(
                                        [dcc.ConfirmDialog(
                                            id={
                                                'type': 'save-changes-confirm',
                                                'index': 1
                                                },
                                                message='',
                                        )],
                                        overlay_style={"visibility": "visible", 
                                        "opacity": 0.5,
                                        "background": "white"},
                                        custom_spinner=html.H2(["Saving changes...", dbc.Spinner(color="primary")]),
                                    ),
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader("Are you sure you want to save the changes?"),
                                            dbc.ModalBody([
                                                html.Div('Please make sure the changes are correct before saving.'),
                                                dbc.Button('Save', id={
                                                    'type': 'save-changes-confirm-button',
                                                    'index': 1
                                                }, style={'margin': 'auto', 'textAlign': 'center'})
                                        ]),
                                            dbc.ModalFooter(
                                                dbc.Button('Close', id={
                                                    'type': 'save-changes-close-button',
                                                    'index': 1
                                                })
                                            ),
                                        ],
                                        id={
                                            'type': 'save-changes-modal',
                                            'index': 1
                                        },
                                        is_open=False    
                                    ),
                                    dcc.ConfirmDialog(
                                        id={
                                            'type': 'no-changes-confirm',
                                            'index': 1
                                        },
                                        message='No changes were made.'
                                    )
                                    
                                ]),                        
                        ),
                            dbc.Col(
                                html.Div(
                                    id={
                                        'type': 'changes-ecg',
                                        'index': 1
                                    },
                                    children=[]
                                
                                ),
                            ),
                            dbc.Col(
                                html.Div(
                                    id={
                                        'type': 'changes-gsc',
                                        'index': 1
                                    },
                                    children=[]
                                )
                            )]
                    ),
                        dbc.Row([
                            dbc.Col(
                                dcc.Slider(0.0,
                                            1.0,
                                            0.01,
                                            marks = {i: str(round(i,2)) for i in np.arange(0.0, 1.0, 0.05)},
                                            value=0.5,
                                            id={
                                                'type': 'slider-ecg',
                                                'index': 1
                                            }),
                            ),
                        ]),
                        dbc.Row([
                            html.Div(
                                id={
                                    'type': 'slider-output-ecg',
                                    'index': 1
                                }
                            )
                        ])
                ]
            )

                return graph_layout
                
            # except Exception as e:
            #     print(f'Error plotting signals: {e}')
            #     return html.Div("Error plotting signals, check if there is data in the subject's data folder.", style={"color": "red", "fontSize": 20})
            


        else:
            return html.Div("No signals to plot", style={"color": "red", "fontSize": 20})
        

@callback(
    Output({'type': 'ecg-plot', 'index': ALL}, 'figure', allow_duplicate=True),
    Output({'type': 'slider-output-ecg', 'index': ALL}, 'children'),
    Output({'type': 'slider-ecg', 'index': ALL}, 'value'),
    Input({'type': 'slider-ecg', 'index': ALL}, 'value'),
    Input({'type': 'ecg-plot', 'index': ALL}, 'figure'),
    prevent_initial_call=True
)
def update_ecg_plot_slider(slider_value, figure):
    
    if not figure:
        raise PreventUpdate
    if not slider_value:
        raise PreventUpdate
    
    if slider_value[0] is None:
        raise PreventUpdate
    
    fig = figure[0]


    quality_list = []
    treaces = 0
    for trace in fig['data']:
        
        if treaces == 2:
            for key, value in trace.items():        
                if key == 'hovertext':
                    print(f'Hovertext: {value[:5]}')
                    for val in value:
                        print(f'Val: {val}')
                        quality = val.split('<br>Quality: ')[1]
                        quality_list.append(quality)
        treaces += 1
            


    print(f'Quality list: {quality_list[:15]}')

    quality_list = [float(x) for x in quality_list]

    slider_value = slider_value[0]

    print(f'Slider value: {slider_value}')

    patched_figure = Patch()

    for i in range(len(quality_list)):
        if fig['data'][2]['marker']['color'][i] == 'yellow':
            patched_figure['data'][2]['marker']['size'][i] = 6
            patched_figure['data'][2]['marker']['color'][i] = 'yellow'
            patched_figure['data'][2]['marker']['opacity'][i] = 1.0

        elif quality_list[i] <= slider_value:
            patched_figure['data'][2]['marker']['size'][i] = 13
            patched_figure['data'][2]['marker']['color'][i] = 'orange'
            patched_figure['data'][2]['marker']['opacity'][i] = 0.8
        elif quality_list[i] > slider_value:
            patched_figure['data'][2]['marker']['size'][i] = 6
            patched_figure['data'][2]['marker']['color'][i] = 'red'
            patched_figure['data'][2]['marker']['opacity'][i] = 1.0
        else:
            patched_figure['data'][2]['marker']['size'][i] = 6
            patched_figure['data'][2]['marker']['color'][i] = 'red'
            patched_figure['data'][2]['marker']['opacity'][i] = 1.0

    title = f'Quality threshold: {slider_value}'

    slider_value = [None]

    return [patched_figure], [title], slider_value
        
        



@callback(
    Output({'type': 'ecg-plot', 'index': ALL}, 'figure', allow_duplicate=True),
    Output('signal-processing-store', 'data', allow_duplicate=True),
    Output('click-data-history-store', 'data', allow_duplicate=True),
    Input({'type': 'ecg-plot', 'index': ALL}, 'clickData'),
    Input('signal-processing-store', 'data'),
    State('click-data-history-store', 'data'),
    State('plot-container', 'children'),
    State('project-selection-dropdown', 'value'),
    State({'type': 'subjects-AgGrid', 'index': ALL}, 'selectedRows'),
    prevent_initial_call=True
)
def update_rpeaks_data(clickData, data, history_data, children, project, selectedRows):
    if not clickData:
        raise PreventUpdate
    
    if clickData == []:
        raise PreventUpdate


    if not project:
        raise PreventUpdate
    
    if not children:
        raise PreventUpdate
    
    if not selectedRows:
        raise PreventUpdate
    
    if clickData is None:
        raise PreventUpdate
    
    for click in clickData:
        if click is None:
            raise PreventUpdate
        
        if 'points' in click.keys():
            print(f'clickData: {click}')
            point = click['points'][0]

            if point['curveNumber'] == 2:
                print(f'clickData: {clickData}')
                
                selected_row_index = selectedRows[0][0]['Index']

                rpeaks_colors = children['props']['children'][0]['props']['figure']['data'][2]['marker']['color']
                rpeaks = children['props']['children'][0]['props']['figure']['data'][2]['x']
                clicked_rpeak_index = clickData[0]['points'][0]['pointIndex']
                clicked_rpeak_time = clickData[0]['points'][0]['x']




                rpeaks_length = len(rpeaks)
                yellow_peaks = [i for i in range(len(rpeaks)) if rpeaks_colors[i] == 'yellow']

                clicked_rpeak_color = clickData[0]['points'][0]['marker.color']
                clicked_rpeak = clickData[0]['points'][0]['x']

                

                

                # If the clicked rpeak is a yellow peak, change it to red
                if clicked_rpeak_color == 'yellow':
                    new_rpeak_color = 'red'
                    # Remove the clicked rpeak from the yellow peaks list if it exists
                    if clicked_rpeak_index in yellow_peaks:
                        yellow_peaks.remove(clicked_rpeak_index)
                # If the clicked rpeak is a red peak, change it to yellow
                elif clicked_rpeak_color == 'red':
                    new_rpeak_color = 'yellow'
                    # Add the clicked rpeak to the yellow peaks list
                    yellow_peaks.append(clicked_rpeak_index)

                if len(yellow_peaks) == 0:
                    artifact_percentage = 0.0
                else:
                    if len(rpeaks) == 0:
                        artifact_percentage = 0.0
                    else:
                        artifact_percentage = len(yellow_peaks) / len(rpeaks) * 100
                        artifact_percentage = round(artifact_percentage, 2)

                patched_figure = Patch()
                patched_figure['data'][2]['marker']['color'][clicked_rpeak_index] = new_rpeak_color

                # Take the subject Id from the selected rows
                sub_id = selectedRows[0][0]['Id']
                event = selectedRows[0][0]['event_name']

                if data == []:
                    data = []
                    data.append({project: {sub_id: {event: {clicked_rpeak_time: new_rpeak_color, 'artifact_percentage': artifact_percentage}}}})
                else:
                    if project not in data[0]:
                        data[0][project] = {sub_id: {event: {clicked_rpeak_time: new_rpeak_color, 'artifact_percentage': artifact_percentage}}}
                    else:
                        if sub_id not in data[0][project]:
                            data[0][project][sub_id] = {event: {clicked_rpeak_time: new_rpeak_color, 'artifact_percentage': artifact_percentage}}
                        else:
                            if event not in data[0][project][sub_id]:
                                data[0][project][sub_id][event] = {clicked_rpeak_time: new_rpeak_color, 'artifact_percentage': artifact_percentage}
                            else:
                                data[0][project][sub_id][event][clicked_rpeak_time] = new_rpeak_color
                                data[0][project][sub_id][event]['artifact_percentage'] = artifact_percentage


                history_data.append(data.copy())

                print(f'Updated data: {data}')

                return [patched_figure], data, history_data
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate
        





@callback(
    Output('click-data-container', 'children'),
    Input({'type': 'ecg-plot', 'index': ALL}, 'clickData'),
    Input('signal-processing-store', 'modified_timestamp'),
    State('signal-processing-store', 'data'),
    prevent_initial_call=True
)
def update_click_data_container(clickData, modified_timestamp, data):
    if clickData is None:
        raise PreventUpdate
    if not clickData:
        raise PreventUpdate
    if clickData == []:
        raise PreventUpdate
    if modified_timestamp is None:
        raise PreventUpdate
    
    if data == []:
        return html.Div('No data to display')

    updated_data = []

    for project, subjects in data[0].items():
        for sub_id, events in subjects.items():
            for event, times in events.items():
                time_entries = []
                artifact_percentage = events[event]['artifact_percentage']
                for time, color in times.items():
                    if time != 'artifact_percentage':

                        time_entries.append(html.P(f"Time: {time}, Marker Color: {color}"))

                updated_data.append(html.Div([
                    html.P(f"Project: {project}"),
                    html.P(f"Subject ID: {sub_id}"),
                    html.P(f"Event: {event}"),
                    html.P(f"Artifact Percentage: {artifact_percentage}"),
                    *time_entries,  # Unpack the list of time entries
                    html.Hr()  # Add a horizontal rule for better separation
                ]))

    return updated_data


@callback(
    Output({'type': 'save-changes-modal', 'index': ALL}, 'is_open'),
    Output({'type': 'no-changes-confirm', 'index': ALL}, 'displayed'),
    Input({'type': 'save-changes-button', 'index': ALL}, 'n_clicks'),
    State('signal-processing-store', 'data'),
    prevent_initial_call=True
)
def save_changes(n_clicks, data):
    if n_clicks is None:
        raise PreventUpdate
    
    if n_clicks == 0:
        raise PreventUpdate
    
    if data == []:
        return [False], [True]

    return [True], [False]


    

@callback(
    Output({'type':'save-changes-confirm', 'index': ALL}, 'message'),
    Output({'type':'save-changes-confirm', 'index': ALL}, 'displayed'),
    Output('signal-processing-store', 'data'),
    Output({'type': 'ecg-plot', 'index': ALL}, 'clickData'),
    Input({'type': 'save-changes-confirm-button', 'index': ALL}, 'n_clicks'),
    State('signal-processing-store', 'data'),
    prevent_initial_call=True
)
def save_changes_confirm(n_clicks, data):
    if n_clicks is None:
        raise PreventUpdate
    
    if n_clicks == 0:
        raise PreventUpdate
    
    if data is None:
        raise PreventUpdate

    if data == []:
        raise PreventUpdate
    
    def save_signal_changes(data):
        for project in list(data[0].keys()):  # Convert to list to avoid RuntimeError
            print(f'project: {project}')
            hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'

            hdf_file = os.path.join(hdf_file, f'{project}_mindware.hdf5')

            
            if not os.path.exists(hdf_file):
                return 'HDF file does not exist'

            subjects = list(data[0][project].keys())

            print(f'subjects: {subjects}')
            
            for sub_id in subjects:
                print(f'sub_id: {sub_id}')
                file = h5py.File(hdf_file, 'r')
                print(f'file: {file}')

                

                try:
                    signal_attributes = json.loads(file[sub_id]['rpeaks'].attrs['signal_attributes'].decode('utf-8'))
                except:
                    signal_attributes = json.loads(file[sub_id]['rpeaks'].attrs['signal_attributes'])

                print(f'signal_attributes: {signal_attributes}')
                file.close()

                print('file closed...')

                # events_df = pd.read_hdf(hdf_file, key='events_data')
                events_df = pd.read_parquet(hdf_file.replace('.hdf5', '.parquet'))

                signals_df = pd.read_hdf(hdf_file, key=f'{sub_id}/data')

                print(f'signals_df: {signals_df.head()}')
                time_v = signals_df['time'].to_list()

                time_v = [float(x) for x in time_v]

                print(f'time: {time_v[:5]}')

    
                for event in data[0][project][sub_id].keys():
                    print(f'event: {event}')
                    artifact_percentage = data[0][project][sub_id][event]['artifact_percentage']
                    print(f'artifact_percentage: {artifact_percentage}')

                    for time, color_index in data[0][project][sub_id][event].items():
                        if time == 'artifact_percentage':
                            continue
                        print(f'time: {time}')
                        print(f'color_index: {color_index}')

                        # Get the index of the rpeak in the signals dataframe by the time_v
                        rpeak_index = time_v.index(float(time))
                        print(f'rpeak_index: {rpeak_index}')

                        # Get the yellow peaks indices from the signal attributes
                        yellow_peaks = signal_attributes['yellow_peaks']

                        print(f'yellow_peaks: {yellow_peaks}')

                        # If the rpeak is a red peak, and it is in the yellow peaks list, remove it
                        if color_index == 'red' and rpeak_index in yellow_peaks:
                            yellow_peaks.remove(rpeak_index)
                            print(f'yellow_peaks: {yellow_peaks}')

                        # If the rpeak is a yellow peak, and it is not in the yellow peaks list, add it
                        elif color_index == 'yellow' and rpeak_index not in yellow_peaks:
                            yellow_peaks.append(rpeak_index)
                            print(f'yellow_peaks: {yellow_peaks}')

                        # Update the signal attributes with the new yellow peaks list
                        signal_attributes['yellow_peaks'] = yellow_peaks

                        # Enter the new artifact percentage to artifact_percentage column in the events dataframe for the specific subject and event
                        events_df.loc[(events_df['Id'] == sub_id) & (events_df['event_name'] == event), 'artifacts_percentage_%'] = artifact_percentage
                        events_df.loc[(events_df['Id'] == sub_id) & (events_df['event_name'] == event), 'done'] = True

                        # Update the signal attributes in the hdf file
                        file = h5py.File(hdf_file, 'r+')
                        file[sub_id]['rpeaks'].attrs['signal_attributes'] = json.dumps(signal_attributes).encode('utf-8')

                        
                        file.close()

                        # Update the events dataframe in the hdf file
                        events_df.to_parquet(hdf_file.replace('.hdf5', '.parquet'), index=False)

                        print(f'Changes saved successfully!')

                        # Path to your JSON file
                        file_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'

                        file_path = os.path.join(file_path, f'{project}_UpdatedData.csv')

                        # Read the existing data
                        if os.path.exists(file_path):
                            updated_df = pd.read_csv(file_path)
                        else:
                            updated_df = pd.DataFrame()

                        # Assuming `data` is a list of dictionaries and you want to update the first item
                        new_data = {
                            'project': project,
                            'subject_id': sub_id,
                            'event': event,
                            'rpeak_index': rpeak_index,
                            'rpeak_time': float(time),
                            'new color': color_index,
                            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'status': 'New'
                        }
                        updated_df = pd.concat([updated_df, pd.DataFrame(new_data, index=[0])], ignore_index=True)


                        # Write the updated data back to the file
                        updated_df.to_csv(file_path, index=False)

                        message = 'Changes saved successfully!'


        return message

    confirm_message = save_signal_changes(data)

    return [confirm_message], [True], [], [{}]












                


# @callback(
#     Output('signal-processing-plot', 'figure'),
#     Input({'type': 'subjects-AgGrid', 'index': ALL}, 'virtualRowData'),
#     prevent_initial_call=True
# )
# def update_plot(virtualRowData):
#     print(f'virtualRowData: {virtualRowData}')
#     return {}

# @callback(
#     Output({'type': 'subjects-AgGrid', 'index': ALL}, 'getRowStyle', allow_duplicate=True),
#     Input({'type': 'columns-dropdown', 'index': ALL}, 'value'),
#     Input({'type': 'std-slider', 'index': ALL}, 'value'),
#     Input('subjects-table-container', 'children'),
#     State('project-selection-dropdown', 'value'),
#     prevent_initial_call=True
# )
# def highlight_outliers(column, std_threshold, children, project):
#     if project != 'fibro':
#         raise PreventUpdate
    

#     fibro_hdf_file =  r'C:\Users\PsyLab-7084\Desktop\RoeeProj\pages\data\mindwareData\fibro_mindware.hdf5'
#     fibro_events_stats_file = h5py.File(fibro_hdf_file, 'r')
#     fibro_events_stats = json.loads(fibro_events_stats_file['events_data'].attrs['description'])
#     fibro_events_stats_file.close()

#     column = column[0]

#     std_threshold = float(std_threshold[0])
    
#     print(f'column: {column}')

#     column_mean = fibro_events_stats[column]['mean']
    
#     print(f'column_mean: {column_mean}')

#     column_std = fibro_events_stats[column]['std']

#     print(f'column_std: {column_std}')

#     threshold = column_mean + std_threshold * column_std

#     print(f'threshold: {threshold}')

#     getRowStyle = {
#         "styleConditions": [
#             {
#                 "condition": f"params.data.{column} > {threshold}",
#                 "style": {"backgroundColor": "red"},
#             },
#         ],
#         "defaultStyle": {"backgroundColor": "white", "color": "black"},
#     }

#     print(f'getRowStyle: {getRowStyle}')

#     return [getRowStyle]


    











