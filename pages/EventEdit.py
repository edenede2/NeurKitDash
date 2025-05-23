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
import datetime

dash.register_page(__name__, name="Events Editing", path="/events_edit", order=4)
pages = {}

for page in os.listdir(r"C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates"):
    page_name = page.split("_")[0]
    page_value = page.split("_")[0]

    pages[page_name] = page_value

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Events Editing"),
                html.Hr()
            ])

        ]),
        dbc.Row([
            dbc.Col([
                html.H2("Select project"),
                dcc.Dropdown(
                    id='project-dropdown-events',
                    options=[
                        {'label': key, 'value': value} for key, value in pages.items()
                    ],
                    value='fibro',
                    style={'width': '60%', 'color': 'black', 'margin': 'auto', 'textAlign': 'left'}
                    )   
                ]),
                dbc.Button('Show',
                            id='show-project-events-button',
                              n_clicks=0,
                              color='primary'),    
            ]),
        dbc.Row([
            dbc.Col([
                html.H2("Select subject"),
                html.Div(id='subject-ag-grid-events-container')
               ])
            ]),
        ]),
    dbc.Container([
        dcc.Loading(
            [html.Div(id='events-edit-plot-container', children=[])],
            overlay_style={'visibility': 'visible',
                            'opacity': 0.5,
                            'background': 'white'},
            custom_spinner= html.H2(['Loading signal plot...', dbc.Spinner(color='primary')])
        )
        ]),
    dcc.Store(id='events-edit-store')
    ])

@callback(
    Output('subject-ag-grid-events-container', 'children'),
    Input('show-project-events-button', 'n_clicks'),
    State('project-dropdown-events', 'value')
)
def show_subjects(n_click, project):
    if n_click == 0:
        raise PreventUpdate
    
    if n_click is None:
        raise PreventUpdate
    
    if project == 'fibro':
        json_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates\fibro_template.json'
        with open(json_file, 'r') as file:
            template = json.load(file)
            file.close()

        hdf_file = template['HDF5']['path']

        subjects_start_end_times = {}

        with h5py.File(hdf_file, 'r') as file:
            for subject in file.keys():
                if re.search(r'^sub_', subject):
                    events_data = json.loads(file[subject]['data'].attrs['events_data'])
                    subject_start_end_times = {}
                    for event in events_data.keys():
                        event_data = events_data[event]
                        start_time = event_data['Start_Time']
                        end_time = event_data['End_Time']
                        subject_start_end_times[event] = [start_time, end_time]
                    subjects_start_end_times[subject] = subject_start_end_times


                        
            file.close()


        subjects = list(subjects_start_end_times.keys())
        
        subjects_df = pd.DataFrame(columns=['Subject', 'Event', 'Start Time', 'End Time'])

        for subject in subjects:
            for event in subjects_start_end_times[subject].keys():
                start_time = subjects_start_end_times[subject][event][0]
                end_time = subjects_start_end_times[subject][event][1]
                new_row = {'Subject': subject, 'Event': event, 'Start Time': start_time, 'End Time': end_time}
                subjects_df = pd.concat([subjects_df, pd.DataFrame(new_row, index=[0])])






        columnDefs=[
                {'headerName': 'id', 'valueGetter': {'function': 'params.node.id'}, 'checkboxSelection': True},
            ]
        
        columnDefs += [{'headerName': col, 'field': col} for col in subjects_df.columns]

        ag_grid_subjects = dag.AgGrid(
            id={
                'type': 'ag-grid-subjects-events',
                'index': 1,
            },
            columnDefs=columnDefs,
            defaultColDef={'filter': True, 'sortable': True, 'resizable': True, 'rowDrag': True, 'editable': True},
            rowData=subjects_df.to_dict('records'),
            columnSize='autoSize',
            dashGridOptions={'pagination': True, 'paginationPageSize': 20, 'undoRedoCellEditing': True, 'rowSelection': 'single'},
        )

        show_signal_button = dbc.Button('Show Signal',
                                        id={'type': 'show-signal-events-button','index': 1},
                                        n_clicks=0,
                                        color='primary')
        
        gsc_or_ecg_radio = dbc.RadioItems(
            id={'type': 'gsc-or-ecg-radio', 'index': 1},
            options=[
                {'label': 'GSC', 'value': 'gsc'},
                {'label': 'ECG', 'value': 'ecg'}
            ],
            value='ecg'
        )


        return html.Div([ag_grid_subjects, show_signal_button, gsc_or_ecg_radio])
    
    else:
        return html.Div('Not implemented yet')
    
@callback(
    Output('events-edit-plot-container', 'children'),
    Input({'type': 'show-signal-events-button', 'index': ALL}, 'n_clicks'),
    Input({'type': 'ag-grid-subjects-events', 'index': ALL}, 'virtualRowData'),
    Input({'type': 'ag-grid-subjects-events', 'index': ALL}, 'selectedRows'),
    State('project-dropdown-events', 'value'),
    State({'type': 'gsc-or-ecg-radio', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def show_signal(n_click, virtualRows, selected_rows, project, signal_type):
    
    if n_click is None:
        raise PreventUpdate
    if not n_click:
        raise PreventUpdate
    
    if n_click[0] == 0:
        raise PreventUpdate
    
    print(n_click)
    
    print(selected_rows)
    if project == 'fibro':
        print('fibro')
        json_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates\fibro_template.json'
        
        print('loading json file')
        
        with open(json_file, 'r') as file:
            template = json.load(file)
            file.close()

        print('json file loaded')

        print(template)
        hdf_file = template['HDF5']['path']

        print('hdf file loaded')


        print(selected_rows)

        if selected_rows[0] is None:
            raise PreventUpdate
        

        subject_name = selected_rows[0][0]['Subject']
        event_name = selected_rows[0][0]['Event']
        start_time = selected_rows[0][0]['Start Time']
        end_time = selected_rows[0][0]['End Time']

        print(f'subject: {subject_name}, event: {event_name}, start time: {start_time}, end time: {end_time}')
            

        signal_table = pd.read_hdf(hdf_file, key=subject_name + '/data')


        print('signal table loaded')

        signal_table_df = pd.DataFrame(signal_table)


        print(f'signal table: {signal_table_df}')

        if signal_type == 'gsc':
            print('Loading GSC signal')
            if 'gsc' not in signal_table_df.columns:
                raise PreventUpdate
            signal_table_df['gsc'] = (signal_table_df['gsc'] - signal_table_df['gsc'].mean()) / signal_table_df['gsc'].std()
            signal = signal_table_df['gsc'].to_list()
        else:
            print('Loading ECG signal')
            if 'signal' not in signal_table_df.columns:
                raise PreventUpdate
            signal_table_df['signal'] = (signal_table_df['signal'] - signal_table_df['signal'].mean()) / signal_table_df['signal'].std()
            signal = signal_table_df['signal'].to_list()


        print('Manipulating motion signal')
        signal_table_df['motion'] = (signal_table_df['motion'] - signal_table_df['motion'].mean()) / signal_table_df['motion'].std()
        motion = signal_table_df['motion'].to_list()

        print('Manipulating time signal')
        time = signal_table_df['time'].to_list()

        # Take the time, signal and motion vectors and downsample them from 500Hz to 10Hz
        down_time = time[::50]
        down_signal = signal[::50]
        down_motion = motion[::50]

        try:
            if signal_type != 'gsc':
                print('Loading ECG rpeaks')
                rpeaks = pd.read_hdf(hdf_file, key=subject_name + '/rpeaks')
                print('Rpeaks loaded')
                rpeaks_df = pd.DataFrame(rpeaks)
                print('Rpeaks dataframe created')
                rpeaks = [int(i) for i in rpeaks_df[0]]
                print('Rpeaks list created')
                rpeaks_time = [time[i] for i in rpeaks]
                print('Rpeaks time list created')
        except:
            print('No rpeaks found or signal type is not ECG')
            rpeaks = None
            rpeaks_time = None

        print('Loading events data')
        subject_event_start_end_times = {}

        print('Loading events data from hdf file')
        events_df = pd.DataFrame(virtualRows[0])

        print('Events dataframe created')
        events_data = {}

        for i in range(len(events_df)):
            subject = events_df['Subject'][i]
            event = events_df['Event'][i]
            start_times = events_df['Start Time'][i]
            end_times = events_df['End Time'][i]
            if subject not in subject_event_start_end_times.keys():
                subject_event_start_end_times[subject] = {}
            subject_event_start_end_times[subject][event] = [start_times, end_times]
            events_data[event] = {'Start_Time': start_times, 'End_Time': end_times}

        print('Events data loaded')


        if signal_type == 'gsc':
            print('Loading GSC attributes')
            with h5py.File(hdf_file, 'r') as file:
                gsc_attrs = json.loads(file[subject_name]['data'].attrs['gsc_attributes'].decode('utf-8'))
                file.close()
                print('GSC attributes loaded')
        else:
            print('Loading ECG attributes')
            with h5py.File(hdf_file, 'r') as file:
                try:
                    ecg_attrs = json.loads(file[subject_name]['rpeaks'].attrs['signal_attributes'].decode('utf-8'))
                except:
                    ecg_attrs = json.loads(file[subject_name]['rpeaks'].attrs['signal_attributes'])
                file.close()
                print('ECG attributes loaded')
                print(f'ECG attributes: {ecg_attrs}')



        color_list = [ 'green', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'salmon', 'gold', 'lightblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightcoral', 'darkcoral', 'lightgold', 'darkgold', 'lightblue', 'darkblue', 'lightcyan', 'darkcyan', 'lightmagenta', 'darkmagenta', 'lightlime', 'darklime', 'lightteal', 'darkteal', 'lightlavender', 'darklavender', 'lightsalmon', 'darksalmon', 'lightgray', 'darkgray', 'lightpink', 'darkpink', 'lightbrown']

        print('Creating shapes for events')
        shapes_not_selected = [{
            'type': 'rect',
            'x0': subject_event_start_end_times[subject_name][event][0],
            'y0': 0,
            'x1': subject_event_start_end_times[subject_name][event][1],
            'y1': 1,
            'yref': 'paper',
            'fillcolor': color_list[i],
            'opacity': 0.3 if event != event_name else 0.7,
            'line': {'width': 0 if event != event_name else 1},
            'layer': 'below'
        } for i, event in enumerate(subject_event_start_end_times[subject_name].keys())]



        if signal_type == 'gsc':
            print('Creating GSC plot')
            try:
                print('Manipulating EDA signal components')
                eda_tonic = signal_table_df['eda_tonic'].to_list()
                eda_phasic = signal_table_df['eda_phasic'].to_list()

                down_tonic = eda_tonic[::50]
                down_phasic = eda_phasic[::50]

                print('Loading SCR components')
                scr_onsets_index = gsc_attrs['SCR_Onsets']
                scr_onsets = [time[i] for i in scr_onsets_index]
                
                scr_peaks_index = gsc_attrs['SCR_Peaks']
                scr_peaks = [time[i] for i in scr_peaks_index]

                scr_recovery_index = gsc_attrs['SCR_Recovery']
                scr_recovery = [time[i] for i in scr_recovery_index]

                print('Creating GSC plot')
                graph = dcc.Graph(
                    id={
                        'type': 'gsc-plot',
                        'index': 1
                    },
                    figure={
                        'data': [
                            go.Scatter(
                                x=down_time,
                                y=down_motion,
                                name='Motion',
                                line=dict(color='black'),
                                hovertext=[f'time: {time[i]}, <br>motion: {motion[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=down_time,
                                y=down_signal,
                                name='GSC',
                                mode='lines',
                                line = dict(color='blue'),
                                hovertext=[f'time: {time[i]}, <br>gsc: {signal[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=down_time,
                                y=down_tonic,
                                name='EDA Tonic',
                                mode='lines',
                                line=dict(color='green'),
                                hovertext=[f'time: {time[i]}, <br>eda_tonic: {eda_tonic[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=down_time,
                                y=down_phasic,
                                name='EDA Phasic',
                                mode='lines',
                                line=dict(color='red'),
                                hovertext=[f'time: {time[i]}, <br>eda_phasic: {eda_phasic[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=scr_onsets,
                                y=[signal[i] for i in scr_onsets_index],
                                name='SCR Onsets',
                                mode='markers',
                                marker=dict(color='green'),
                                hovertext=[f'time: {scr_onsets[i]}, <br>gsc: {signal[i]}' for i in range(len(scr_onsets))]
                            ),
                            go.Scatter(
                                x=scr_peaks,
                                y=[signal[i] for i in scr_peaks_index],
                                name = 'SCR Peaks',
                                mode='markers',
                                marker=dict(color='red'),
                                hovertext=[f'time: {scr_peaks[i]}, <br>gsc: {signal[i]}' for i in range(len(scr_peaks))]
                            ),
                            go.Scatter(
                                x=scr_recovery,
                                y=[signal[i] for i in scr_recovery_index],
                                name = 'SCR Recovery',
                                mode='markers',
                                marker=dict(color='blue'),
                                hovertext=[f'time: {scr_recovery[i]}, <br>gsc: {signal[i]}' for i in range(len(scr_recovery))]
                            )
                        ],
                        'layout': go.Layout(
                            title='GSC Signal',
                            clickmode='event',
                            shapes= shapes_not_selected,
                            annotations=[{
                                'x': ((subject_event_start_end_times[subject_name][event][1] - subject_event_start_end_times[subject_name][event][0]) / 2) + subject_event_start_end_times[subject_name][event][0],
                                'y': 1,
                                'yref': 'paper',
                                'text': event,
                                'showarrow': False,
                                'yshift': (i % 2) * 30,
                            } for i, event in enumerate(subject_event_start_end_times[subject_name].keys())],
                            xaxis=dict(
                                tickangle=45
                            )
                        )}
                )
                print('GSC plot created')
            except:
                print('No SCR components found')
                graph = dcc.Graph(
                    id={
                        'type': 'gsc-plot',
                        'index': 1
                    },
                    figure={
                        'data': [
                            go.Scatter(
                                x=down_time,
                                y=down_motion,
                                name='Motion',
                                line=dict(color='black'),
                                hovertext=[f'time: {time[i]}, <br>motion: {motion[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=down_time,
                                y=down_signal,
                                name='GSC',
                                mode='lines',
                                line = dict(color='blue'),
                                hovertext=[f'time: {time[i]}, <br>gsc: {signal[i]}' for i in range(len(time))]
                            )
                        ],
                        'layout': go.Layout(
                            title='GSC Signal',
                            shapes= shapes_not_selected,
                            annotations=[{
                                'x': ((subject_event_start_end_times[event][1] - subject_event_start_end_times[event][0]) / 2) + subject_event_start_end_times[event][0],
                                'y': 1,
                                'yref': 'paper',
                                'text': event,
                                'showarrow': True,
                                'yshift': (i % 2) * 30,
                            } for i, event in enumerate(subject_event_start_end_times[subject_name].keys())],
                            xaxis=dict(
                                tickangle=45
                            )
                        )}
                )
                print('GSC plot created')

        else:
            try:
                print('Manipulating ECG signal components')
                rpeaks_time = [time[i] for i in rpeaks]
                print(f'rpeaks time: {rpeaks_time[0:10]}')
                print(f'rpeaks: {rpeaks[0:10]}')

                print(f'Time: {time[0:10]}')
                print(f'Time from the end: {time[-10:]}')

                print(f'Time vector length: {len(time)}')

                print('Loading yellow rpeaks')
                print(f'yellow rpeaks: {ecg_attrs}')
                yellow_rpeaks_mapped = ecg_attrs['yellow_peaks']
                print(f'yellow rpeaks mapped: {yellow_rpeaks_mapped}')
                for i in range(len(yellow_rpeaks_mapped)):
                    yellow_rpeaks_mapped[i] = int(yellow_rpeaks_mapped[i])
                    print(f'yellow rpeaks mapped: {yellow_rpeaks_mapped[i]}')
                    print(f'yellow rpeaks time: {time[yellow_rpeaks_mapped[i]]}')
                    
                yellow_rpeaks = [time[i] for i in yellow_rpeaks_mapped]
                print('Yellow rpeaks loaded')


                print('Creating ECG plot')
                graph = dcc.Graph(
                    id={
                        'type': 'ecg-plot',
                        'index': 1
                    },
                    figure={
                        'data': [
                            go.Scatter(
                                x=down_time,
                                y=down_motion,
                                name='Motion',
                                mode='lines',
                                line=dict(color='black'),
                                hovertext=[f'time: {time[i]}, <br>motion: {motion[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=down_time,
                                y=down_signal,
                                name='ECG',
                                mode='lines',
                                line=dict(color='blue'),
                                hovertext=[f'time: {time[i]}, <br>ecg: {signal[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=rpeaks_time,
                                y=[signal[i] for i in rpeaks],
                                name='Rpeaks',
                                mode='markers',
                                marker=dict(color='red'),
                                hovertext=[f'time: {rpeaks_time[i]}, <br>rpeaks: {signal[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=yellow_rpeaks,
                                y=[signal[i] for i in yellow_rpeaks_mapped],
                                name='Yellow Rpeaks',
                                mode='markers',
                                marker=dict(color='yellow'),
                                hovertext=[f'time: {yellow_rpeaks[i]}, <br>yellow_rpeaks: {signal[i]}' for i in range(len(yellow_rpeaks))]
                            )
                        ],
                        'layout': go.Layout(
                            title='ECG Signal',
                            clickmode='event',
                            shapes= shapes_not_selected,
                            annotations=[{
                                'x': ((subject_event_start_end_times[subject_name][event][1] - subject_event_start_end_times[subject_name][event][0]) / 2) + subject_event_start_end_times[subject_name][event][0],
                                'y': 1,
                                'yref': 'paper',
                                'text': event,
                                'showarrow': False,
                                'yshift': (i % 2) * 30,
                            } for i, event in enumerate(subject_event_start_end_times[subject_name].keys())],
                            xaxis=dict(
                                tickangle=45
                            )
                        )}
                )
                print('ECG plot created')
            except Exception as e:
                print('No rpeaks found')
                graph = dcc.Graph(
                    id={
                        'type': 'ecg-plot',
                        'index': 1
                    },
                    figure={
                        'data': [
                            go.Scatter(
                                x=down_time,
                                y=down_signal,
                                name='ECG',
                                mode='lines',
                                line=dict(color='blue'),
                                hovertext=[f'time: {time[i]}, <br>ecg: {signal[i]}' for i in range(len(time))]
                            )
                        ],
                        'layout': go.Layout(
                            title='ECG Signal',
                            clickmode='event',
                            shapes= shapes_not_selected,
                            annotations=[{
                                'x': ((subject_event_start_end_times[subject_name][event][1] - subject_event_start_end_times[subject_name][event][0]) / 2) + subject_event_start_end_times[subject_name][event][0],
                                'y': 1,
                                'yref': 'paper',
                                'text': event,
                                'showarrow': False,
                                'yshift': (i % 2) * 30,
                            } for i, event in enumerate(subject_event_start_end_times[subject_name].keys())],
                            xaxis=dict(
                                tickangle=45
                            )
                        )}
                )
                print('ECG plot created')


        print('Creating graph layout')
        start_time = down_time.index(round(start_time, 1))
        end_time = down_time.index(round(end_time, 1))
        graph_layout = html.Div(
            [
                graph,
                dbc.Row([
                    dbc.Col([
                        dcc.RangeSlider(
                            id={'type': 'range-slider-events', 'index': 1},
                            min=0,
                            max=len(down_time),
                            step=0.1,
                            value=[start_time, end_time],
                            marks={i: {'label': str(down_time[i])} for i in range(0, len(down_time), 5000)}
                        )
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Save Changes',
                                    id={'type': 'save-changes-events-button', 'index': 1},
                                    n_clicks=0,
                                    color='primary'),
                                    dcc.Loading(
                                        [
                                            dcc.ConfirmDialog(
                                                id={'type': 'confirm-dialog-events', 'index': 1},
                                                message='',
                                            )],
                                            overlay_style={'visibility': 'visible',
                                                           'opacity': 0.5,
                                                           'background': 'white'},
                                                           custom_spinner= html.H2(['Saving changes...', dbc.Spinner(color='primary')]),
                                    ),
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader('Are you sure you want to save changes?'),
                                            dbc.ModalBody([
                                                html.Div('Please make sure that the changes you made are correct before saving.'),
                                                dbc.Button('Save Changes', 
                                                            id={'type': 'confirm-save-changes-events-button', 'index': 1},
                                                            style={'margin': 'auto', 'textAlign': 'center'},
                                                )
                                            ]),
                                            dbc.ModalFooter(
                                                dbc.Button('Close', 
                                                            id={'type': 'close-modal-events-button', 'index': 1},
                                                            style={'margin': 'auto', 'textAlign': 'center'}
                                                )
                                            ),
                                        ],
                                        id={'type': 'modal-events', 'index': 1},
                                        is_open=False,
                                    ),
                                    dcc.ConfirmDialog(
                                        id={'type': 'no-changes-events', 
                                            'index': 1},
                                        message='No changes were made.',
                                    )
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id={'type': 'events-changes', 
                                     'index': 1},
                                     children=[])
                    ])
                ])



            ]     
        )
        print('Graph layout created')


        # Save the graph object as json file
        with open(r'graph.json', 'w') as file:
            file.write(json.dumps(dict(graph)))
            file.close()

        return graph_layout
    else:
        return html.Div('Not implemented yet')
    

# @callback(
#     Output({'type': 'ecg-plot', 'index': ALL}, 'figure'),
#     Output({'type': 'gsc-plot', 'index': ALL}, 'figure'),
#     Output({'type': 'range-slider-events', 'index': ALL}, 'value'),
#     Output({'type': 'subject-AgGrid-events', 'index': ALL}, 'virtualRowData'),
#     Output({'type': 'events-changes', 'index': ALL}, 'children'),
#     Input('events-edit-store', 'modified_timestamp'),
#     State('events-edit-store', 'data'),
#     prevent_initial_call=True
# )
# def update_plot(ts, data):
#     if data == []:
#         raise PreventUpdate
    

#     new_shapes = []

#     for subject in data.keys():
#         if data[subject]['']



#     patched_shapes = Patch()
#     patched_shapes['layout']['shapes'] = 


# listener for range slider
@callback(
    Output('events-edit-store', 'data'),
    Input({'type': 'range-slider-events', 'index': ALL}, 'value'),
    Input('events-edit-store', 'modified_timestamp'),
    State('events-edit-store', 'data'),
    State({'type': 'ag-grid-subjects-events', 'index': ALL}, 'selectedRows'),
    prevent_initial_call=True
)
def store_changes(value, ts, data, selected_rows):
    if data == []:
        raise PreventUpdate

    if value[0] == data:
        raise PreventUpdate
    
    if selected_rows[0] is None:
        raise PreventUpdate

    data = value[0]

    with open(r'slider_data.txt', 'w') as file:
        file.write(str(value))
        file.close()

    return data



# save changes button - modal 
@callback(
    Output({'type': 'modal-events', 'index': ALL}, 'is_open'),
    Output({'type': 'no-changes-events', 'index': ALL}, 'displayed'),
    Input({'type': 'save-changes-events-button', 'index': ALL}, 'n_clicks'),
    State('events-edit-store', 'data'),
    prevent_initial_call=True
)
def save_changes(n_click, data):

    if n_click is None:
        raise PreventUpdate

    if n_click[0] == 0:
        raise PreventUpdate


    if data == []:
        return [False], [True]

    return [True], [False]
    



                

                            










            


