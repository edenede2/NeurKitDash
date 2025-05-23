from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
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
from pymongo import MongoClient
import dash

dash.register_page(__name__, path='/')


# # Initialize Flask app
# flask_app = Flask(__name__)




# # Initialize Dash app
# dash_app = Dash(__name__, server=flask_app, external_stylesheets=[dbc.themes.DARKLY])

# window = webview.create_window('Mindware App', flask_app, width=1200, height=800, resizable=True, fullscreen=False)

print('Dash app created')
layout = html.Div([
    html.H1('Mindware new app'),
    html.Div([
        dcc.Markdown('''
        Welcome to the Mindware app
                        
        Please upload the signal and events files to start analyzing the data.
        
        i.e. signal file: `sub_(3 digit number)_data.txt` 
                        
        i.e. events file: `sub_(3 digit number)_events.txt`
                        
        The signal file should contain the following columns:
        - Time (s)
        - Bio (ECG)
        - X axis
        - Y axis
        - Z axis (optional)
        - GSC (optional)
                        
        The events file should contain the following columns:
        - Time
        - Event Type
                        
        For more information, please refer to Eden via Gmail.

        `edenede2@gmail.com`

        Enjoy!
        '''),
    ]),
                        
    # html.Div([
    #     dcc.Tabs(id='tabs', value='tab-1', children=[
    #         dcc.Tab(label='Tab 1', value='tab-1'),
    #         dcc.Tab(label='Tab 2', value='tab-2')
    #     ]),
    #     html.Div(id='tabs-content')
    # ]),

    html.H2(children='Select Project'),
    dcc.Dropdown(
        id='project-dropdown',
        options=[
            {'label': 'Fibro', 'value': 'fibro'},
            {'label': 'Driving&Stress', 'value': 'driving'},
            {'label': 'RDoC', 'value': 'rdoc'}
        ],
        style={'width': '50%', 'color': 'black'},
        value='fibro'
    ),

    html.Div([
        dcc.Loading(id='loading-subjects-output', type='default', children=[html.Div(id='loading-subjects-output')]),
    ]),

    html.Div(id='subjects-container', children=[]),

    html.Div([
        dcc.Store(id='subject-store', data={}),
    ]),

    html.H2(children='', id='project-markdown'),
    html.H3(children='', id='channel-markdown'),

    html.Div([
        dcc.Loading(id='loading-events-table', type='default', children=[html.Div(id='loading-events-table-output')]),
    ]),

    html.Div(id='events-table-container', children=[]),

    html.Br(),
    html.Br(),

    html.Div([
        dcc.Loading(id='loading-whole-signal', type='default', children=[html.Div(id='loading-whole-signal-output')]),
    ]),

    html.Div(id='whole-signal-container', children=[]),
    html.Div(id='signal-cards-container', children=[]),
    html.Div([
        dcc.Store(id='whole-signal-store', data={}),
        dcc.Store(id='whole-signal-filename', data=''),
        dcc.Store(id='events-store', data={}),
        dcc.Store(id='events-filename', data=''),
        dcc.Store(id='signal-cards-store', data={})
    ]),
])

# @callback(
#     Output('loading-subjects-output', 'children'),
#     Output('subjects-container', 'children'),
#     Output('subject-store', 'data'),
#     [Input('project-dropdown', 'value')]
# )
# def display_subjects(project):
    
#     if project:
#         print(f"Selected project: {project}")
#         with open(f'pages/data/mindwareData/{project}_mindware.json') as f:
#             project_data = json.load(f)
#             f.close()
#     else:
#         print("No project selected")
#         project_data = {}

#     if project_data == {}:
#         print("No subjects found")
#         return 'No subjects found', [], {}
    
#     subjects = list(project_data.keys())
#     print(f"Subjects: {subjects}")
#     subjects.sort()

#     subjects_df = pd.DataFrame(subjects, columns=['Subject'])
#     subjects_df['Quality'] = [x['signal']['quality_value_overall'] for x in project_data.values()]
#     subjects_df['Quality'] = subjects_df['Quality'].apply(lambda x: f'{x:.2f}%')
#     subjects_df['Artifacts'] = [x['signal']['artifacts_percentage_overall'] for x in project_data.values()]
#     subjects_df['Artifacts'] = subjects_df['Artifacts'].apply(lambda x: f'{x:.2f}%')
#     subjects_df['Num of Events'] = [len(x['events']['events']) for x in project_data.values()]

#     subjects_columns = [{'name': i, 'id': i, 'editable': True} for i in subjects_df.columns]
#     subjects_data = subjects_df.to_dict('records')

#     print(f"Subjects DataFrame: {subjects_df.head()}")

#     return 'Subjects:', html.Div(
#         style={'width': '95%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
#         children=[
#         dash_table.DataTable(
#             id={
#                 'type': 'subjects-table',
#                 'index': 1
#                 },
#             columns=subjects_columns,
#             data=subjects_data,
#             editable=False,
#             filter_action='native',
#             sort_action='native',
#             sort_mode='multi',
#             column_selectable='single',
#             row_deletable=False,
#             row_selectable='single',
#             selected_rows=[],
#             selected_columns=[],
#             page_action='native',
#             page_current=0,
#             page_size=10,
#             style_data={
#                 'backgroundColor': 'white',
#                 'color': 'black'
#             },
#             style_header={
#                 'backgroundColor': 'white',
#                 'color': 'black'
#             },
#             style_cell_conditional=[
#                 {
#                     'if': {'column_id': c},
#                     'textAlign': 'left'
#                 } for c in ['Subject']
#             ],
#             style_data_conditional=[
#                 {
#                     'if': {
#                         'filter_query': '{% Artifacts} = "0.00%"',
#                             'column_id': 'Artifacts'}, 'backgroundColor': 'lightgreen', 'color': 'black'},
#                     {'if': {'filter_query': '{% Artifacts} != "0.00%"', 'column_id': 'Artifacts'}, 'backgroundColor': 'lightorange', 'color': 'black'},
#                     {'if': {'column_id': 'Subject'}, 'backgroundColor': 'white', 'color': 'red'}
#                 ]
#             ),
#             html.Button('Show subject data', id={'type': 'show-subject-data', 'index': 1}
#                         )
#         ]
#     ), project_data

@callback(
    [Output('project-markdown', 'children'),
     Output('channel-markdown', 'children'),
     Output('whole-signal-store', 'data'),
     Output('events-store', 'data')],
    [Input({'type': 'show-subject-data', 'index': ALL}, 'n_clicks'),
     Input({'type': 'subjects-table', 'index': ALL}, 'derived_virtual_selected_rows'),
     Input('project-dropdown', 'value'),
     Input({'type': 'show-subject-data', 'index': ALL}, 'data')],
)
def upload_signal(show_data ,selected_row, project, data):
    
    if selected_row is None:
        raise PreventUpdate
    
    if selected_row[0] is None:
        raise PreventUpdate

    if show_data[0] == 0 or show_data[0] is None:
        raise PreventUpdate    

    print(f"Selected project1: {project}")
    # client = MongoClient('mongodb+srv://edenEldar:Eden1996@cluster0.rwebk7f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    # Open the json file of the template from the folder templates
    with open(f'pages/templates/{project}_template.json') as f:
        project_template = json.load(f)

        sampling_rate = project_template['parameters']['sampling_rate']
        events_mapping = project_template['events']
        channels = project_template['channel']
        if project_template.get('blocks_shorts'):
            blocks_shorts = project_template['blocks_shorts']

        # close the file
        f.close()

    print(f"Project template: {project_template}")
    project_markdown = f"""
    Project: {project}
    """

    channel_markdown = f"""
    Channels: {channels}
    """



        
        # # mindware_db = client['MindWare']
        # # print(f"Selected database: {mindware_db}")
        # project_params = mindware_db[selected_project].find_one()
        # print(f"Project params: {project_params}")
        # sampling_rate = project_params['parameters']['sampling_rate']
        # events_mapping = project_params['events']
        # print(f"Events mapping: {events_mapping}")
        # channels = project_params['channel']
        # blocks_shorts = project_params.get('blocks_shorts')
        # print(f"Mongodb data: {sampling_rate}, {events_mapping}, {channels}, {blocks_shorts}")
        # print("MongoDB data fetched successfully")
        # client.close()
        # project_markdown = f"""
        # Project: {selected_project}
        # """

    channel_markdown = f"""
    Channels: {channels}
    """


    # signal_content = json.loads('pages/data/mindware_data/{project}_mindware.json')

    # print('Start uploading signal')
    json_folder = r'pages/data/mindwareData'

    # check if there is a json file in the folder
    if os.path.exists(json_folder):
        json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
        print(f"Json files: {json_files}")
        for json_file in json_files:
            if project in json_file:
                with open(f'{json_folder}/{json_file}') as f:
                    project_data = json.load(f)
                    print(f"Project data exists: {json_file}")
                    f.close()
                    break
            else:
                project_data = {}
    else:
        project_data = {}

    print(f"Selected row: {selected_row}")

    if project_data == {}:
        return project_markdown, channel_markdown, {}, {}
    
    selected_subject = selected_row[0][0]

    # convert project data to a DataFrame
    project_data = pd.DataFrame(project_data)
    project_data = project_data.to_dict('index')

    print(f"Project data: {project_data}")

    print(f"Selected subject: {selected_subject}")
        
    if 'signal' in project_data[selected_subject].keys():
        cleaned_signal = project_data[selected_subject]['signal']['signal']
        time = project_data[selected_subject]['signal']['time']
        motion = project_data[selected_subject]['signal']['motion']
        rpeaks_indices = project_data[selected_subject]['signal']['rpeaks']
        peaks_color = project_data[selected_subject]['signal']['peaks_color']
        quality_rpeaks = project_data[selected_subject]['signal']['quality']
        quality = project_data[selected_subject]['signal']['ecg_quality']
        ectopic_beats = project_data[selected_subject]['signal']['ectopic_beats']
        longshort_beats = project_data[selected_subject]['signal']['longshort_beats']
        false_negatives = project_data[selected_subject]['signal']['false_negatives']
        false_positives = project_data[selected_subject]['signal']['false_positives']
        if 'gsc' in project_data[selected_subject]['signal'].keys():
            gsc_cleaned = project_data[selected_subject]['signal']['gsc']
            gsc_peaks = project_data[selected_subject]['signal']['gsc_peaks']
            eda_phasic_df = project_data[selected_subject]['signal']['eda_phasic']
        new_events_data = project_data[selected_subject]['events']

        print("Signal parameters loaded successfully")

    else:
        print("No signal data found")
        return project_markdown, channel_markdown, {}, {}

    stored_signal = {
        'signal': cleaned_signal,
        'time': time,
        'motion': motion,
        'rpeaks': rpeaks_indices,
        'peaks_color': peaks_color,
        'updated_rpeaks_color': peaks_color,
        'quality': quality_rpeaks,
        'ecg_quality': quality,
        'ectopic_beats': ectopic_beats,
        'longshort_beats': longshort_beats,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'updated_rpeaks': rpeaks_indices,
        'gsc': gsc_cleaned if 'gsc' in project_data[selected_subject]['signal'].keys() else None,
        'gsc_peaks': gsc_peaks['SCR_Peaks'] if 'gsc' in project_data[selected_subject]['signal'].keys() else None,
        'updated_gsc_peaks': gsc_peaks['SCR_Peaks'] if 'gsc' in project_data[selected_subject]['signal'].keys() else None,
        'eda_phasic': eda_phasic_df['EDA_Phasic'] if 'gsc' in project_data[selected_subject]['signal'].keys() else None,
        'eda_tonic': eda_phasic_df['EDA_Tonic'] if 'gsc' in project_data[selected_subject]['signal'].keys() else None,
        'gsc_peaks_color': ['red' for _ in gsc_peaks['SCR_Peaks']] if 'gsc' in project_data[selected_subject]['signal'].keys() else None,
        'updated_gsc_peaks_color': ['red' for _ in gsc_peaks['SCR_Peaks']] if 'gsc' in project_data[selected_subject]['signal'].keys() else None
    }
    stored_events = {'events': new_events_data}

    print("Signal stored successfully")
    return project_markdown, channel_markdown, stored_signal, stored_events

@callback(
    [Output('loading-events-table-output', 'children'),
    Output('events-table-container', 'children')],
    [Input('events-store', 'data'),
     Input({'type': 'subjects-table', 'index': 1}, 'selected_rows')],
    State('whole-signal-store', 'data')
)
def display_events_table(events_data, selected_subject ,whole_signal_data):
    if not events_data:
        raise PreventUpdate
    
    if selected_subject is None:
        raise PreventUpdate
    
    print("Start displaying events table")
    print(f"Events data: {events_data}")

    events_data = events_data['events']
    events_df = pd.DataFrame(events_data)
    

    
    events_columns = [{'name': i, 'id': i, 'editable': True} for i in events_df.columns]
    events_data = events_df.to_dict('records')

    

    loading_message = 'Events table:'

    # events_df = pd.DataFrame(events_data)
    # events_columns = [{'name': i, 'id': i, 'editable': True} for i in events_df.columns]
    return loading_message, html.Div(
        style={'width': '95%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
        children=[
        dash_table.DataTable(
            id={
                'type':'events-table',
                'index': 1
                },
            columns=events_columns,
            data=events_data,
            editable=True,
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            row_selectable='multi',
            selected_rows=[],
            selected_columns=[],
            page_action='native',
            page_current=0,
            page_size=10,

            style_data={
                'backgroundColor': 'white',
                'color': 'black'
            },
            style_header={
                'backgroundColor': 'white',
                'color': 'black'
            },
            style_cell_conditional=[
                # align text columns to left. By default they are aligned to right
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ['Start_Flag', 'End_Flag', 'Classification']
            ],
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{% Artifacts} = "0%"',
                        'column_id': '% Artifacts'
                    },
                    'backgroundColor': 'lightgreen',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{% Artifacts} != "0%"',
                        'column_id': '% Artifacts'
                    },
                    'backgroundColor': 'lightorange',
                    'color': 'black'
                },
                {
                    'if': {'column_id': 'Classification'},
                    'backgroundColor': 'white',
                    'color': 'red'
                }
            ]
        
        ),
        html.Button('Apply Events',
                    id={
                        'type': 'apply-events',
                        'index': 1
                    }
        ),
        html.Button('Show Whole Signal',
                    id={
                        'type': 'show-whole-signal',
                        'index': 1
                    }
        ),
        html.Button('Refresh Events',
                    id={
                        'type': 'refresh-events',
                        'index': 1
                    }
        )
    ])

@callback(
    [Output('whole-signal-container', 'children'),
    Output('loading-whole-signal-output', 'children')],
    [Input({'type': 'show-whole-signal', 'index': ALL}, 'n_clicks')],
    [State('events-store', 'data'),
    State('whole-signal-store', 'data'),
    State('events-store', 'data'),
    State('whole-signal-filename', 'data')]
)
def display_whole_signal(show,events_data, whole_signal_data, events_store, whole_signal_filename):
    
    if not 'events' in events_data.keys() and not whole_signal_data:
        raise PreventUpdate
    
    if show[0] == 0 or show[0] is None:
        raise PreventUpdate

    print("Start displaying whole signal")

    # selected_row = derived_virtual_selected_rows_state[0]
    color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'salmon', 'gold', 'lightblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightcoral', 'darkcoral', 'lightgold', 'darkgold', 'lightblue', 'darkblue', 'lightcyan', 'darkcyan', 'lightmagenta', 'darkmagenta', 'lightlime', 'darklime', 'lightteal', 'darkteal', 'lightlavender', 'darklavender', 'lightsalmon', 'darksalmon', 'lightgray', 'darkgray', 'lightpink', 'darkpink', 'lightbrown', 'darkbrown', 'lightblack', 'darkblack']


    
    print(f"Derived virtual selected rows: {events_data}")
    print("Start displaying whole signal")

    # Print the data to debug
    # print(f"Data: {data}")
    # print(f"Whole signal data: {whole_signal_data}")
    print(f"Whole signal filename: {whole_signal_filename}")

    # Create a DataFrame from the data
    # if events_data and isinstance(events_data[0], list):
    #     # Flatten the list of lists
    #     records = [item for sublist in events_data for item in sublist]
    # else:

    if 'events' in events_data.keys():
        events_data = events_data['events']
    else:
        # Handle the case where 'events' is not in events_data
        # For example, you might want to set events_data to an empty list
        events_data = []

    records = events_data

    events_df = pd.DataFrame(records)
    print(f"Events DataFrame: {events_df.head()}")

    # Check for column names
    if 'Start_Time' not in events_df.columns or 'End_Time' not in events_df.columns or 'Classification' not in events_df.columns:
        print("Column names do not match expected values.")
        raise PreventUpdate

    # Assign colors
    # colors = ['#90BDE1' if i in derived_virtual_selected_rows else 'white' for i in range(len(events_df))]
    

    sub_name = whole_signal_filename.split('.')[0]  
    loading_message = f'Whole signal for {sub_name}:'

    signal = whole_signal_data['signal']
    time = whole_signal_data['time']
    motion = whole_signal_data['motion']
    rpeaks = whole_signal_data['rpeaks']
    peaks_color = whole_signal_data['peaks_color']
    quality = whole_signal_data['quality']
    ecg_quality = whole_signal_data['ecg_quality']

    ectopic_beats = whole_signal_data['ectopic_beats']
    longshort_beats = whole_signal_data['longshort_beats']
    false_negatives = whole_signal_data['false_negatives']
    false_positives = whole_signal_data['false_positives']


    rpeaks_time = [time[i] for i in rpeaks]

    # Debugging prints for signal data
    print(f"Signal data length: {len(signal)}")
    print(f"Time data length: {len(time)}")
    # print(f"Derived virtual selected rows: {derived_virtual_selected_rows}")
    # print(f"Derived virtual indices: {derived_virtual_indices}")
    # print(f"Derived virtual row ids: {derived_virtual_row_ids}")
    # print(f"Active cell: {active_cell}")
    # print(f"Selected cells: {selected_cells}")
    # print(f"Rpeaks: {rpeaks}")
    # print(f"Rpeaks time: {rpeaks_time}")

    if 'gsc' in whole_signal_data.keys():
        gsc = whole_signal_data['gsc']
        gsc_peaks = whole_signal_data['gsc_peaks']
        gsc_peaks_time = [time[i] for i in gsc_peaks]
        eda_phasic = whole_signal_data['eda_phasic']
        eda_tonic = whole_signal_data['eda_tonic']
        gsc_peaks_color = ['red' for _ in gsc_peaks]
        for i in range(len(events_df)):
            events_df['Start_Time'] = pd.to_numeric(events_df['Start_Time'], errors='coerce')
            events_df['End_Time'] = pd.to_numeric(events_df['End_Time'], errors='coerce')
            print(f"Rectangle length: {((events_df.loc[i, 'Start_Time'] + events_df.loc[i, 'End_Time']) / 2)}")
        graph_layout = html.Div(
            style={'width': '90%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
            children=[
                dcc.Graph(
                    id={'type': 'ecg-plot', 'index': 1},
                    figure={
                        'data': [
                            go.Scatter(
                                x=time,
                                y=signal,
                                mode='lines',
                                name='ECG',
                                line=dict(color='blue'),
                                hovertext=[f'Time: {time[i]}<br>Quality: {ecg_quality[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=rpeaks_time,
                                y=[signal[i] for i in rpeaks],
                                mode='markers',
                                name='R-peaks',
                                marker=dict(color=peaks_color),
                                hovertext=[f'Time: {rpeaks_time[i]}<br>Quality: {quality[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=rpeaks_time,
                                y=[signal[i] for i in ectopic_beats],
                                mode='markers',
                                name='Ectopic Beats',
                                marker=dict(symbol='x', color='red', size=10),
                                hovertext=[f'Time: {rpeaks_time[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=rpeaks_time,
                                y=[signal[i] for i in longshort_beats],
                                mode='markers',
                                name='Long/Short Beats',
                                marker=dict(symbol='x', color='green', size=10),
                                hovertext=[f'Time: {rpeaks_time[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=rpeaks_time,
                                y=[signal[i] for i in false_negatives],
                                mode='markers',
                                name='False Negatives',
                                marker=dict(symbol='x', color='yellow', size=10),
                                hovertext=[f'Time: {rpeaks_time[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=rpeaks_time,
                                y=[signal[i] for i in false_positives],
                                mode='markers',
                                name='False Positives',
                                marker=dict(symbol='x', color='purple', size=10),
                                hovertext=[f'Time: {rpeaks_time[i]}' for i in range(len(rpeaks_time))]
                            ),
                            go.Scatter(
                                x=time,
                                y=[x*0.1 for x in motion],
                                mode='lines',
                                name='Motion',
                                line=dict(color='green'),
                                hovertext=[f'Time: {time[i]}' for i in range(len(time))]
                            ),
                            go.Scatter(
                                x=[events_df.loc[i, 'Start_Time'] for i in range(len(events_df))],
                                y=[min(signal) for _ in range(len(events_df))],
                                mode='markers',
                                marker=dict(size=0, opacity=0),
                                hoverinfo='text',
                                hovertext=[f'{events_df.loc[i, "Classification"]}' for i in range(len(events_df))]
                            ),
                            go.Scatter(
                                x=[events_df.loc[i, 'End_Time'] for i in range(len(events_df))],
                                y=[max(signal) for _ in range(len(events_df))],
                                mode='markers',
                                marker=dict(size=0, opacity=0),
                                hoverinfo='text',
                                hovertext=[f'{events_df.loc[i, "Classification"]}' for i in range(len(events_df))]
                            
                            )],
                        'layout': go.Layout(
                            title='ECG Signal',
                            clickmode='event',
                            shapes=[{
                                
                                'type': 'rect',
                                'x0': events_df.loc[i, 'Start_Time'],
                                'x1': events_df.loc[i, 'End_Time'],
                                'y0': min(signal),
                                'y1': max(signal),
                                'fillcolor': color_list[i],
                                'opacity': 0.3,
                                'line': {'width': 0},
                                'layer': 'below'
                            } for i in range(len(events_df))],
                            annotations = [
                                {
                                    'x': ((events_df.loc[i, 'Start_Time'] + events_df.loc[i, 'End_Time']) / 2),
                                    'y': max(signal),
                                    'text': events_df.loc[i, 'Classification'],
                                    'showarrow': False,
                                    'textangle': 45,
                                    'yshift': (i % 2) * 30  # Stagger the labels
                                } for i in range(len(events_df))],
                            
                            xaxis=dict(
                                tickangle=45  # Rotate x-axis labels
                            )
                            )}
                    ),
                
                dcc.Graph(
                    style={'width': '90%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
                    id={'type': 'gsc-plot', 'index': 1},
                    figure={
                        'data': [
                            go.Scatter(
                                x=time,
                                y=gsc,
                                mode='lines',
                                name='GSC',
                                line=dict(color='blue')
                            ),
                            go.Scatter(
                                x=gsc_peaks_time,
                                y=[gsc[i] for i in gsc_peaks],
                                mode='markers',
                                name='SCR Peaks',
                                marker=dict(color=gsc_peaks_color),
                                hovertext=[f'Time: {gsc_peaks_time[i]}' for i in range(len(gsc_peaks_time))]
                                
                            ),
                            go.Scatter(
                                x=time,
                                y=eda_tonic,
                                mode='lines',
                                name='EDA Tonic',
                                line=dict(color='orange')
                            ),
                            go.Scatter(
                                x=time,
                                y=eda_phasic,
                                mode='lines',
                                name='EDA Phasic',
                                line=dict(color='purple')
                            ),
                            go.Scatter(
                                x=time,
                                y=[x*100 for x in motion],
                                mode='lines',
                                name='Motion',
                                line=dict(color='green')
                            )],
                        'layout': go.Layout(
                            title='GSC Signal',
                            clickmode='event',
                            shapes=[{
                                'type': 'rect',
                                'x0': events_df.loc[i, 'Start_Time'],
                                'x1': events_df.loc[i, 'End_Time'],
                                'y0': min(eda_phasic),
                                'y1': max(gsc),
                                'fillcolor': color_list[i],
                                'opacity': 0.3,
                                'line': {'width': 0},
                                'layer': 'below'
                            } for i in range(len(events_df))],
                            annotations = [
                                {
                                    'x': ((events_df.loc[i, 'Start_Time'] + events_df.loc[i, 'End_Time']) / 2),
                                    'y': max(gsc),
                                    'text': events_df.loc[i, 'Classification'],
                                    'showarrow': False,
                                    'textangle': 45,
                                    'yshift': (i % 2) * 30  # Stagger the labels
                                } for i in range(len(events_df))],
                            
                            xaxis=dict(
                                tickangle=45  # Rotate x-axis labels
                            )
                            
                        )
                    }),
                    html.Button('Hide Whole Signal',
                                id={
                        'type': 'hide-whole-signal',
                        'index': 1
                    }),
                    html.Button('Reset ECG Peaks',
                                id={
                        'type': 'reset-ecg-peaks',
                        'index': 1
                    }),
                    html.Button('Reset GSC Peaks',
                                id={
                        'type': 'reset-gsc-peaks',
                        'index': 1
                    })]
        
        )
        
    else:
        graph_layout = html.Div([
            dcc.Graph(
                style={'width': '90%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
                id='ECG_plot',
                figure={
                    'data': [
                        go.Scatter(
                            x=time,
                            y=signal,
                            mode='lines',
                            name='ECG',
                            line=dict(color='blue')
                        ),
                        go.Scatter(
                            x=time,
                            y=[signal[i] for i in rpeaks],
                            mode='markers',
                            name='R-peaks',
                            marker=dict(color=peaks_color)
                        ),
                        go.Scatter(
                            x=time,
                            y=motion,
                            mode='lines',
                            name='Motion',
                            line=dict(color='green')
                        )],
                    'layout': go.Layout(
                        title='ECG Signal',
                        clickmode='event'
                    )
                }),
                html.Button('Hide Whole Signal',
                            id={
                    'type': 'hide-whole-signal',
                    'index': 1
                }),
                html.Button('Reset ECG Peaks',
                            id={
                    'type': 'reset-ecg-peaks',
                    'index': 1
                })
        ])
    return graph_layout, loading_message



@callback(
    Output({'type': 'events-table', 'index': ALL}, 'data', allow_duplicate=True),
    [Input({'type': 'refresh-events', 'index': ALL}, 'n_clicks'),
        Input({'type': 'events-table', 'index': ALL}, 'derived_virtual_data'),
        Input({'type': 'ecg-plot', 'index': ALL}, 'figure')],
        State('whole-signal-store', 'data'),
        prevent_initial_call=True
)
def refresh_events_table(n_clicks, data, ecg_plot,whole_signal_data):
    if data is None:
        raise PreventUpdate
    
    if n_clicks == 0 or n_clicks[0] is None:
        raise PreventUpdate
    
    print("Start refreshing events table")
    records = [item for sublist in data for item in sublist]
    events_df = pd.DataFrame(records)
    events_columns = [{'name': i, 'id': i, 'editable': True} for i in events_df.columns]
    events_data = events_df.to_dict('records')

    whole_signal_data = whole_signal_data

    signal = whole_signal_data['signal']
    time = whole_signal_data['time']
    quality = whole_signal_data['ecg_quality']
    rpeaks = whole_signal_data['rpeaks']
    rpeaks_time = [time[i] for i in rpeaks]
    # print(f"Figure data: {json.dumps(ecg_plot[0], indent=2)}")
    peaks_color = ecg_plot[0]['data'][1]['marker']['color']
    print(f"Peaks color: {peaks_color}")

    updated_events_df = []

    for idx, row in events_df.iterrows():
        start_time = row['Start_Time']
        start_time = pd.to_numeric(start_time, errors='coerce')
        end_time = row['End_Time']
        end_time = pd.to_numeric(end_time, errors='coerce')
        # Find the point with the minimum absolute distance to the start time
        time = np.array(time)

        differences_start_time = np.abs(time - start_time)

        min_diff_start_time = np.argmin(differences_start_time)

        closest_start_time = time[min_diff_start_time]

        # find the index of the closest start time
        start_idx = np.where(time == closest_start_time)

        # Find the point with the minimum absolute distance to the end time
        differences_end_time = np.abs(time - end_time)
        min_diff_end_time = np.argmin(differences_end_time)
        closest_end_time = time[min_diff_end_time]
        # find the index of the closest end time
        end_idx = np.where(time == closest_end_time)

        print(f"Start index: {start_idx[0][0]}")
        print(f"End index: {end_idx[0][0]}")

        quality_values = quality[start_idx[0][0]:end_idx[0][0]]

        rpeaks_time = np.array(rpeaks_time)

        differences_start_time_peak = np.abs(rpeaks_time - start_time)
        differences_end_time_peak = np.abs(rpeaks_time - end_time)
        
        closest_start_time_peak = rpeaks_time[np.argmin(differences_start_time_peak)]
        closest_end_time_peak = rpeaks_time[np.argmin(differences_end_time_peak)]
        print(f"Closest start time peak: {closest_start_time_peak}")
        print(f"Closest end time peak: {closest_end_time_peak}")
        start_idx_peak = np.where(rpeaks_time == closest_start_time_peak)
        end_idx_peak = np.where(rpeaks_time == closest_end_time_peak)

        # if start_idx_peak[0].size == 0 or end_idx_peak[0].size == 0:
        #     print(f"Skipping event at index {idx} due to invalid peak start or end index")
        #     print(f"Start peak index first: {start_idx_peak[0]}")
        #     print(f"End peak index first: {end_idx_peak[0]}")
        #     continue

        print(f"Start peak index: {start_idx_peak[0][0]}")
        print(f"End peak index: {end_idx_peak[0][0]}")

        # print(f"Quality values: {quality_values[0]} {quality_values[-1]}")
        print(f"Peaks color length: {len(peaks_color)}")
        print(f"Peaks color: {peaks_color[start_idx_peak[0][0]:end_idx_peak[0][0]]}")
        artifacts = [1 if peak_color == 'yellow' else 0 for peak_color in peaks_color[start_idx_peak[0][0]:end_idx_peak[0][0]]]
        print(f"Artifacts: {artifacts}")
        artifacts_count = sum(artifacts)
        if artifacts_count == 0:
            artifacts_percentage = 0
        else:
            artifacts_percentage = artifacts_count / len(artifacts)

        quality_values = round(np.mean(quality_values), 2)
        print(f"Quality values mean: {quality_values}")
        artifacts_percentage = round(artifacts_percentage, 2)
        events_df.loc[idx, '% Artifacts'] = f'{artifacts_percentage}%'
        events_df.loc[idx, 'Quality'] = quality_values

    print(f"Events data before: {events_data}")

    events_data = events_df.to_dict('records')

    print(f"Events data after: {events_data}")
    
    # patched_dataTable = Patch()

    # patched_dataTable['data'] = [events_data]

    return [events_data]




    


@callback(
    Output('whole-signal-container', 'children', allow_duplicate=True),
    [Input({'type': 'hide-whole-signal', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def hide_whole_signal(n_clicks):
    if n_clicks == 0 or n_clicks[0] is None:
        raise PreventUpdate
    
    print("Start hiding whole signal")
    print(f"Hide whole signal: {n_clicks}")
    return []

# @callback(
#     Output({'type': 'ecg-plot', 'index': ALL}, 'figure', allow_duplicate=True),
#     [Input({'type': 'events-table', 'index': ALL}, 'derived_virtual_data')],
#     prevent_initial_call=True
# )
# def update_whole_signal(data):
#     if not data:
#         raise PreventUpdate
    
#     records = [item for sublist in data for item in sublist]
#     events_df = pd.DataFrame(records)
#     # events_columns = [{'name': i, 'id': i, 'editable': True} for i in events_df.columns]
#     # events_data = events_df.to_dict('records')

#     patched_figure = Patch()
#     patched_figure = patched_figure['layout']
#     print(f"Events data PATCH: {patched_figure} ____________")
#     return patched_figure

@callback(
    [Output({'type': 'ecg-plot', 'index': ALL}, 'figure', allow_duplicate=True),
    Output({'type': 'gsc-plot', 'index': ALL}, 'figure', allow_duplicate=True)],
    [Input({'type': 'events-table', 'index': ALL}, 'derived_virtual_data')],
    [State('whole-signal-store', 'data'),
        State('whole-signal-container', 'children')],
    prevent_initial_call=True
)
def update_events_whole_signal(data, whole_signal_data, whole_signal_container):
    if not data and whole_signal_container == []:
        raise PreventUpdate
    
    print("Start updating whole signal")
    records = [item for sublist in data for item in sublist]
    events_df = pd.DataFrame(records)

    color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'salmon', 'gold', 'lightblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightcoral', 'darkcoral', 'lightgold', 'darkgold', 'lightblue', 'darkblue', 'lightcyan', 'darkcyan', 'lightmagenta', 'darkmagenta', 'lightlime', 'darklime', 'lightteal', 'darkteal', 'lightlavender', 'darklavender', 'lightsalmon', 'darksalmon', 'lightgray', 'darkgray', 'lightpink', 'darkpink', 'lightbrown', 'darkbrown', 'lightblack', 'darkblack']
    


    signal = whole_signal_data['signal']
    gsc = whole_signal_data['gsc']
    eda_phasic = whole_signal_data['eda_phasic']
    

    # Change to numeric
    events_df['Start_Time'] = pd.to_numeric(events_df['Start_Time'], errors='coerce')
    events_df['End_Time'] = pd.to_numeric(events_df['End_Time'], errors='coerce')

    print(f"Events data: {events_df}____")
    # Create a Patch object to store the layout changes
    patched_ecg_figure = Patch()
    patched_gsc_figure = Patch()


    patched_ecg_figure['layout']['shapes'] = []
    patched_ecg_figure['layout']['annotations'] = []
    patched_gsc_figure['layout']['shapes'] = []
    patched_gsc_figure['layout']['annotations'] = []
    

    for i in range(len(events_df)):
        updated_shape = {
            'type': 'rect',
            'x0': events_df.loc[i, 'Start_Time'],
            'x1': events_df.loc[i, 'End_Time'],
            'y0': min(signal),  # Adjust based on your requirements
            'y1': max(signal),  # Adjust based on your requirements
            'fillcolor': color_list(i),
            'opacity': 0.3,  # Adjust based on your requirements
            'line': {'width': 0},
            'layer': 'below'
        }
        updated_annotation = {
            'x': (events_df.loc[i, 'Start_Time'] + events_df.loc[i, 'End_Time']) / 2,
            'y': max(signal),  # Adjust based on your requirements
            'text': events_df.loc[i, 'Classification'],
            'showarrow': False,
            'textangle': 45,  # Rotate the label text
            'yshift': (i % 2) * 30  # Stagger the labels
        }
        print(f"Rectangle new points: {events_df.loc[i, 'Start_Time']} and {events_df.loc[i, 'End_Time']}")
        patched_ecg_figure['layout']['shapes'].append(updated_shape)
        patched_ecg_figure['layout']['annotations'].append(updated_annotation)
        # patched_gsc_figure['layout']['shapes'].append(updated_shape)
        # patched_gsc_figure['layout']['annotations'].append(updated_annotation)

    for i in range(len(events_df)):
        updated_shape_gsc = {
            'type': 'rect',
            'x0': events_df.loc[i, 'Start_Time'],
            'x1': events_df.loc[i, 'End_Time'],
            'y0': min(eda_phasic),  # Adjust based on your requirements
            'y1': max(gsc),  # Adjust based on your requirements
            'fillcolor': color_list(i),
            'opacity': 0.3,  # Adjust based on your requirements
            'line': {'width': 0},
            'layer': 'below'
        }
        updated_annotation_gsc = {
            'x': (events_df.loc[i, 'Start_Time'] + events_df.loc[i, 'End_Time']) / 2,
            'y': max(gsc),  # Adjust based on your requirements
            'text': events_df.loc[i, 'Classification'],
            'showarrow': False,
            'textangle': 45,  # Rotate the label text
            'yshift': (i % 2) * 30  # Stagger the labels
        }
        print(f"Rectangle new points: {events_df.loc[i, 'Start_Time']} and {events_df.loc[i, 'End_Time']}")

        patched_gsc_figure['layout']['shapes'].append(updated_shape_gsc)
        patched_gsc_figure['layout']['annotations'].append(updated_annotation_gsc)


    return [patched_ecg_figure], [patched_gsc_figure]


@callback(
    [Output({'type': 'ecg-plot', 'index': ALL}, 'figure', allow_duplicate=True),
        Output('whole-signal-store', 'data', allow_duplicate=True)],
    [Input({'type': 'ecg-plot', 'index': ALL}, 'clickData')],
    [State('whole-signal-store', 'data'),
        State('whole-signal-container', 'children')],
    prevent_initial_call=True
)
def display_ecg_click_data(click_data, whole_signal_data, whole_signal_container):
    if not click_data or whole_signal_container == []:
        raise PreventUpdate
    
    if click_data == []:
        raise PreventUpdate

    print("Start displaying click data")
    print(f"Click data: {click_data}")
    
    signal = whole_signal_data['signal']
    time = whole_signal_data['time']
    original_rpeaks = whole_signal_data['rpeaks']
    rpeaks = whole_signal_data['updated_rpeaks']
    peaks_color = whole_signal_data['updated_rpeaks_color']

    rpeaks_time = [time[i] for i in original_rpeaks]
    
    patched_figures = []

    if click_data[0] is None:
        raise PreventUpdate

    # If the user clicked on a rpeak in the ECG plot
    for data in click_data:
        if 'points' in data.keys():
            print(f"Data: {data}")
            point = data['points'][0]
            print(f"Point: {point}")
            if point['curveNumber'] == 1:
                clicked_rpeak = point['x']
                
                clicked_rpeak_index = rpeaks_time.index(clicked_rpeak)
                current_color = point['marker.color']
                print(f"Clicked Rpeak: {clicked_rpeak}")
                print(f"Clicked Rpeak index: {clicked_rpeak_index}")
                print(f"Current color: {current_color}")

                # Toggle color of the clicked point based on its current color
                updated_peaks_color = peaks_color[:]
                if current_color == 'red':
                    updated_peaks_color[clicked_rpeak_index] = 'yellow'
                else:
                    updated_peaks_color[clicked_rpeak_index] = 'red'

                # print(f"Updated Peaks color: {updated_peaks_color}")

                patched_figure = Patch()
                patched_figure['data'][1]['marker']['color'] = updated_peaks_color
                patched_figures.append(patched_figure)

                print(f"Original peaks color: {whole_signal_data['updated_rpeaks_color'][clicked_rpeak_index-5:clicked_rpeak_index+5]}")
                print(f"Original rpeaks: {whole_signal_data['updated_rpeaks'][clicked_rpeak_index-5:clicked_rpeak_index+5]}")
                

                updated_peaks_indecs = []

                print(f"Lenght of updated peaks color: {len(updated_peaks_color)}")
                print(f"Lenght of original rpeaks: {len(original_rpeaks)}")
                print(f"Lenght of rpeaks: {len(rpeaks)}")

                # whole_signal_data['updated_rpeaks'] = []

                for i in range(len(updated_peaks_color)):
                    if updated_peaks_color[i] == 'red':
                        # print(f"Rpeak index: {rpeaks[i]}")
                        updated_peaks_indecs.append(original_rpeaks[i])
                    elif updated_peaks_color[i] == 'yellow':
                        if original_rpeaks[i] in updated_peaks_indecs:
                            updated_peaks_indecs.remove(original_rpeaks[i])
                        else:
                            continue
                    

                print(f"Updated length of peaks color: {len(updated_peaks_color)}")
                print(f"Updated length of rpeaks: {len(rpeaks)}")
                print(f"Updated length of updated rpeaks: {len(updated_peaks_indecs)}")

                diff_between_lists = len(rpeaks) - len(updated_peaks_indecs)

                clicked_rpeak_index_updated = clicked_rpeak_index - diff_between_lists

                whole_signal_data['updated_rpeaks_color'] = updated_peaks_color
                whole_signal_data['updated_rpeaks'] = updated_peaks_indecs
                    
                print(f"Updated peaks color: {whole_signal_data['updated_rpeaks_color'][clicked_rpeak_index-5:clicked_rpeak_index+5]}")
                print(f"Updated rpeaks: {whole_signal_data['updated_rpeaks'][clicked_rpeak_index_updated-5:clicked_rpeak_index_updated+5]}")
                print(f"Updated rpeaks indices: {updated_peaks_indecs[clicked_rpeak_index_updated-5:clicked_rpeak_index_updated+5]}")
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate
            

    return [patched_figures, whole_signal_data]
    

@callback(
    [Output({'type': 'gsc-plot', 'index': ALL}, 'figure', allow_duplicate=True)],
    [Input({'type': 'gsc-plot', 'index': ALL}, 'clickData')],
    [State('whole-signal-store', 'data'),
        State('whole-signal-container', 'children')],
    prevent_initial_call=True
)
def display_gsc_click_data(click_data, whole_signal_data, whole_signal_container):
    if not click_data or whole_signal_container == []:
        raise PreventUpdate
    
    print("Start displaying click data")
    print(f"Click data: {click_data}")
    
    gsc = whole_signal_data['gsc']
    time = whole_signal_data['time']
    gsc_peaks = whole_signal_data['gsc_peaks']
    gsc_peaks_time = [time[i] for i in gsc_peaks]
    gsc_peaks_color = whole_signal_data['gsc_peaks_color']
    
    patched_figures = []

    for data in click_data:
        # If the user clicked on a rpeak in the ECG plot
        if 'points' in data.keys():
            point = data['points'][0]
            print(f"Point: {point}")
            if point['curveNumber'] == 1:
                clicked_peak = point['x']
                clicked_peak_index = gsc_peaks_time.index(clicked_peak)

                # print(f"GSC figure: {json.dumps(gsc_figures[0]['data'][1], indent=2)}")
                current_color = point['marker.color']
                print(f"Clicked Peak: {clicked_peak}")
                print(f"Clicked Peak index: {clicked_peak_index}")
                print(f"Current color: {current_color}")


                # Toggle color of the clicked point based on its current color
                updated_peaks_color = gsc_peaks_color[:]
                if current_color == 'red':
                    updated_peaks_color[clicked_peak_index] = 'yellow'
                else:
                    updated_peaks_color[clicked_peak_index] = 'red'

                print(f"Updated Peaks color: {updated_peaks_color}")

                patched_figure = Patch()
                patched_figure['data'][1]['marker']['color'] = updated_peaks_color
                
                patched_figures.append(patched_figure)
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate
            
    return [patched_figures]

@callback(
    Output('signal-cards-container', 'children'),
    Output('signal-cards-store', 'data', allow_duplicate=True),
    [Input({'type': 'apply-events', 'index': ALL}, 'n_clicks'),
        Input({'type': 'events-table', 'index': ALL}, 'derived_virtual_selected_rows'),
        Input('signal-cards-store', 'data'),
    ],
    [State('whole-signal-store', 'data'),
        State('events-store', 'data')],
    prevent_initial_call=True
)
def generate_events_cards(n_clicks, selected_rows, signal_cards_store ,whole_signal_data, events_data):
    if n_clicks == 0 or n_clicks[0] is None:
        raise PreventUpdate
    
    print("Start generating events cards")
    print(f"Selected rows: {selected_rows}")
    print(f"Events data: {events_data}")
    # print(f"Whole signal data: {whole_signal_data}")

    if 'events' in events_data.keys():
        events_data = events_data['events']
    else:
        # Handle the case where 'events' is not in events_data
        # For example, you might want to set events_data to an empty list
        events_data = []

    records = events_data

    events_df = pd.DataFrame(records)

    if 'Start_Time' not in events_df.columns or 'End_Time' not in events_df.columns or 'Classification' not in events_df.columns:
        print("Column names do not match expected values.")
        raise PreventUpdate

    if selected_rows[0] is None:
        raise PreventUpdate

    selected_events = events_df.iloc[selected_rows[0]]
    print(f"Selected events: {selected_events}")

    cards = []

    whole_signal = whole_signal_data

    original_rpeaks = whole_signal['rpeaks']
    original_rpeaks_time = [whole_signal['time'][i] for i in original_rpeaks]
    rpeaks = whole_signal['updated_rpeaks']
    rpeaks_time = [whole_signal['time'][i] for i in rpeaks]
    peaks_color = whole_signal['updated_rpeaks_color']
    time = whole_signal['time']
    signal = whole_signal['signal']
    ecg_quality = whole_signal['ecg_quality']
    ectopic_beats = whole_signal['ectopic_beats']
    longshort_beats = whole_signal['longshort_beats']
    false_negatives = whole_signal['false_negatives']
    false_positives = whole_signal['false_positives']
    motion = whole_signal['motion']

    if 'gsc' in whole_signal.keys():
        gsc = whole_signal['gsc']
        gsc_peaks = whole_signal['gsc_peaks']
        gsc_peaks_time = [whole_signal['time'][i] for i in gsc_peaks]
        gsc_peaks_color = whole_signal['gsc_peaks_color']
        eda_tonic = whole_signal['eda_tonic']
        eda_phasic = whole_signal['eda_phasic']




    for idx, row in selected_events.iterrows():
        start_time = row['Start_Time']
        end_time = row['End_Time']
        classification = row['Classification']

        start_time = pd.to_numeric(start_time, errors='coerce')
        end_time = pd.to_numeric(end_time, errors='coerce')

        time = np.array(time)

        differences_start_time = np.abs(time - start_time)
        min_diff_start_time = np.argmin(differences_start_time)
        closest_start_time = time[min_diff_start_time]

        start_idx = np.where(time == closest_start_time)

        differences_end_time = np.abs(time - end_time)
        min_diff_end_time = np.argmin(differences_end_time)
        closest_end_time = time[min_diff_end_time]
        end_idx = np.where(time == closest_end_time)

        sliced_signal = signal[start_idx[0][0]:end_idx[0][0]]
        sliced_time = time[start_idx[0][0]:end_idx[0][0]]
        sliced_ecg_quality = ecg_quality[start_idx[0][0]:end_idx[0][0]]
        sliced_motion = motion[start_idx[0][0]:end_idx[0][0]]

        rpeaks_time = np.array(rpeaks_time)

        differences_start_time_peak = np.abs(rpeaks_time - start_time)
        differences_end_time_peak = np.abs(rpeaks_time - end_time)
        closest_start_time_peak = rpeaks_time[np.argmin(differences_start_time_peak)]
        closest_end_time_peak = rpeaks_time[np.argmin(differences_end_time_peak)]
        start_idx_peak = np.where(rpeaks_time == closest_start_time_peak)
        end_idx_peak = np.where(rpeaks_time == closest_end_time_peak)

        sliced_rpeaks = rpeaks[start_idx_peak[0][0]:end_idx_peak[0][0]]
        sliced_rpeaks_time = rpeaks_time[start_idx_peak[0][0]:end_idx_peak[0][0]]
        sliced_peaks_color = peaks_color[start_idx_peak[0][0]:end_idx_peak[0][0]]
        sliced_ectopic_beats = ectopic_beats[start_idx_peak[0][0]:end_idx_peak[0][0]]
        sliced_longshort_beats = longshort_beats[start_idx_peak[0][0]:end_idx_peak[0][0]]
        sliced_false_negatives = false_negatives[start_idx_peak[0][0]:end_idx_peak[0][0]]
        sliced_false_positives = false_positives[start_idx_peak[0][0]:end_idx_peak[0][0]]



        if 'gsc' in whole_signal.keys():
            
            gsc_peaks_time = np.array(gsc_peaks_time)

            differences_start_time_peak_gsc = np.abs(gsc_peaks_time - start_time)
            differences_end_time_peak_gsc = np.abs(gsc_peaks_time - end_time)
            closest_start_time_peak_gsc = gsc_peaks_time[np.argmin(differences_start_time_peak_gsc)]
            closest_end_time_peak_gsc = gsc_peaks_time[np.argmin(differences_end_time_peak_gsc)]
            
            

            start_idx_peak_gsc = np.where(gsc_peaks_time == closest_start_time_peak_gsc)
            end_idx_peak_gsc = np.where(gsc_peaks_time == closest_end_time_peak_gsc)

            

            sliced_gsc = gsc[start_idx[0][0]:end_idx[0][0]]
            sliced_gsc_peaks = gsc_peaks[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]]
            sliced_gsc_peaks_time = gsc_peaks_time[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]]
            sliced_gsc_peaks_color = gsc_peaks_color[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]]
            sliced_eda_tonic = eda_tonic[start_idx[0][0]:end_idx[0][0]]
            sliced_eda_phasic = eda_phasic[start_idx[0][0]:end_idx[0][0]]

            print(f"Classification: {classification}")


            signal_cards_store = {
                'events': {f'{classification}':
                                {'start_time': start_time,
                                'end_time': end_time,
                                'classification': classification,
                                'signal': sliced_signal,
                                'time': sliced_time,
                                    'quality': sliced_ecg_quality,
                                    'rpeaks': rpeaks[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'rpeaks_time': rpeaks_time[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'peaks_color': peaks_color[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'ectopic_beats': ectopic_beats[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'longshort_beats': longshort_beats[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'false_negatives': false_negatives[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'false_positives': false_positives[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'gsc': gsc[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]],
                                    'gsc_peaks': gsc_peaks[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]],
                                    'gsc_peaks_time': gsc_peaks_time[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]],
                                    'gsc_peaks_color': gsc_peaks_color[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]],
                                    'eda_tonic': eda_tonic[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]],
                                    'eda_phasic': eda_phasic[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]],
                                }
                
                }
            }
            


            card = dbc.Card(
                [
                    dbc.CardHeader(f"Event {idx + 1}"),
                    dbc.CardBody(
                        [
                            html.H5(f"Classification: {classification}", className="card-title"),
                            html.P(f"Start Time: {start_time}     End Time: {end_time}", className="card-text"),

                        ]
                    ),
                    dcc.Graph(
                        style={'width': '90%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
                        id={'type': 'event-plot', 'index': idx},
                        figure={
                            'data': [
                                go.Scatter(
                                    x=sliced_time,
                                    y=sliced_signal,
                                    mode='lines',
                                    name='ECG',
                                    line=dict(color='blue'),
                                    hovertext = [f'Time: {sliced_time[i]}<br>Signal: {sliced_signal[i]}<br>Quality: {sliced_ecg_quality[i]}' for i in range(len(sliced_time))]
                                ),
                                go.Scatter(
                                    x=sliced_rpeaks_time,
                                    y=[signal[i] for i in sliced_rpeaks],
                                    mode='markers',
                                    name='R-peaks',
                                    marker=dict(color=sliced_peaks_color),
                                    hovertext=[f'Time: {sliced_rpeaks_time[i]}<br>Signal: {signal[sliced_rpeaks[i]]}<br>Quality: {ecg_quality[sliced_rpeaks[i]]}' for i in range(len(sliced_rpeaks_time))]
                                ),
                                go.Scatter(
                                    x=sliced_rpeaks_time,
                                    y=[signal[i] for i in sliced_ectopic_beats],
                                    mode='markers',
                                    name='Ectopic Beats',
                                    marker=dict(symbol='x', color='red')
                                ),
                                go.Scatter(
                                    x=sliced_rpeaks_time,
                                    y=[signal[i] for i in sliced_longshort_beats],
                                    mode='markers',
                                    name='Longshort Beats',
                                    marker=dict(symbol='x', color='green')
                                ),
                                go.Scatter(
                                    x=sliced_rpeaks_time,
                                    y=[signal[i] for i in sliced_false_negatives],
                                    mode='markers',
                                    name='False Negatives',
                                    marker=dict(symbol='x', color='orange')
                                ),
                                go.Scatter(
                                    x=sliced_rpeaks_time,
                                    y=[signal[i] for i in sliced_false_positives],
                                    mode='markers',
                                    name='False Positives',
                                    marker=dict(symbol='x', color='purple'),
                                ),
                                go.Scatter(
                                    x=sliced_time,
                                    y=[x*0.1 for x in sliced_motion],
                                    mode='lines',
                                    name='Motion',
                                    marker=dict(symbol='x', color='orange'),
                                )],
                                    
                            'layout': go.Layout(
                                title='ECG Signal',
                                clickmode='event',
                                xaxis=dict(
                                    tickangle=45  # Rotate x-axis labels
                                )
                            )
                        }
                    ),
                        html.Div(
                            id={'type': 'ecg-table-events', 'index': idx},
                            children=[]
                    ),
                    dcc.Graph(
                        style={'width': '90%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
                        id={'type': 'gsc-plot', 'index': idx},
                        figure={
                            'data': [
                                go.Scatter(
                                    x=sliced_time,
                                    y=sliced_gsc,
                                    mode='lines',
                                    name='GSC',
                                    line=dict(color='blue'),
                                    hovertext = [f'Time: {sliced_time[i]}<br>GSC: {gsc[i]}<br>EDA Tonic: {eda_tonic[i]}<br>EDA Phasic: {eda_phasic[i]}' for i in range(len(sliced_time))]
                                ),
                                go.Scatter(
                                    x=sliced_gsc_peaks_time,
                                    y=[gsc[i] for i in sliced_gsc_peaks],
                                    mode='markers',
                                    name='GSC Peaks',
                                    marker=dict(color=gsc_peaks_color[start_idx_peak_gsc[0][0]:end_idx_peak_gsc[0][0]]),
                                    hovertext=[f'Time: {sliced_gsc_peaks_time[i]}<br>GSC: {gsc[sliced_gsc_peaks[i]]}<br>EDA Tonic: {eda_tonic[sliced_gsc_peaks[i]]}<br>EDA Phasic: {eda_phasic[sliced_gsc_peaks[i]]}' for i in range(len(sliced_gsc_peaks_time))]
                                ),
                                go.Scatter(
                                    x=sliced_time,
                                    y=sliced_eda_tonic,
                                    mode='lines',
                                    name='EDA Tonic',
                                    line=dict(color='orange')
                                ),
                                go.Scatter(
                                    x=sliced_time,
                                    y=sliced_eda_phasic,
                                    mode='lines',
                                    name='EDA Phasic',
                                    line=dict(color='purple')
                                ),
                                go.Scatter(
                                    x=sliced_time,
                                    y=[x*100 for x in sliced_motion],
                                    mode='lines',
                                    name='Motion',
                                    line=dict(color='green'),
                                    hovertext=[f'Time: {sliced_time[i]}<br>Motion: {sliced_motion[i]}' for i in range(len(sliced_time))]
                                )],
                            'layout': go.Layout(
                                title='GSC Signal',
                                clickmode='event',
                                xaxis=dict(
                                    tickangle=45  # Rotate x-axis labels
                                )
                            )
                        }
                    )
                    ,html.Div(
                            id={'type': 'gsc-table-events', 'index': idx},
                            children=[]
                    )
                ]
            )
            cards.append(card)



        else:

            signal_cards_store = {
                'events': {{f'{classification}'}:
                                {'start_time': start_time,
                                'end_time': end_time,
                                'classification': classification,
                                'signal': sliced_signal,
                                'time': sliced_time,
                                    'quality': sliced_ecg_quality,
                                    'rpeaks': rpeaks[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'rpeaks_time': rpeaks_time[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'peaks_color': peaks_color[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'ectopic_beats': ectopic_beats[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'longshort_beats': longshort_beats[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'false_negatives': false_negatives[start_idx_peak[0][0]:end_idx_peak[0][0]],
                                    'false_positives': false_positives[start_idx_peak[0][0]:end_idx_peak[0][0]]
                                }
                }
                }
                    
            



    return cards, signal_cards_store

            
@callback(
    Output({'type': 'ecg-table-events', 'index': MATCH}, 'children'),
    Input('signal-cards-store', 'data'),
    Input({'type': 'events-table', 'index': ALL}, 'derived_virtual_selected_rows'),
    Input({'type': 'apply-events', 'index': ALL}, 'n_clicks'),
    State('events-store', 'data'),
    prevent_initial_call=True
)
def generate_ecg_table_events(signal_cards_store, selected_rows, n_clicks, events_data):

    if selected_rows[0] is None:
        raise PreventUpdate
    if n_clicks == 0 or n_clicks[0] is None:
        raise PreventUpdate
    
    if 'events' in events_data.keys():
        events_data = events_data['events']

    else:
        events_data = []

    records = events_data

    events_df = pd.DataFrame(records)

    
    selected_events = events_df.iloc[selected_rows[0]]
    print(f"Selected events: {selected_events}")

    for idx, row in selected_events.iterrows():

        classification = row['Classification']

        event_dict = signal_cards_store['events'][classification]

        signal = event_dict['signal']
        time = event_dict['time']
        rpeaks = event_dict['rpeaks']

        hrv_time = nk.hrv_time(rpeaks, sampling_rate=500)
        hrv_frequency = nk.hrv_frequency(rpeaks, sampling_rate=500)
        hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=500)
        
        hrv_time_df = pd.DataFrame(hrv_time, index=[0])
        hrv_frequency_df = pd.DataFrame(hrv_frequency, index=[0])
        hrv_nonlinear_df = pd.DataFrame(hrv_nonlinear, index=[0])

        hrv_df = pd.concat([pd.DataFrame(hrv_time, index=[0]), pd.DataFrame(hrv_frequency, index=[0]), pd.DataFrame(hrv_nonlinear, index=[0])], keys=['Time Domain', 'Frequency Domain', 'Nonlinear Domain'])
        table = dbc.Table.from_dataframe(hrv_df, striped=True, bordered=True, hover=True)
        
        
        return table
    

    
    



        


@callback(
    [Output('whole-signal-store', 'data', allow_duplicate=True),
        Output({'type': 'ecg-plot', 'index': ALL}, 'figure'),
        Output({'type': 'reset-ecg-peaks', 'index': ALL}, 'n_clicks')],
    [Input({'type': 'reset-ecg-peaks', 'index': ALL}, 'n_clicks'),
        Input('whole-signal-store', 'data')],
        prevent_initial_call=True
)
def reset_ecg_peaks(n_clicks, whole_signal_data):
    if whole_signal_data is None:
        raise PreventUpdate
    
    print (f"n_clicks: {json.dumps(n_clicks, indent=2)}")
    if n_clicks == []:
        raise PreventUpdate
    
    if n_clicks[0] == None:
        raise PreventUpdate
    
    


    else:
        if 'peaks_color' in whole_signal_data.keys():
            print("Start resetting ECG peaks")
            updated_rpeaks_color = whole_signal_data['peaks_color']
            whole_signal_data['updated_rpeaks_color'] = updated_rpeaks_color

            updated_rpeaks = whole_signal_data['rpeaks']
            whole_signal_data['updated_rpeaks'] = updated_rpeaks
            
            patched_figures = []

            patched_data = Patch()
            patched_data['data'][1]['marker']['color'] = updated_rpeaks_color
            patched_figures.append(patched_data)

            # Reset the click data
            n_clicks = [None]


            return [whole_signal_data, patched_figures, n_clicks]






# if __name__ == '__main__':
    


#     webview.start()