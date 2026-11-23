from flask import session, signals
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objs as go
import pandas as pd
import numpy as np
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

from pages.paths import DATA_PATH, TEMPLATES_PATH, project_names
from pages.data_manager import atomic_write, get_data_manager
import logging

logger = logging.getLogger(__name__)

dash.register_page(__name__, name="Signal Processing Page", path="/", order = 6)

pages = {name: name for name in project_names()}


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
    
    
    fibro_hdf_file = DATA_PATH

    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware_events.parquet')

    print(f'fibro_hdf_file: {fibro_hdf_file}')

    if not os.path.exists(fibro_hdf_file):
        return html.Div("Error loading data. Please load subjects to the app and try again.", style={"color": "red", "fontSize": 20})


    print(f'Loading fibro subjects table...')

    try:
        subjects = pl.read_parquet(fibro_hdf_file)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return html.Div("Error loading data. Please check the file and try again.", style={"color": "red", "fontSize": 20})

    print(f'Loaded fibro subjects file')


    



    subjects = (
        subjects
        .drop_nulls(subset=['subject_id'])
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
            'field': 'subject_id',
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
            'field': 'artifacts_percentage',
            'type': 'rightAligned',
        },
        {
            'headerName': 'corrected_rpeaks_%',
            'field': 'corrected_rpeaks_percentage',
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
            'field': 'duration_minutes',
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
    if not selectedRows or not selectedRows[0] or not gsc_or_ecg:
        return []
    row = selectedRows[0][0]
    subject = row["subject_id"]
    event = row["event_name"]
    mode = gsc_or_ecg[0]
    manager = get_data_manager(project)
    message_style = {"color": "red", "fontSize": 20}

    try:
        bounds = (
            pl.scan_parquet(manager.events_parquet_path)
            .filter((pl.col("subject_id") == subject) & (pl.col("event_name") == event))
            .select("start_time", "end_time").head(1).collect()
        )
        if bounds.is_empty():
            return html.Div(f"No event data for {subject}, {event}.", style=message_style)
        start, end = bounds.row(0)

        subject_signals = manager.scan_subject_signal(subject)
        schema = subject_signals.collect_schema()
        channel = "signal" if mode == "ecg" else "gsc"
        if channel not in schema:
            return html.Div(f"{mode.upper()} is unavailable for {subject}.", style=message_style)
        columns = ["time", "motion", channel]
        columns.extend(name for name in ("eda_tonic", "eda_phasic") if mode == "gsc" and name in schema)
        event_signal = (
            subject_signals.filter(pl.col("time").is_between(start, end))
            .select(columns).sort("time").collect().to_pandas()
        )
        if event_signal.empty:
            return html.Div(f"No signal samples for {subject}, {event}.", style=message_style)

        attributes = manager.load_subject_attributes(subject, include_quality=False) or {}
        sample_columns = ["time", channel]
        if mode == "ecg" and "quality" in schema:
            sample_columns.append("quality")
        samples = subject_signals.select(sample_columns).collect()
        original_time = samples["time"].to_numpy()
        original_signal = samples[channel].to_numpy()
        for col in (channel, "motion"):
            values = event_signal[col]
            deviation = values.std()
            if pd.notna(deviation) and deviation > 0:
                event_signal[col] = (values - values.mean()) / deviation

        time = event_signal["time"].to_list()
        motion = event_signal["motion"].to_list()
        if mode == "gsc":
            return plot_gsc_signals(event_signal, time, motion, original_signal,
                                    attributes.get("gsc_attributes", {}), original_time, start, end)

        _apply_yellow_peak_corrections(project, subject, original_time, attributes)
        if "quality" in samples.columns:
            quality = samples["quality"].to_numpy()
        else:
            quality = attributes.get("quality")
        if (quality is None or len(quality) != len(original_time)) and attributes.get("rpeaks"):
            with open(TEMPLATES_PATH / f"{project}_template.json", encoding="utf-8") as file:
                rate = json.load(file)["parameters"]["sampling_rate"]
            quality = nk.ecg_quality(original_signal, sampling_rate=rate)
        return plot_ecg_signals(event_signal, time, motion, original_signal, attributes,
                                original_time, start, end, quality)
    except Exception:
        logger.exception("Unable to plot project %s, subject %s, event %s", project, subject, event)
        return html.Div(f"Could not plot {subject}, {event}. See the app log for details.", style=message_style)


def _apply_yellow_peak_corrections(project, subject, original_time, attributes):
    path = DATA_PATH / f"{project}_UpdatedData.json"
    if not path.exists() or len(original_time) == 0:
        return
    with open(path, encoding="utf-8") as file:
        updates = json.load(file)
    corrected = set(attributes.get("yellow_peaks", []))
    sorted_time = np.all(original_time[1:] >= original_time[:-1])
    for update in updates:
        subject_updates = update.get(project, {}).get(subject, {})
        for event in subject_updates.values():
            for timestamp, color in event.items():
                if timestamp == "artifact_percentage" or color != "yellow":
                    continue
                target = float(timestamp)
                if sorted_time:
                    right = min(int(np.searchsorted(original_time, target)), len(original_time) - 1)
                    left = max(0, right - 1)
                    index = min((left, right), key=lambda i: abs(original_time[i] - target))
                else:
                    index = int(np.argmin(np.abs(original_time - target)))
                corrected.add(index)
    attributes["yellow_peaks"] = sorted(corrected)


def plot_gsc_signals(signals, time, motion, origin_gsc, gsc_attributes, original_time, event_start_time, event_end_time):
    """Plot GSC signals with SCR features"""
    gsc_signal = signals['gsc'].to_list()
    eda_tonic = signals['eda_tonic'].to_list() if 'eda_tonic' in signals else [0] * len(time)
    eda_phasic = signals['eda_phasic'].to_list() if 'eda_phasic' in signals else [0] * len(time)

    # Handle missing GSC attributes gracefully
    if not gsc_attributes or gsc_attributes == {}:
        scr_onsets_time = []
        scr_peaks_time = []
        scr_recovery_time = []
        scr_onsets_index = []
        scr_peak_index = []
        scr_recovery_index = []
    else:
        # Process SCR features
        scr_onset = gsc_attributes.get('SCR_Onsets', [])
        scr_onset = [original_time[i] for i in scr_onset if i < len(original_time)]
        scr_onsets_time = [i for i in scr_onset if i >= event_start_time and i <= event_end_time]
        scr_onsets_index = [i for i in range(len(original_time)) if original_time[i] in scr_onsets_time]
        
        scr_peak = gsc_attributes.get('SCR_Peaks', [])
        scr_peak = [original_time[i] for i in scr_peak if i < len(original_time)]
        scr_peaks_time = [i for i in scr_peak if i >= event_start_time and i <= event_end_time]
        scr_peak_index = [i for i in range(len(original_time)) if original_time[i] in scr_peaks_time]
        
        scr_recovery = gsc_attributes.get('SCR_Recovery', [])
        scr_recovery = [int(x) for x in scr_recovery if not np.isnan(float(x)) and int(x) < len(original_time)]
        scr_recovery = [original_time[i] for i in scr_recovery]
        scr_recovery_time = [i for i in scr_recovery if i >= event_start_time and i <= event_end_time]
        scr_recovery_index = [i for i in range(len(original_time)) if original_time[i] in scr_recovery_time]

    graph = dcc.Graph(
        id={'type': 'gsc-plot', 'index': 1},
        figure={
            'data': [
                go.Scatter(
                    x=time, y=[x*0.1 for x in motion], mode='lines', name='Motion',
                    line=dict(color='black'),
                    hovertext=[f'Time: {time[i]}<br>Motion: {motion[i]}' for i in range(len(time))]
                ),
                go.Scatter(
                    x=time, y=gsc_signal, mode='lines', name='GSC',
                    line=dict(color='blue'),
                    hovertext=[f'Time: {time[i]}<br>GSC: {gsc_signal[i]}' for i in range(len(time))]
                ),
                go.Scatter(
                    x=time, y=eda_tonic, mode='lines', name='EDA Tonic',
                    line=dict(color='green'),
                    hovertext=[f'Time: {time[i]}<br>EDA Tonic: {eda_tonic[i]}' for i in range(len(time))]
                ),
                go.Scatter(
                    x=time, y=eda_phasic, mode='lines', name='EDA Phasic',
                    line=dict(color='red'),
                    hovertext=[f'Time: {time[i]}<br>EDA Phasic: {eda_phasic[i]}' for i in range(len(time))]
                ),
                go.Scatter(
                    x=scr_onsets_time, y=[origin_gsc[i] if i < len(origin_gsc) else 0 for i in scr_onsets_index],
                    mode='markers', name='SCR Onsets', marker=dict(symbol='x', color='green'),
                    hovertext=[f'Time: {scr_onsets_time[i]}<br>SCR Onset' for i in range(len(scr_onsets_time))]
                ),
                go.Scatter(
                    x=scr_peaks_time, y=[origin_gsc[i] if i < len(origin_gsc) else 0 for i in scr_peak_index],
                    mode='markers', name='SCR Peaks', marker=dict(symbol='x', color='red'),
                    hovertext=[f'Time: {scr_peaks_time[i]}<br>SCR Peak' for i in range(len(scr_peaks_time))]
                ),
                go.Scatter(
                    x=scr_recovery_time, y=[origin_gsc[i] if i < len(origin_gsc) else 0 for i in scr_recovery_index],
                    mode='markers', name='SCR Recovery', marker=dict(symbol='x', color='yellow'),
                    hovertext=[f'Time: {scr_recovery_time[i]}<br>SCR Recovery' for i in range(len(scr_recovery_time))]
                ),
            ],
            'layout': {'title': 'GSC Signal', 'xaxis': {'title': 'Time'}, 'yaxis': {'title': 'GSC'}, 'hovermode': 'closest'}
        }
    )

    return create_graph_layout(graph)

def plot_ecg_signals(signals, time, motion, origin_ecg, signal_attributes, original_time,
                     event_start_time, event_end_time, ecg_quality):
    """Plot ECG and use stored quality values for R-peak hover text."""
    ecg_signal = signals["signal"].to_list()
    rpeaks = signal_attributes.get("rpeaks", [])
    if not rpeaks:
        return html.Div("No R-peaks data available for ECG visualization.", style={"color": "red", "fontSize": 20})

    def event_indices(indices):
        return [i for i in indices if 0 <= i < len(original_time)
                and event_start_time <= original_time[i] <= event_end_time]

    rpeaks_index = event_indices(rpeaks)
    rpeaks_time = [original_time[i] for i in rpeaks_index]
    rpeaks_quality = [ecg_quality[i] for i in rpeaks_index] if ecg_quality is not None else []
    event_peak_indices = np.searchsorted(time, rpeaks_time)
    rpeaks_y = np.asarray(ecg_signal)[event_peak_indices].tolist()
    yellow_peaks = set(signal_attributes.get("yellow_peaks", []))
    ectopic_beats_index = event_indices(signal_attributes.get("ectopic_beats", []))
    ectopic_beats_time = [original_time[i] for i in ectopic_beats_index]

    graph = dcc.Graph(
        id={'type': 'ecg-plot', 'index': 1},
        figure={
            'data': [
                go.Scatter(
                    x=time, y=[x*1 for x in motion], mode='lines', name='Motion',
                    line=dict(color='black'),
                    hovertext=[f'Time: {time[i]}<br>Motion: {motion[i]}' for i in range(len(time))]
                ),
                go.Scatter(
                    x=time, y=ecg_signal, mode='lines', name='ECG',
                    line=dict(color='blue'),
                    hovertext=[f'Time: {time[i]}<br>ECG: {ecg_signal[i]}' for i in range(len(time))]
                ),
                go.Scatter(
                    # x=rpeaks_time, y=[origin_ecg[i] if i < len(origin_ecg) else 0 for i in rpeaks_index],
                    x=rpeaks_time, y=rpeaks_y,
                    mode='markers', name='R-Peaks',
                    marker=dict(color=['yellow' if i in yellow_peaks else 'red' for i in rpeaks_index]),
                    hovertext=[f'Time: {rpeaks_time[i]}<br>R-Peak<br>Quality: {rpeaks_quality[i] if i < len(rpeaks_quality) else "N/A"}' for i in range(len(rpeaks_time))]
                ),
                go.Scatter(
                    x=ectopic_beats_time, y=[origin_ecg[i] if i < len(origin_ecg) else 0 for i in ectopic_beats_index],
                    mode='markers', name='Ectopic Beats', marker=dict(symbol='x', color='black'),
                    hovertext=[f'Time: {ectopic_beats_time[i]}<br>Ectopic Beat' for i in range(len(ectopic_beats_time))]
                ),
            ],
            'layout': {'title': 'ECG Signal', 'xaxis': {'title': 'Time'}, 'yaxis': {'title': 'ECG'}, 'hovermode': 'closest'}
        }
    )

    # Add quality slider for ECG
    return html.Div([
        graph,
        create_save_controls(),
        dbc.Row([
            dbc.Col(
                dcc.Slider(0.0, 1.0, 0.01,
                          marks={i: str(round(i,2)) for i in np.arange(0.0, 1.0, 0.05)},
                          value=0.5, id={'type': 'slider-ecg', 'index': 1}),
            ),
        ]),
        dbc.Row([html.Div(id={'type': 'slider-output-ecg', 'index': 1})])
    ])

def create_graph_layout(graph):
    """Create the common graph layout with save controls"""
    return html.Div([
        graph,
        create_save_controls()
    ])

def create_save_controls():
    """Create the save controls section"""
    return dbc.Row([
        dbc.Col(
            dbc.Container([
                dbc.Button('Save Changes', id={'type': 'save-changes-button', 'index': 1},
                          n_clicks=0, style={'margin': 'auto', 'textAlign': 'center'}),
                dcc.Loading([
                    dcc.ConfirmDialog(id={'type': 'save-changes-confirm', 'index': 1}, message='')
                ], overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white"},
                custom_spinner=html.H2(["Saving changes...", dbc.Spinner(color="primary")])),
                dbc.Modal([
                    dbc.ModalHeader("Are you sure you want to save the changes?"),
                    dbc.ModalBody([
                        html.Div('Please make sure the changes are correct before saving.'),
                        dbc.Button('Save', id={'type': 'save-changes-confirm-button', 'index': 1},
                                  style={'margin': 'auto', 'textAlign': 'center'})
                    ]),
                    dbc.ModalFooter(dbc.Button('Close', id={'type': 'save-changes-close-button', 'index': 1})),
                ], id={'type': 'save-changes-modal', 'index': 1}, is_open=False),
                dcc.ConfirmDialog(id={'type': 'no-changes-confirm', 'index': 1},
                                 message='No changes were made.')
            ]),
        ),
        dbc.Col(html.Div(id={'type': 'changes-ecg', 'index': 1}, children=[])),
        dbc.Col(html.Div(id={'type': 'changes-gsc', 'index': 1}, children=[]))
    ])


# ECG Plot Slider Callback
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
                print(f'selectedRows: {selectedRows}')
                print(f'selectedRows[0]: {selectedRows[0]}')
                print(f'selectedRows[0][0]: {selectedRows[0][0]}')
                sub_id = selectedRows[0][0]['subject_id']
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
    
    confirm_message = "Changes saved successfully!"
    try:
        for project, subjects in data[0].items():
            manager = get_data_manager(project)
            updated_path = DATA_PATH / f"{project}_UpdatedData.json"
            csv_path = DATA_PATH / f"{project}_UpdatedData.csv"
            with manager.write_lock:
                events_df = pl.read_parquet(manager.events_parquet_path).to_pandas()
                if updated_path.exists():
                    with open(updated_path, encoding="utf-8") as file:
                        updates = json.load(file)
                else:
                    updates = []

                for subject, events in subjects.items():
                    time_values = set(manager.scan_subject_signal(subject, ["time"]).collect()["time"])
                    if not time_values:
                        raise ValueError(f"No signal samples for {subject}")
                    updates.append({
                        project: {subject: events},
                        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "New",
                    })
                    for event, changes in events.items():
                        timestamps = [float(t) for t in changes if t != "artifact_percentage"]
                        if any(t in time_values for t in timestamps):
                            mask = (events_df["subject_id"] == subject) & (events_df["event_name"] == event)
                            events_df.loc[mask, "artifacts_percentage"] = changes["artifact_percentage"]
                            events_df.loc[mask, "done"] = True

                atomic_write(updated_path, lambda path: manager._write_metadata(path, updates))
                atomic_write(csv_path, lambda path: pd.json_normalize(updates, sep="_").to_csv(path, index=False))
                atomic_write(manager.events_parquet_path, pl.from_pandas(events_df).write_parquet)
    except Exception as error:
        logger.exception("Unable to save signal corrections")
        return [f"Could not save changes: {error}"], [True], data, dash.no_update

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


    











