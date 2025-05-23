
from flask import session
from dash import dcc, html, Dash, CeleryManager, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
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
import subprocess
import threading
import queue
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import platform



dash.register_page(__name__, name="Project Settings", path="/settings", order=1)

projects = [(project.replace(".json", "")).split("_")[0] for project in os.listdir(r"C:\\Users\\PsyLab-6028\\Desktop\\RoeeProj\\pages\\templates") if project.endswith(".json")]
projects.append('Add Project')

layout = html.Div([
    html.H1("Project Settings"),
    html.H2("Select or Create Project"),
    dcc.Dropdown(
        id='project-dropdown',
        options=[{'label': project, 'value': project} for project in projects],
        value=projects[0],
        style={'color': 'black'}
    ),
    html.Div(id='project-settings', children=[]),
    html.Div([
        html.Label('Project Name:'),
        dcc.Input(id='project-name', type='text', value=''),
    ]),
    html.Button('Save', id='save-button', n_clicks=0),
    html.Div(id='save-message', children="")
])

def get_project_settings(project_name):
    project_settings = {}
    if project_name != 'Add Project':
        with open(f"C:\\Users\\PsyLab-6028\\Desktop\\RoeeProj\\pages\\templates\\{project_name}_template.json", "r") as f:
            project_settings = json.load(f)
    else:
        # Template for new project
        project_settings = {
            "parameters": {
                "sampling_rate": 500
            },
            "events": {},
            "blocks_shorts": {},
            "channel": [
                "Time",
                "Bio",
                "GSC"
            ],
            "output": {
                "path": ""
            },
            "input": {
                "path": ""
            },
            "HDF5": {
                "path": ""
            }
        }
    return project_settings

def save_project_settings(project_name, project_settings):
    project_name = project_name.replace(" ", "_")
    with open(f"C:\\Users\\PsyLab-6028\\Desktop\\RoeeProj\\pages\\templates\\{project_name}_template.json", "w") as f:
        json.dump(project_settings, f, indent=4)
    return True

@callback(
    Output('project-settings', 'children'),
    Output('project-name', 'value'),
    Input('project-dropdown', 'value')
)
def display_project_settings(project_name):
    settings = get_project_settings(project_name)
    return [
        html.Div([
            html.Label('Sampling Rate:'),
            dcc.Input(id='sampling-rate', type='number', value=settings['parameters']['sampling_rate'])
        ]),
        html.Div([
            html.Label('Events:'),
            html.Div(id='events-container', children=[
                create_event_component(event_key, event) for event_key, event in settings['events'].items()
            ]),
            html.Button('Add Event', id='add-event-button', n_clicks=0)
        ]),
        html.Div([
            html.Label('Blocks Shorts:'),
            html.Div(id='blocks-shorts-container', children=[
                create_block_short_component(short_key, short_value) for short_key, short_value in settings.get('blocks_shorts', {}).items()
            ]),
            html.Button('Add Block Short', id='add-block-short-button', n_clicks=0)
        ]),
        html.Div([
            html.Label('Channels:'),
            dcc.Checklist(
                id='channel-checklist',
                options=[{'label': ch, 'value': ch} for ch in ["Time", "Bio", "GSC", "X", "Y", "Z"]],
                value=settings['channel']
            )
        ]),
        html.Div([
            html.Label('Output Path:'),
            dcc.Input(id='output-path', type='text', value=settings['output']['path'])
        ]),
        html.Div([
            html.Label('Input Path:'),
            dcc.Input(id='input-path', type='text', value=settings['input']['path'])
        ]),
        html.Div([
            html.Label('HDF5 Path:'),
            dcc.Input(id='hdf5-path', type='text', value=settings['HDF5']['path'])
        ])
    ], project_name

def create_event_component(event_key, event):
    return html.Div([
        html.H4(f'Event {event_key}'),
        html.Div([
            html.Label('Name:'),
            dcc.Input(id={'type': 'event-name', 'index': event_key}, type='text', value=event['name'])
        ]),
        html.Div([
            html.Label('Start:'),
            dcc.Input(id={'type': 'event-start', 'index': event_key}, type='text', value=event['start'])
        ]),
        html.Div([
            html.Label('End:'),
            dcc.Input(id={'type': 'event-end', 'index': event_key}, type='text', value=event['end'])
        ]),
        html.Div([
            html.Label('Group:'),
            dcc.Input(id={'type': 'event-group', 'index': event_key}, type='text', value=event.get('group', ''))
        ]),
        html.Div([
            html.Label('Window Length:'),
            dcc.Input(id={'type': 'event-window-length', 'index': event_key}, type='number', value=event.get('window', {}).get('length', ''))
        ]),
        html.Div([
            html.Label('Window Step:'),
            dcc.Input(id={'type': 'event-window-step', 'index': event_key}, type='number', value=event.get('window', {}).get('step', ''))
        ]),
        html.Div([
            html.Label('Binah:'),
            dcc.Checklist(
                id={'type': 'event-binah', 'index': event_key},
                options=[{'label': 'Binah', 'value': 'Binah'}],
                value=['Binah'] if event.get('Binah', False) else []
            )
        ]),
        html.Div([
            html.Label('Korro:'),
            dcc.Checklist(
                id={'type': 'event-korro', 'index': event_key},
                options=[{'label': 'Korro', 'value': 'Korro'}],
                value=['Korro'] if event.get('Korro', False) else []
            )
        ]),
    ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin-bottom': '10px'})

def create_block_short_component(short_key, short_value):
    return html.Div([
        html.Div([
            html.Label('Full Name:'),
            dcc.Input(id={'type': 'block-short-key', 'index': short_key}, type='text', value=short_key)
        ]),
        html.Div([
            html.Label('Short:'),
            dcc.Input(id={'type': 'block-short-value', 'index': short_key}, type='text', value=short_value)
        ]),
    ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin-bottom': '10px'})

@callback(
    Output('events-container', 'children'),
    Input('add-event-button', 'n_clicks'),
    State('events-container', 'children')
)
def add_event(n_clicks, children):
    if n_clicks > 0:
        new_event_key = f'event_{n_clicks}'
        new_event = create_event_component(new_event_key, {"name": "", "start": "", "end": ""})
        children.append(new_event)
    return children

@callback(
    Output('blocks-shorts-container', 'children'),
    Input('add-block-short-button', 'n_clicks'),
    State('blocks-shorts-container', 'children')
)
def add_block_short(n_clicks, children):
    if n_clicks > 0:
        new_block_short = create_block_short_component("", "")
        children.append(new_block_short)
    return children

@callback(
    Output('save-message', 'children'),
    Input('save-button', 'n_clicks'),
    State('project-name', 'value'),
    State('sampling-rate', 'value'),
    State('events-container', 'children'),
    State('blocks-shorts-container', 'children'),
    State('channel-checklist', 'value'),
    State('output-path', 'value'),
    State('input-path', 'value'),
    State('hdf5-path', 'value')
)
def save_project(n_clicks, project_name, sampling_rate, events, blocks_shorts, channels, output_path, input_path, hdf5_path):
    if n_clicks > 0:
        if not project_name:
            return "Project name is required to save the settings."

        project_settings = {
            "parameters": {
                "sampling_rate": sampling_rate
            },
            "events": {},
            "blocks_shorts": {},
            "channel": channels,
            "output": {
                "path": output_path
            },
            "input": {
                "path": input_path
            },
            "HDF5": {
                "path": hdf5_path
            }
        }

        for event in events:
            event_props = event['props']['children']
            event_key = event_props[0]['props']['children']
            event_name = event_props[1]['props']['children'][1]['props']['value']
            event_start = event_props[2]['props']['children'][1]['props']['value']
            event_end = event_props[3]['props']['children'][1]['props']['value']
            event_group = event_props[4]['props']['children'][1]['props']['value']
            event_window_length = event_props[5]['props']['children'][1]['props']['value']
            event_window_step = event_props[6]['props']['children'][1]['props']['value']
            event_binah = 'Binah' in event_props[7]['props']['children'][1]['props']['value']
            event_korro = 'Korro' in event_props[8]['props']['children'][1]['props']['value']
            
            project_settings["events"][event_key] = {
                "name": event_name,
                "start": event_start,
                "end": event_end
            }

            if event_group:
                project_settings["events"][event_key]["group"] = event_group
            
            if event_window_length and event_window_step:
                project_settings["events"][event_key]["window"] = {
                    "length": event_window_length,
                    "step": event_window_step
                }

            if event_binah:
                project_settings["events"][event_key]["Binah"] = event_binah
            
            if event_korro:
                project_settings["events"][event_key]["Korro"] = event_korro

        for block_short in blocks_shorts:
            block_short_props = block_short['props']['children']
            short_key = block_short_props[0]['props']['children'][1]['props']['value']
            short_value = block_short_props[1]['props']['children'][1]['props']['value']
            if short_key and short_value:
                project_settings["blocks_shorts"][short_key] = short_value

        save_project_settings(project_name, project_settings)
        return "Project settings saved successfully!"
    return ""