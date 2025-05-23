# from flask import session
# from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
# import os
# from dash.exceptions import PreventUpdate
# import dash_bootstrap_components as dbc
# import plotly.graph_objs as go
# import pandas as pd
# import pickle as pkl
# import h5py
# import numpy as np
# import webview
# # import dash_core_components as dcc
# from flask import Flask
# import neurokit2 as nk
# import base64
# import json
# import io
# import re
# from pymongo import MongoClient
# import dash


# dash.register_page(__name__, name='Loading_PsychoPy')

# pages = {}

# for page in os.listdir(r"C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates"):
#     page_name = page.split("_")[0]
#     page_value = page.split("_")[0]

#     pages[page_name] = page_value



# layout = html.Div([
#     html.Div("PsychoPy Data Loading", style={"font-size": "30px", "margin-top": "20px", "margin-bottom": "20px"}),
#     html.Div("Select Project", style={"font-size": "20px", "margin-top": "20px", "margin-bottom": "20px"}),

#     dcc.Dropdown(
#         id='project-dropdown-psypy',
#         options=[{'label': i, 'value': i} for i in pages.keys()],
#         style={"color": "black"},
#         value=None
#     ),

#     dbc.Container(
#         [
#             html.Div("Upload PsychoPy CSV file", style={"font-size": "20px", "margin-top": "20px", "margin-bottom": "20px"}),
#             html.Hr(),
#             dcc.Upload(
#                 id='upload-psypy',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '60px',
#                     'lineHeight': '60px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '10px'
#                 },
#                 # Allow multiple files to be uploaded
#                 multiple=False
#             ),

#             dcc.Loading(
#                 [dcc.ConfirmDialog(
#                     id='confirm-psypy',
#                     message='',
#                     ),],
#                     overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white", "filter": "blur(2px)"},
#                     custom_spinner=html.H2(["Uploading PsychoPy Data...", dbc.Spinner(color="primary")]),
#             ),
#             dbc.Container(
#                 [
#                     html.Div("Check for updates in the Google Drive folders", style={"font-size": 20, "textAlign": "center"}),
#                     html.Hr(),
#                     dbc.Button("Check for updates", id="check-updates-psypy", color="primary", className="mr-1", style={"margin": "10px"}),
#                     dcc.Loading(
#                         [dcc.ConfirmDialog(
#                             id='update-confirm-psypy',
#                             message='',
#                             ),],
#                         overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white", "filter": "blur(2px)"},
#                         custom_spinner=html.H2(["Checking for updates...", dbc.Spinner(color="primary")]),
#                     ),
#                         dbc.Modal(
#                             [
#                                 dbc.ModalHeader("Updates found"),
#                                 dbc.ModalBody(
#                                     [
#                                         dcc.Checklist(id="subject-checklist-psypy"),
#                                         dbc.Button("Update selected subjects", id="update-google-psypy", color="primary", className="mr-1", style={"margin": "10px"}),

#                                     ]
#                                 ),
#                                 dbc.ModalFooter(
#                                     dbc.Button("Close", id="close-update-psypy", className="ml-auto")
#                                 ),
#                             ],
#                             id="update-modal-psypy",
#                             is_open=False,
#                         ),
#                         dcc.ConfirmDialog(
#                             id='no-updates-psypy',
#                             message='No updates found',
#                         ),
#                 ]
#             ),

#             dcc.Store(id="uploaded-psypy-store"),
#             dcc.Store(id="subjects-psypy-to-update-store"),
#         ]
#     ),
# ])


# @callback(
#     [Output("confirm-psypy", "message"),
#      Output("confirm-psypy", "displayed"),
#      Output("upload-psypy", "contents"),],
#      [Input("project-dropdown-psypy", "value"),
#       Input("upload-psypy", "contents")],
#       State("upload-psypy", "filename"),
#       prevent_initial_call=True,
# )
# def upload_psypy_data(project, contents, filename):

#     if contents:

#         if project:
#             with open(f"pages/templates/{project}_template.json") as f:
#                 json_data = json.load(f)

                
#                 f.close()
#             trial_data = json_data["psychopy_map"]

#             try:
#                 content_type, content_string = contents.split(',')
#                 decoded_data = base64.b64decode(content_string)

#                 psypy_df = pd.read_csv(io.StringIO(decoded_data.decode('utf-8')))
#                 print('Data uploaded successfully')

#                 time_col_name = trial_data["Time"]
#                 label_col_name = trial_data["Label"]
#                 n_trials = trial_data["n_trials_per_block"]
#                 blocks_map = trial_data["blocks"]

#                 time_values = psypy_df[time_col_name].values.flatten()
#                 label_values = psypy_df[label_col_name].values.flatten()

#                 sub_id = filename.split("_")[0] + "_" + filename.split("_")[1]

#                 hdf_file_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
#                 hdf_file_path = f'{project}_mindware.hdf5'
                

#                 events_df = pd.read_hdf(hdf_file_path, key="events_data")
#                 events_df = events_df[events_df["subject_id"] == sub_id]




#             except Exception as e:
#                 return str(e), True, None


