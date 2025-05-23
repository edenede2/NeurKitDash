from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import polars as pl
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

dash.register_page(__name__, name="Dataset Preview", order = 5)

pages = {}

for page in os.listdir(r"C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates"):
    page_name = page.split("_")[0]
    page_value = page.split("_")[0]

    pages[page_name] = page_value




layout = html.Div([
    dbc.Container(
        [
            html.H1("Dataset Preview", className="display-3"),
            html.P("This is the dataset preview page", className="lead"),
            html.Hr(className="my-2"),
            html.Div('In this page you can preview the data of the selected project, remove subjects and see the details of the data.', style={"fontSize": 20, "textAlign": "center"}),
        ]
    ),
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Select Project", style={"fontSize": 50, "textAlign": "center"}),
                        ]),
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id="project-preview-dropdown",
                                    options=[{"label": project, "value": project} for project in pages],
                                    style={"width": "60%", "color": "black", "margin": "auto"},
                                    value=None
                                )
                            ]),
                        dbc.Col(
                            [
                                dbc.Button("Submit", id="submit-button", color="primary", style={"margin": "auto"})
                            ]
                        )
                ]
            )
        ]
    ),
    dbc.Container(
        [
            dcc.Loading(
                [html.Div(id="preview-table-container", children = [])],
                overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white"},
                custom_spinner=html.H2(["Loading subjects data...", dbc.Spinner(color="primary")]),
            ),
        ]
    ),
    dbc.Container(
        [
            dcc.Loading(
                [html.Div(id="preview-pie-chart-container", children = [])],
                overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white"},
                custom_spinner=html.H2(["Loading pie chart...", dbc.Spinner(color="primary")]),
            ),
        ]
    ),
    dbc.Container(
        [
            dcc.Loading(
                [html.Div(id="preview-details-container", children = [])],
                overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white"},
                custom_spinner=html.H2(["Loading details...", dbc.Spinner(color="primary")]),
            ),
        ]
    )
])

@callback(
    Output("preview-table-container", "children"),
    Input("submit-button", "n_clicks"),
    State("project-preview-dropdown", "value")
)
def update_table(n_clicks, project):
    if n_clicks is None:
        raise PreventUpdate

    fibro_hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware.hdf5')

    if not os.path.exists(fibro_hdf_file):
        if not os.path.exists(fibro_hdf_file.replace(".hdf5", ".parquet")):
            return html.H2("File not found")
        
    
    data_df = pl.read_parquet(fibro_hdf_file.replace(".hdf5", ".parquet"))

    data_df = (
        data_df
        .drop_nulls(subset=['Id'])
        .select(
            'Id',
            'event_name',
            'duration(minutes)',
            'done'
        )
    )


    # subject_list = [group for group in file.keys() if re.search(r'^sub_', group)]
    # file.close()
    total_subjects = len(data_df)
    results_list = {}
    print(f'Total subjects: {total_subjects}')


    subjects_df = data_df.to_pandas()
    print(subjects_df.head())

    subjects_df['Index'] = subjects_df.index

    subjects_df['done'] = subjects_df['done'].astype(bool)

    columns_def = [
        {"headerName": "Index", "field": "Index", "checkboxSelection": True},
        {"headerName": "Id", "field": "Id"},
        {"headerName": "Event Name", "field": "event_name"},
        {"headerName": "Duration (minutes)", "field": "duration(minutes)"},
        {"headerName": "Done", "field": "done"}
    ]
    rows = subjects_df.to_dict("records")

    sub_table = dag.AgGrid(
        id={
            "type": "sub-table",
            "index": 1
        },
        columnDefs=columns_def,
        rowData=rows,
        defaultColDef={'resizable': True, 'sortable': True, 'filter': True},
        columnSize='responsiveSizeToFit',
        dashGridOptions={'pagination': True, 'paginationPageSize': 10, 'rowSelection': 'multiple'},
    )

    remove_button = dbc.Button("Remove", 
                               id={
                                      "type": "remove-button",
                                      "index": 1
                                 },
                                 color="danger",
                                    style={"margin": "auto"}
                                )

    children = html.Div([sub_table, remove_button])

    return children


@callback(
    Output({"type": "sub-table", "index": MATCH}, "rowData"),
    Input({"type": "remove-button", "index": MATCH}, "n_clicks"),
    State({"type": "sub-table", "index": ALL}, "selectedRows"),
    State("project-preview-dropdown", "value")
)
def remove_subject(n_clicks, selected_rows, project):
    if n_clicks is None:
        raise PreventUpdate

    fibro_hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware.hdf5')
    fibro_par_file = os.path.join(fibro_hdf_file, f'{project}_mindware.parquet')

    par = False
    hdf = False

    if not os.path.exists(fibro_par_file):
        print("Parquet file not found")
    else:
        par = True
        selected_rows = selected_rows[0]
        print(selected_rows)
        # events_df = pd.read_hdf(fibro_hdf_file, key="events_data")
        events_df = pl.read_parquet(fibro_par_file)
        

        selected_subjects = [row['Id'] for row in selected_rows]
        selected_events = [row['event_name'] for row in selected_rows]

        events_df = (
            events_df
            .filter(
                ~pl.col('Id').is_in(selected_subjects),
                ~pl.col('event_name').is_in(selected_events)
            )

        )

        print("after", events_df.head())

        events_df = (
            events_df
            .drop_nulls(subset=['Id'])
        )

        print(events_df.head())

        print(events_df.dtypes)

                

        
        events_df.write_parquet(fibro_par_file)



    if not os.path.exists(fibro_hdf_file):
        print("HDF file not found")
    else:
        hdf = True
        file = h5py.File(fibro_hdf_file, "a")
        for subject in selected_subjects:
            if f'{subject}' in file:
                del file[f'{subject}']
        file.close()
        

    if not par and not hdf:
        return html.H2("There was an error removing the subjects")

    # events_df.to_hdf(fibro_hdf_file, key="events_data", mode="a", format="table", append=False)

    updated_df = (
        events_df
        .select(
            'Id',
            'event_name',
            'duration(minutes)',
            'done'
        )
        .with_row_index()
    )
    

    columns_def = [
        {"headerName": "Index", "field": "Index", "checkboxSelection": True},
        {"headerName": "Subject ID", "field": "Id"},
        {"headerName": "Event Name", "field": "event_name"},
        {"headerName": "Duration (minutes)", "field": "duration(minutes)"},
        {"headerName": "Done", "field": "done"}
    ]
    rows = updated_df.to_pandas().to_dict("records")

    return rows




            