

from flask import session
from dash import dcc, html, Dash, CeleryManager, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objs as go
import pandas as pd
import h5py
import polars as pl
import numpy as np
import os
# import dash_core_components as dcc
from flask import Flask
import neurokit2 as nk
import base64
import json
import shutil
import io
import re
import dash
import subprocess
import threading
import logging
import sys
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pages.paths import DATA_PATH, TEMPLATES_PATH, PAGES_DIR, LOG_PATH, project_names
from pages.data_manager import atomic_write

logger = logging.getLogger(__name__)
_analysis_lock = threading.Lock()
_analysis_runs = {}

# Helper function to create initial column filter placeholder
def create_column_filters(sample_row):
    """Create initial placeholder for filtering components"""
    return html.Div("Loading filter options...", style={"text-align": "center", "margin": "20px"})



dash.register_page(__name__, name="NeuroKit Values New", path="/neurokit_values_new", order = 3)



pages = {name: name for name in project_names()}


layout = html.Div([
    dbc.Container(
        [
            dbc.Row(
                [
                    html.Div("NeuroKit Analysis Page", style={"font-size": "30px", "margin-top": "20px", "margin-bottom": "20px"}),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Select Project", style={"font-size": 20, "text-align": "right"}),
                        ]),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="project-dropdown-neurokit",
                                options=[{'label': k, 'value': v} for k, v in pages.items()
                                ],
                                style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"},
                                value="fibro"
                            ),
                        ]),
                    dbc.Col([
                        dbc.Button(
                            id="load-button-neurokit",
                            children="Load",
                            n_clicks=0,
                            style={"margin": "auto", "textAlign": "left"}
                        )
                    ]),
                    dbc.Col([
                        dbc.Button(
                            id="check-update-button-neurokit",
                            children="Check for Update",
                            n_clicks=0,
                            style={"margin": "auto", "textAlign": "left"}
                        ),
                        html.Div(id="neurokit-update-container"),
                        dbc.Button(
                            id="update-button-neurokit",
                            children="Update",
                            n_clicks=0,
                            style={"margin": "auto", "textAlign": "left"}
                        ),

                    ]),
                    dbc.Col([
                        dbc.Button(
                            id="run-all-subjects-button-neurokit",
                            children="Run All Subjects",
                            n_clicks=0,
                            color="success",
                            style={"margin": "auto", "textAlign": "left"}
                        ),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div("Filter by Events (optional):", style={"font-size": 14, "margin-top": "10px"}),
                    ], width=3),
                    dbc.Col([
                        dcc.Dropdown(
                            id="event-filter-dropdown-neurokit",
                            options=[],
                            multi=True,
                            placeholder="Select events (leave empty for all)",
                            style={"width": "100%", "color": "black"}
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Button(
                            id="load-events-button-neurokit",
                            children="Load Events",
                            n_clicks=0,
                            size="sm",
                            style={"margin-top": "10px"}
                        ),
                    ], width=3),
                ]),
        ]),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Run Analysis Script",
                                id="run-script-button",
                                color="primary",
                                style={"margin-top": "20px"}
                            ),
                            width="auto",
                        )
                    ]
                ),
                html.Div(id="script-output-container"),
                html.Div(id="analysis-run-status"),
            ],
        ),

        dcc.Interval(id="script-status-interval", interval=1000, n_intervals=0),
        dcc.Store(id="neurokit-data-store", data={}),
        html.Div(id="neurokit-values-container"),
        html.Div(id="neurokit-plot-container")

])



        



        

# Callback for loading neurokit values
@callback(
    Output("neurokit-values-container", "children"),
    [Input("load-button-neurokit", "n_clicks")],
    [State("project-dropdown-neurokit", "value")],
    prevent_initial_call=True
)
def load_neurokit_values(n_clicks, project):

    

    if n_clicks == 0:
        raise PreventUpdate
    
    print("Loading Fibro data")
    fibro_hdf_file = DATA_PATH
    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware_results.parquet')

    results = []

    if os.path.exists(fibro_hdf_file):
        results = pl.read_parquet(fibro_hdf_file).to_dicts()
    else:
        print(f'File {fibro_hdf_file} does not exist')

    if results != []:


        columnsDefs = [
            {'headerName': 'id', 'valueGetter': {'function': 'params.node.id'}, 'checkboxSelection': True},
        ]

        columnsDefs += [{'headerName': param, 'field': param} for param in dict(results[0]).keys()]


        ag_results = dag.AgGrid(
            id={'type': 'neurokit-results-ag-grid', 'index': 1},
            columnDefs=columnsDefs,
            defaultColDef={'filter': True, 'sortable': True, 'resizable': True},
            rowData=results,
            columnSize='autoSize',
            dashGridOptions={'pagination': True, 'paginationPageSize': 30, 'undoRedoCellEditing': True, 'rowSelection': 'multiple' },
        )

        # Dynamic filtering system - select columns and add filters
        filter_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("🔍 Dynamic Data Filters", style={"margin-bottom": "0px"}),
                ], width=8),
                dbc.Col([
                    dbc.Switch(
                        id={'type': 'neurokit-filter-toggle', 'index': 1},
                        label="Enable Filters",
                        value=False,
                        style={"margin-top": "5px"}
                    )
                ], width=4)
            ], style={"margin-top": "20px", "margin-bottom": "10px"}),

            dbc.Collapse(
                html.Div([
                    html.P("Select columns to filter, then configure filter logic based on data type.",
                           style={"color": "gray", "font-size": "14px", "margin-bottom": "15px"}),

                    # Add new filter row
                    dbc.Card([
                        dbc.CardHeader("Add New Filter"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Column:", style={"font-weight": "bold", "margin-bottom": "5px"}),
                                    dcc.Dropdown(
                                        id={'type': 'neurokit-column-selector', 'index': 1},
                                        options=[{'label': col, 'value': col} for col in results[0].keys()],
                                        placeholder="Choose column to filter...",
                                        style={"margin-bottom": "10px"}
                                    )
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "➕ Add Filter",
                                        id={'type': 'neurokit-add-filter-button', 'index': 1},
                                        color="success",
                                        size="sm",
                                        disabled=True
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Button(
                                        "🗑️ Clear All",
                                        id={'type': 'neurokit-clear-all-filters-button', 'index': 1},
                                        color="danger",
                                        size="sm",
                                        outline=True
                                    )
                                ], width=6)
                            ])
                        ])
                    ], style={"margin-bottom": "15px"}),

                    # Active filters container
                    html.Div(id={'type': 'neurokit-active-filters-container', 'index': 1}, children=[]),

                    # Apply filters section
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "🎯 Apply All Filters",
                                id={'type': 'neurokit-apply-filters-button', 'index': 1},
                                color="primary",
                                size="sm",
                                style={"width": "100%"}
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                "🔄 Reset to Original",
                                id={'type': 'neurokit-reset-filters-button', 'index': 1},
                                color="warning",
                                size="sm",
                                style={"width": "100%"}
                            )
                        ], width=6)
                    ], style={"margin-top": "15px"}),

                    dbc.Alert(
                        id={'type': 'neurokit-filter-status', 'index': 1},
                        children="Ready to add filters",
                        color="info",
                        style={"margin-top": "15px"},
                        dismissable=True,
                        is_open=False
                    )
                ]),
                id={'type': 'neurokit-filter-collapse', 'index': 1},
                is_open=False
            ),
            html.Hr(style={"margin-top": "15px"})
        ])

        # Store original data for filter reset and active filters tracking
        original_data_store = dcc.Store(
            id={'type': 'neurokit-original-data-store', 'index': 1},
            data=results
        )

        active_filters_store = dcc.Store(
            id={'type': 'neurokit-active-filters-store', 'index': 1},
            data=[]
        )

        target_dropdown = dcc.Dropdown(
            id={'type': 'neurokit-target-dropdown', 'index': 1},
            options=[{'label': f, 'value': f} for f in results[0].keys()],
            multi=True,
            value=[list(results[0].keys())[0]],
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        x_dropdown = dcc.Dropdown(
            id={'type': 'neurokit-x-dropdown', 'index': 1},
            options=[{'label': f, 'value': f} for f in results[0].keys()],
            value=list(results[0].keys())[0],
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        agg_function = dcc.Dropdown(
            id={'type': 'neurokit-agg-function-dropdown', 'index': 1},
            options=[{'label': 'None', 'value': 'none'}, {'label': 'Mean', 'value': 'mean'},
                        {'label': 'Standard Deviation', 'value': 'std'}, {'label': 'Minimum', 'value': 'min'},
                        {'label': 'Maximum', 'value': 'max'}, {'label': 'Sum', 'value': 'sum'}, {'label': 'Count', 'value': 'count'}],
            value='none',
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        plot_type = dcc.Dropdown(
            id={'type': 'neurokit-plot-type-dropdown', 'index': 1},
            options=[{'label': 'Bar', 'value': 'bar'}, {'label': 'Line', 'value': 'line'}, {'label': 'Scatter', 'value': 'scatter'}],
            value='bar',
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        stack_checkbox = dbc.Checkbox(
            id={'type': 'neurokit-stack-checkbox', 'index': 1},
            label="Stack Parameters (Bar Plot Only)",
            style={"margin-top": "10px"}
        )

        normalize_checkbox = dbc.Checkbox(
            id={'type': 'neurokit-normalize-checkbox', 'index': 1},
            label="Normalize Parameters (Line Plot Only)",
            style={"margin-top": "10px"}
        )

        clustering_checkbox = dbc.Checkbox(
            id={'type': 'neurokit-clustering-checkbox', 'index': 1},
            label="Apply Clustering (Scatter Plot Only)",
            style={"margin-top": "10px"}
        )

        clustering_type = dcc.Dropdown(
            id={'type': 'neurokit-clustering-type-dropdown', 'index': 1},
            options=[
                {'label': 'K-Means', 'value': 'kmeans'},
                {'label': 'Agglomerative', 'value': 'agglomerative'},

            ],
            value='kmeans',
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        apply_dimensions_reduction = dbc.Checkbox(
            id={'type': 'neurokit-apply-dimensions-reduction-checkbox', 'index': 1},
            label="Apply Dimensions Reduction",
            style={"margin-top": "10px"}
        )

        dimensions_reduction = dcc.Dropdown(
            id={'type': 'neurokit-dimensions-reduction-dropdown', 'index': 1},
            options=[
                {'label': 'PCA', 'value': 'pca'},
                {'label': 't-SNE', 'value': 'tsne'},
            ],
            value='pca',
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        n_clusters = dcc.Slider(
            id={'type': 'neurokit-n-clusters-slider', 'index': 1},
            min=2,
            max=10,
            step=1,
            value=3,
            marks={i: str(i) for i in range(2, 11)}
        )

        show_button = dbc.Button(
            "Show Plot",
            id={'type': 'neurokit-show-plot-button', 'index': 1},
            color="primary",
            style={"margin-top": "20px"}
        )

        file_name = dcc.Input(
            id={'type': 'neurokit-file-name', 'index': 1},
            type='text',
            placeholder='Enter file name',
            value='MindwareData_',
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        user_name = dcc.Input(
            id={'type': 'neurokit-user-name', 'index': 1},
            type='text',
            placeholder='Enter user name',
            style={"width": "60%", "color": "black", "margin": "auto", "textAlign": "left"}
        )

        download_button = html.A(
            dbc.Button("Download CSV", id={'type': 'neurokit-download-button', 'index': 1}, color="primary"),
            href="",
            target="_blank"
        )

        confirm_message = dcc.ConfirmDialog(
            id={'type': 'download-confirm-dialog', 'index': 1},
            message="File downloaded successfully!",
        )

        return html.Div([
            ag_results,
            original_data_store,
            active_filters_store,
            filter_section,
            target_dropdown,
            x_dropdown,
            agg_function,
            plot_type,
            stack_checkbox,
            normalize_checkbox,
            clustering_checkbox,
            clustering_type,
            apply_dimensions_reduction,
            dimensions_reduction,
            n_clusters,
            show_button,
            file_name,
            user_name,
            download_button,
            confirm_message
        ])

    else:
        print("No results in hdf file")

    print('Loading Fibro data')
    if results == []:
        return html.Div("No results in hdf file, please run the analysis script first!")


    else:
        raise PreventUpdate



# Callback to toggle filter visibility for performance
@callback(
    Output({'type': 'neurokit-filter-collapse', 'index': ALL}, 'is_open'),
    [Input({'type': 'neurokit-filter-toggle', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def toggle_filters(toggle_values):
    if toggle_values:
        return toggle_values
    return [False]

# Callback to enable/disable add filter button when column is selected
@callback(
    Output({'type': 'neurokit-add-filter-button', 'index': ALL}, 'disabled'),
    [Input({'type': 'neurokit-column-selector', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def enable_add_filter_button(column_values):
    if column_values and column_values[0]:
        return [False]
    return [True]

# Callback to add new filter based on selected column and data type
@callback(
    [Output({'type': 'neurokit-active-filters-container', 'index': ALL}, 'children'),
     Output({'type': 'neurokit-active-filters-store', 'index': ALL}, 'data'),
     Output({'type': 'neurokit-column-selector', 'index': ALL}, 'value')],
    [Input({'type': 'neurokit-add-filter-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'neurokit-clear-all-filters-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'filter-remove-button', 'index': ALL}, 'n_clicks')],
    [State({'type': 'neurokit-column-selector', 'index': ALL}, 'value'),
     State({'type': 'neurokit-active-filters-store', 'index': ALL}, 'data'),
     State({'type': 'neurokit-original-data-store', 'index': ALL}, 'data')],
    prevent_initial_call=True
)
def manage_filters(add_clicks, clear_clicks, remove_clicks, selected_columns, active_filters, original_data):
    ctx = dash.callback_context

    if not ctx.triggered or not original_data or not original_data[0]:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id']

    # Initialize active filters if empty
    if not active_filters:
        active_filters = [[]]

    current_filters = active_filters[0] if active_filters[0] else []

    # Clear all filters
    if 'clear-all-filters-button' in trigger_id:
        return [[]], [[]], [None]

    # Remove specific filter
    if 'filter-remove-button' in trigger_id:
        button_index = ctx.triggered[0]['prop_id'].split('.')[0].split(',')[1].split(':')[1]
        try:
            filter_index = int(button_index.strip('"}'))
            if 0 <= filter_index < len(current_filters):
                current_filters.pop(filter_index)
        except (ValueError, IndexError):
            pass
        return [create_filter_components(current_filters, original_data[0])], [current_filters], [None]

    # Add new filter
    if 'add-filter-button' in trigger_id and selected_columns and selected_columns[0]:
        column_name = selected_columns[0]

        # Check if filter for this column already exists
        if not any(f['column'] == column_name for f in current_filters):
            df = pd.DataFrame(original_data[0])
            data_type = get_column_data_type(df[column_name])

            new_filter = {
                'column': column_name,
                'data_type': data_type,
                'filter_type': 'range' if data_type == 'numeric' else 'contains',
                'values': None
            }

            current_filters.append(new_filter)

        return [create_filter_components(current_filters, original_data[0])], [current_filters], [None]

    return [create_filter_components(current_filters, original_data[0])], [current_filters], [None]

# Helper function to determine column data type
def get_column_data_type(column_series):
    if column_series.dtype in ['int64', 'float64', 'int32', 'float32']:
        return 'numeric'
    elif column_series.dtype == 'object':
        return 'categorical'
    else:
        return 'text'

# Helper function to create filter components
def create_filter_components(filters_list, original_data):
    if not filters_list:
        return [html.Div("No active filters", style={"text-align": "center", "color": "gray", "margin": "20px"})]

    df = pd.DataFrame(original_data)
    components = []

    for i, filter_config in enumerate(filters_list):
        col_name = filter_config['column']
        data_type = filter_config['data_type']

        if col_name not in df.columns:
            continue

        if data_type == 'numeric':
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())

            filter_component = dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.Strong(f"📊 {col_name}"),
                            html.Small(f" (Numeric: {min_val:.2f} - {max_val:.2f})", style={"color": "gray"})
                        ], width=10),
                        dbc.Col([
                            dbc.Button("×",
                                     id={'type': 'filter-remove-button', 'index': i},
                                     color="danger", size="sm", outline=True)
                        ], width=2)
                    ])
                ]),
                dbc.CardBody([
                    html.Label("Filter Type:", style={"font-weight": "bold", "margin-bottom": "5px"}),
                    dcc.Dropdown(
                        id={'type': 'filter-type-dropdown', 'column': col_name, 'index': i},
                        options=[
                            {'label': 'Range (Between)', 'value': 'range'},
                            {'label': 'Greater Than', 'value': 'greater'},
                            {'label': 'Less Than', 'value': 'less'},
                            {'label': 'Equal To', 'value': 'equal'}
                        ],
                        value='range',
                        style={"margin-bottom": "10px"}
                    ),
                    html.Div(id={'type': 'filter-input-container', 'column': col_name, 'index': i}, children=[
                        dcc.RangeSlider(
                            id={'type': 'filter-range-slider', 'column': col_name, 'index': i},
                            min=min_val,
                            max=max_val,
                            value=[min_val, max_val],
                            marks={min_val: f'{min_val:.1f}', max_val: f'{max_val:.1f}'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ])
                ])
            ], style={"margin-bottom": "10px"})

        else:  # categorical or text
            unique_values = [str(val) for val in df[col_name].unique() if pd.notna(val)]

            filter_component = dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.Strong(f"🏷️ {col_name}"),
                            html.Small(f" ({len(unique_values)} unique values)", style={"color": "gray"})
                        ], width=10),
                        dbc.Col([
                            dbc.Button("×",
                                     id={'type': 'filter-remove-button', 'index': i},
                                     color="danger", size="sm", outline=True)
                        ], width=2)
                    ])
                ]),
                dbc.CardBody([
                    html.Label("Filter Type:", style={"font-weight": "bold", "margin-bottom": "5px"}),
                    dcc.Dropdown(
                        id={'type': 'filter-type-dropdown', 'column': col_name, 'index': i},
                        options=[
                            {'label': 'Contains Text', 'value': 'contains'},
                            {'label': 'Equals Exactly', 'value': 'equals'},
                            {'label': 'Select Multiple', 'value': 'select_multiple'},
                            {'label': 'Starts With', 'value': 'starts_with'},
                            {'label': 'Ends With', 'value': 'ends_with'}
                        ],
                        value='contains',
                        style={"margin-bottom": "10px"}
                    ),
                    html.Div(id={'type': 'filter-input-container', 'column': col_name, 'index': i}, children=[
                        dcc.Input(
                            id={'type': 'filter-text-input', 'column': col_name, 'index': i},
                            type='text',
                            placeholder=f"Enter text to search in {col_name}...",
                            style={"width": "100%"}
                        )
                    ])
                ])
            ], style={"margin-bottom": "10px"})

        components.append(filter_component)

    return components

# Callback to update filter input components based on selected filter type
@callback(
    Output({'type': 'filter-input-container', 'column': MATCH, 'index': MATCH}, 'children'),
    [Input({'type': 'filter-type-dropdown', 'column': MATCH, 'index': MATCH}, 'value')],
    [State({'type': 'neurokit-original-data-store', 'index': ALL}, 'data')],
    prevent_initial_call=True
)
def update_filter_input_component(filter_type, original_data):
    ctx = dash.callback_context

    if not ctx.triggered or not original_data or not original_data[0]:
        raise PreventUpdate

    # Get column name from callback context
    column_name = ctx.triggered[0]['prop_id'].split('"column":"')[1].split('"')[0]

    df = pd.DataFrame(original_data[0])

    if column_name not in df.columns:
        raise PreventUpdate

    col_data = df[column_name]

    if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
        # Numeric column
        min_val = float(col_data.min())
        max_val = float(col_data.max())

        if filter_type == 'range':
            return [dcc.RangeSlider(
                id={'type': 'filter-range-slider', 'column': column_name, 'index': ctx.triggered[0]['prop_id'].split('"index":')[1].split(',')[0]},
                min=min_val,
                max=max_val,
                value=[min_val, max_val],
                marks={min_val: f'{min_val:.1f}', max_val: f'{max_val:.1f}'},
                tooltip={"placement": "bottom", "always_visible": True}
            )]
        else:
            return [dcc.Input(
                id={'type': 'filter-numeric-input', 'column': column_name, 'index': ctx.triggered[0]['prop_id'].split('"index":')[1].split(',')[0]},
                type='number',
                placeholder=f"Enter value for {filter_type} comparison",
                min=min_val,
                max=max_val,
                style={"width": "100%"}
            )]

    else:
        # Categorical/text column
        unique_values = [str(val) for val in col_data.unique() if pd.notna(val)]

        if filter_type == 'select_multiple':
            options = [{'label': val, 'value': val} for val in sorted(unique_values)]
            return [dcc.Dropdown(
                id={'type': 'filter-dropdown-input', 'column': column_name, 'index': ctx.triggered[0]['prop_id'].split('"index":')[1].split(',')[0]},
                options=options,
                value=[],
                multi=True,
                placeholder=f"Select {column_name} values...",
                style={"width": "100%"}
            )]
        else:
            return [dcc.Input(
                id={'type': 'filter-text-input', 'column': column_name, 'index': ctx.triggered[0]['prop_id'].split('"index":')[1].split(',')[0]},
                type='text',
                placeholder=f"Enter text for {filter_type} in {column_name}...",
                style={"width": "100%"}
            )]

# Dynamic filter application callback
@callback(
    [Output({"type": "neurokit-results-ag-grid", "index": ALL}, "rowData"),
     Output({'type': 'neurokit-filter-status', 'index': ALL}, 'children'),
     Output({'type': 'neurokit-filter-status', 'index': ALL}, 'is_open'),
     Output({'type': 'neurokit-filter-status', 'index': ALL}, 'color')],
    [Input({'type': 'neurokit-apply-filters-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'neurokit-reset-filters-button', 'index': ALL}, 'n_clicks')],
    [State({'type': 'neurokit-original-data-store', 'index': ALL}, 'data'),
     State({'type': 'neurokit-active-filters-store', 'index': ALL}, 'data'),
     State({'type': 'filter-range-slider', 'column': ALL, 'index': ALL}, 'value'),
     State({'type': 'filter-numeric-input', 'column': ALL, 'index': ALL}, 'value'),
     State({'type': 'filter-text-input', 'column': ALL, 'index': ALL}, 'value'),
     State({'type': 'filter-dropdown-input', 'column': ALL, 'index': ALL}, 'value'),
     State({'type': 'filter-type-dropdown', 'column': ALL, 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def apply_dynamic_filters(apply_clicks, reset_clicks, original_data, active_filters, range_values, numeric_values, text_values, dropdown_values, filter_types):
    ctx = dash.callback_context

    if not ctx.triggered or not original_data or not original_data[0]:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id']

    # If reset button was clicked
    if 'reset-filters-button' in trigger_id:
        return [original_data[0]], ["All filters reset - showing complete dataset"], [True], ["success"]

    # If apply button was clicked
    if 'apply-filters-button' not in trigger_id:
        raise PreventUpdate

    df = pd.DataFrame(original_data[0])
    original_count = len(df)
    filters_applied = []

    # If no active filters, return original data
    if not active_filters or not active_filters[0]:
        return [original_data[0]], ["ℹ️ No filters to apply - showing all data"], [True], ["info"]

    # Parse filter states and apply them
    try:
        # Create filter mappings from states
        filter_configs = {}

        # Map range sliders
        if range_values and ctx.states_list and len(ctx.states_list) > 2:
            for i, state in enumerate(ctx.states_list[2]):
                if i < len(range_values) and range_values[i] and 'column' in state['id']:
                    col_name = state['id']['column']
                    filter_configs[col_name] = {'type': 'range', 'value': range_values[i]}

        # Map numeric inputs
        if numeric_values and ctx.states_list and len(ctx.states_list) > 3:
            for i, state in enumerate(ctx.states_list[3]):
                if i < len(numeric_values) and numeric_values[i] is not None and 'column' in state['id']:
                    col_name = state['id']['column']
                    filter_configs[col_name] = {'type': 'numeric', 'value': numeric_values[i]}

        # Map text inputs
        if text_values and ctx.states_list and len(ctx.states_list) > 4:
            for i, state in enumerate(ctx.states_list[4]):
                if i < len(text_values) and text_values[i] and 'column' in state['id']:
                    col_name = state['id']['column']
                    filter_configs[col_name] = {'type': 'text', 'value': text_values[i]}

        # Map dropdown selections
        if dropdown_values and ctx.states_list and len(ctx.states_list) > 5:
            for i, state in enumerate(ctx.states_list[5]):
                if i < len(dropdown_values) and dropdown_values[i] and 'column' in state['id']:
                    col_name = state['id']['column']
                    filter_configs[col_name] = {'type': 'dropdown', 'value': dropdown_values[i]}

        # Map filter types
        filter_type_map = {}
        if filter_types and ctx.states_list and len(ctx.states_list) > 6:
            for i, state in enumerate(ctx.states_list[6]):
                if i < len(filter_types) and filter_types[i] and 'column' in state['id']:
                    col_name = state['id']['column']
                    filter_type_map[col_name] = filter_types[i]

        # Apply each filter
        for filter_info in active_filters[0]:
            col_name = filter_info['column']

            if col_name not in df.columns or col_name not in filter_configs:
                continue

            filter_config = filter_configs[col_name]
            filter_type = filter_type_map.get(col_name, 'contains')

            before_count = len(df)

            # Apply filter based on type and value
            if filter_config['type'] == 'range' and len(filter_config['value']) == 2:
                min_val, max_val = filter_config['value']
                df = df[(df[col_name] >= min_val) & (df[col_name] <= max_val)]
                if len(df) != before_count:
                    filters_applied.append(f"{col_name}: {min_val:.1f} - {max_val:.1f}")

            elif filter_config['type'] == 'numeric':
                val = filter_config['value']
                if filter_type == 'greater':
                    df = df[df[col_name] > val]
                elif filter_type == 'less':
                    df = df[df[col_name] < val]
                elif filter_type == 'equal':
                    df = df[df[col_name] == val]

                if len(df) != before_count:
                    filters_applied.append(f"{col_name} {filter_type} {val}")

            elif filter_config['type'] == 'text':
                text_val = str(filter_config['value']).strip()
                if text_val:
                    if filter_type == 'contains':
                        mask = df[col_name].astype(str).str.contains(text_val, case=False, na=False)
                    elif filter_type == 'equals':
                        mask = df[col_name].astype(str).str.lower() == text_val.lower()
                    elif filter_type == 'starts_with':
                        mask = df[col_name].astype(str).str.startswith(text_val, na=False)
                    elif filter_type == 'ends_with':
                        mask = df[col_name].astype(str).str.endswith(text_val, na=False)
                    else:
                        mask = df[col_name].astype(str).str.contains(text_val, case=False, na=False)

                    df = df[mask]
                    if len(df) != before_count:
                        filters_applied.append(f"{col_name} {filter_type} '{text_val}'")

            elif filter_config['type'] == 'dropdown':
                selected_vals = filter_config['value']
                if selected_vals:
                    df = df[df[col_name].astype(str).isin([str(v) for v in selected_vals])]
                    if len(df) != before_count:
                        filters_applied.append(f"{col_name}: {len(selected_vals)} selected")

    except Exception as e:
        return [original_data[0]], [f"❌ Filter error: {str(e)}"], [True], ["danger"]

    filtered_count = len(df)

    if filters_applied:
        status_msg = f"✅ Applied {len(filters_applied)} filters. Showing {filtered_count} of {original_count} rows."
        alert_color = "success"
    else:
        status_msg = f"ℹ️ No filters changed data - showing all {original_count} rows."
        alert_color = "info"

    return [df.to_dict('records')], [status_msg], [True], [alert_color]

# Callback for loading available events
@callback(
    Output("event-filter-dropdown-neurokit", "options"),
    [Input("load-events-button-neurokit", "n_clicks")],
    State("project-dropdown-neurokit", "value"),
    prevent_initial_call=True
)
def load_available_events(n_clicks, project):
    if n_clicks == 0 or not project:
        raise PreventUpdate

    try:
        data_events_path = (DATA_PATH / f"{project}_mindware_events.parquet")

        if not os.path.exists(data_events_path):
            return []

        # Load unique event names
        data_events_df = pl.scan_parquet(data_events_path).select('event_name').unique().collect()
        event_names = sorted(data_events_df['event_name'].to_list())

        return [{'label': event, 'value': event} for event in event_names]
    
    except Exception as e:
        print(f"Error loading events: {e}")
        return []

# Callback for checking for updates
@callback(
    Output("neurokit-update-container", "children"),
    [Input("check-update-button-neurokit", "n_clicks")],
    State("project-dropdown-neurokit", "value"),
    prevent_initial_call=True
)
def check_for_update(n_clicks, project):
    if not n_clicks or not project:
        raise PreventUpdate

    events_path = DATA_PATH / f"{project}_mindware_events.parquet"
    queue_path = DATA_PATH / f"{project}_subjects_to_run.csv"
    keys = ["subject_id", "event_name"]
    if not events_path.exists():
        return f"No events available for {project}. Import recordings first."

    try:
        with _analysis_lock:
            if _analysis_running(project):
                return f"Analysis for {project} is already running."
            events = pl.scan_parquet(events_path).select(keys).unique().collect()
            if queue_path.exists():
                previous = pl.read_csv(queue_path, schema_overrides={"subject_id": pl.String})
                done = previous.filter(pl.col("status") == "done").select(keys).unique()
                pending = events.join(done, on=keys, how="anti")
            else:
                pending = events
            updates = pending.with_columns(project=pl.lit(project), status=pl.lit("New"))
            if queue_path.exists():
                updates = pl.concat([previous.filter(pl.col("status") == "done").select(updates.columns),
                                     updates])
            atomic_write(queue_path, updates.write_csv)
        return f"{pending.height} events pending for {project}."
    except Exception:
        logger.exception("Could not check queued analysis for %s", project)
        return "Could not check queued events. See the app log for details."


def _analysis_running(project):
    run = _analysis_runs.get(project)
    return run is not None and run[0].poll() is None


def _launch_analysis(project, events=None):
    with _analysis_lock:
        if _analysis_running(project):
            return f"Analysis for {project} is already running."
        if events is not None:
            queue_path = DATA_PATH / f"{project}_subjects_to_run.csv"
            queued = events.with_columns(project=pl.lit(project), status=pl.lit("New"))
            atomic_write(queue_path, queued.write_csv)

        LOG_PATH.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = LOG_PATH / f"{project}_analysis_{timestamp}.log"
        command = [sys.executable, "-u", "-m", "pages.scripts.results_generating_smart", project]
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            command.append("--debug")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=PAGES_DIR.parent, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        _analysis_runs[project] = (process, log_path)
        count = f" for {events.height} selected events" if events is not None else ""
        return f"Analysis started{count}."


@callback(
    Output("analysis-run-status", "children"),
    Input("script-status-interval", "n_intervals"),
    Input("project-dropdown-neurokit", "value"),
)
def show_analysis_status(_ticks, project):
    with _analysis_lock:
        run = _analysis_runs.get(project)
        if run is None:
            return ""
        process, log_path = run
        exit_code = process.poll()
    if exit_code is None:
        text = "Analysis running."
    elif exit_code == 0:
        text = "Analysis finished. Reload the results table to view the results."
    else:
        text = f"Analysis ended with errors (exit code {exit_code}). Check the log for details."
    return html.Small(f"{text} Log: {log_path}")


@callback(
    Output("script-output-container", "children"),
    Input("update-button-neurokit", "n_clicks"),
    Input("run-script-button", "n_clicks"),
    Input("run-all-subjects-button-neurokit", "n_clicks"),
    State("project-dropdown-neurokit", "value"),
    State("event-filter-dropdown-neurokit", "value"),
    prevent_initial_call=True,
)
def start_analysis(_update_clicks, _run_clicks, _run_all_clicks, project, selected_events):
    if not project:
        raise PreventUpdate
    try:
        if dash.ctx.triggered_id == "run-all-subjects-button-neurokit":
            path = DATA_PATH / f"{project}_mindware_events.parquet"
            if not path.exists():
                return f"No events available for {project}. Import recordings first."
            events = pl.scan_parquet(path).select("subject_id", "event_name").unique()
            if selected_events:
                events = events.filter(pl.col("event_name").is_in(selected_events))
            events = events.collect()
            if events.is_empty():
                return "No events match the selection."
            return _launch_analysis(project, events)
        return _launch_analysis(project)
    except Exception:
        logger.exception("Could not start analysis for %s", project)
        return "Could not start analysis. See the app log for details."


# Callback for plotting the data
@callback(
    Output("neurokit-plot-container", "children"),
    [Input({"type": "neurokit-results-ag-grid", "index": ALL}, "virtualRowData"),
     Input({"type": "neurokit-show-plot-button", "index": ALL}, "n_clicks")],
    [State({"type": "neurokit-results-ag-grid", "index": ALL}, "selectedRows"),
     State({"type": "neurokit-agg-function-dropdown", "index": ALL}, "value"),
     State({"type": "neurokit-target-dropdown", "index": ALL}, "value"),
     State({"type": "neurokit-plot-type-dropdown", "index": ALL}, "value"),
     State({"type": "neurokit-x-dropdown", "index": ALL}, "value"),
     State({"type": "neurokit-stack-checkbox", "index": ALL}, "value"),
     State({"type": "neurokit-normalize-checkbox", "index": ALL}, "value"),
     State({"type": "neurokit-clustering-checkbox", "index": ALL}, "value"),
     State({"type": "neurokit-clustering-type-dropdown", "index": ALL}, "value"),
     State({"type": "neurokit-apply-dimensions-reduction-checkbox", "index": ALL}, "value"),
     State({"type": "neurokit-dimensions-reduction-dropdown", "index": ALL}, "value"),
     State({"type": "neurokit-n-clusters-slider", "index": ALL}, "value")],
    prevent_initial_call=True
)
def plot_data(virtualRowData, n_clicks, selectedRows, agg_functions, targets, plot_types, x_values, stack_values, normalize_values, clustering_values, clustering_types, apply_dimensions_reduction, dimensions_reduction, n_clusters):
    if not virtualRowData:
        raise PreventUpdate
    
    if not n_clicks:
        raise PreventUpdate

    df = pd.DataFrame(virtualRowData[0])

    selectedRows = pd.DataFrame(selectedRows[0]) if selectedRows else None

    # Exclude selected rows
    if selectedRows is not None:
        df = df[~df.index.isin(selectedRows.index)]        
        

    targets = targets[0]
    agg_function = agg_functions[0]
    plot_type = plot_types[0]
    x_value = x_values[0]
    stack = stack_values[0]
    normalize = normalize_values[0]
    clustering = clustering_values[0]
    clustering_type = clustering_types[0] if clustering_types else None
    apply_dimensions_reduction = apply_dimensions_reduction[0]
    dimensions_reduction = dimensions_reduction[0] if dimensions_reduction else None

    print(f'Cluster number: {n_clusters}')
    num_clusters = n_clusters[0] if n_clusters else 3

    if agg_function != 'none':
        df = df.groupby(x_value).agg({target: agg_function for target in targets}).reset_index()
    if normalize:
        df[targets] = (df[targets] - df[targets].min()) / (df[targets].max() - df[targets].min())



    fig = go.Figure()


    if plot_type == 'bar':
        if stack:
            for target in targets:
                fig.add_trace(go.Bar(x=df[x_value], y=df[target], name=target))
        else:
            for target in targets:
                fig.add_trace(go.Bar(x=df[x_value], y=df[target], name=target))
    elif plot_type == 'line':
        for target in targets:
            fig.add_trace(go.Scatter(x=df[x_value], y=df[target], mode='lines', name=target))
        if normalize:
            fig.update_layout(yaxis_range=[0, 1])
    elif plot_type == 'scatter':
        print("Applying scatter plot")
        if clustering:
            print("Applying clustering")
            if clustering_type == 'kmeans':
                print("Applying KMeans")
                if apply_dimensions_reduction:
                    print("Applying dimensions reduction")
                    if dimensions_reduction == 'pca':
                        print("Applying PCA")
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(df[targets])
                        df['pca-one'] = pca_result[:,0]
                        df['pca-two'] = pca_result[:,1]
                        print("Applying KMeans")
                        cluster_model = KMeans(n_clusters=num_clusters)
                        df['Cluster'] = cluster_model.fit_predict(df[targets])
                    elif dimensions_reduction == 'tsne':
                        print("Applying t-SNE")
                        tsne = TSNE(n_components=2)
                        tsne_result = tsne.fit_transform(df[targets])
                        df['tsne-2d-one'] = tsne_result[:,0]
                        df['tsne-2d-two'] = tsne_result[:,1]
                        print("Applying KMeans")
                        cluster_model = KMeans(n_clusters=num_clusters)
                        df['Cluster'] = cluster_model.fit_predict(df[targets])
                else:
                    print("Applying KMeans")
                    cluster_model = KMeans(n_clusters=num_clusters)
                    df['Cluster'] = cluster_model.fit_predict(df[targets])

            elif clustering_type == 'agglomerative':
                print("Applying Agglomerative")
                if apply_dimensions_reduction:
                    print("Applying dimensions reduction")

                    if dimensions_reduction == 'pca':
                        print("Applying PCA")
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(df[targets])
                        df['pca-one'] = pca_result[:,0]
                        df['pca-two'] = pca_result[:,1]
                        print("Applying Agglomerative")
                        cluster_model = AgglomerativeClustering(n_clusters=num_clusters)
                        df['Cluster'] = cluster_model.fit_predict(df[targets])
                    elif dimensions_reduction == 'tsne':
                        print("Applying t-SNE")
                        tsne = TSNE(n_components=2)
                        tsne_result = tsne.fit_transform(df[targets])
                        df['tsne-2d-one'] = tsne_result[:,0]
                        df['tsne-2d-two'] = tsne_result[:,1]
                        print("Applying Agglomerative")
                        cluster_model = AgglomerativeClustering(n_clusters=num_clusters)
                        df['Cluster'] = cluster_model.fit_predict(df[targets])
                else:    
                    print("Applying Agglomerative")
                    cluster_model = AgglomerativeClustering(n_clusters=num_clusters)
                    df['Cluster'] = cluster_model.fit_predict(df[targets])
            
            for cluster in range(num_clusters):
                print(f'Cluster {cluster}')
                if 'Cluster' in df.columns:
                    cluster_data = df[df['Cluster'] == cluster]
                    
                    if apply_dimensions_reduction:
                        if dimensions_reduction == 'pca':
                            fig.add_trace(go.Scatter(
                                x=cluster_data['pca-one'], 
                                y=cluster_data['pca-two'], 
                                mode='markers', 
                                name=f'Cluster {cluster}',
                                hovertext=f'Subject: {cluster_data["Subject"]}, Event: {cluster_data["Event"]}'
                            ))
                        elif dimensions_reduction == 'tsne':
                            fig.add_trace(go.Scatter(
                                x=cluster_data['tsne-2d-one'], 
                                y=cluster_data['tsne-2d-two'], 
                                mode='markers', 
                                name=f'Cluster {cluster}',
                                hovertext=f'Subject: {cluster_data["Subject"]}, Event: {cluster_data["Event"]}'
                            ))
                    else:

                        fig.add_trace(go.Scatter(
                            x=cluster_data[x_value], 
                            y=cluster_data[targets[0]], 
                            mode='markers', 
                            name=f'Cluster {cluster}',
                            hovertext=f'Subject: {cluster_data["Subject"]}, Event: {cluster_data["Event"]}'

                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=cluster_data[x_value], 
                        y=cluster_data[targets[0]], 
                        mode='markers', 
                        name=f'Cluster {cluster}',
                        hovertext=f'Subject: {cluster_data["Subject"]}, Event: {cluster_data["Event"]}'
                    ))

            
        else:
            for target in targets:
                fig.add_trace(go.Scatter(x=df[x_value], y=df[target], mode='markers', name=target, hovertext=f'{x_value}: {df[x_value]}'))

    fig.update_layout(barmode='stack' if stack else 'group')

    return dcc.Graph(figure=fig)
            
            
# Callback for downloading the data into the output folder of the project
@callback(
    Output({"type": "neurokit-download-button", "index": ALL}, "href"),
    Output({"type": "download-confirm-dialog", "index": ALL}, "displayed"),
    [Input({"type": "neurokit-download-button", "index": ALL}, "n_clicks")],
    [State("project-dropdown-neurokit", "value"),
     State({"type": "neurokit-results-ag-grid", "index": ALL}, "virtualRowData"),
     State({"type": "neurokit-file-name", "index": ALL}, "value"),
     State({"type": "neurokit-user-name", "index": ALL}, "value")],
    prevent_initial_call=True
)
def download_data(n_clicks, project, virtualRowData, file_name, user_name):
    if not n_clicks or not virtualRowData:
        raise PreventUpdate

    # if user_name[0] == '':
    #     user_name[0] = 'unknown'

    user_name = user_name[0]
    # Add date to the end of the user name
    user_name = f'{user_name}_{pd.Timestamp.now().strftime("%Y%m%d %H%M%S")}'

    if file_name[0] != '':
        file_name =  file_name[0]
    else:
        file_name = 'MindwareData'

    df = pd.DataFrame(virtualRowData[0])
    json_path = TEMPLATES_PATH
    project_json_path = os.path.join(json_path, f'{project}_template.json')
    with open(project_json_path, 'r') as file:
        template = json.load(file)
        file.close()
    output_folder = template["output"]["path"]
    if not output_folder:
        return ["Set an output folder in Project Settings before exporting."], [True]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    history_folder = os.path.join(output_folder, 'History', user_name)
    if not os.path.exists(history_folder):
        os.makedirs(history_folder)
    file_path = os.path.join(history_folder, f"{file_name}.csv")

    agg_output_folder = os.path.join(output_folder, f"{file_name}.csv")

    # Check if file_path already exists
    if os.path.exists(agg_output_folder):
        # replace the file in the aggregation folder with the new one and move the old one to the history folder
        shutil.move(agg_output_folder, file_path)

    df.to_csv(agg_output_folder, index=False)
        
        
    return [file_path], [True]
    


    
