

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
import webview
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
import queue
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import platform



dash.register_page(__name__, name="NeuroKit Values New", path="/neurokit_values_new", order = 3)



pages = {}

for page in os.listdir(r".\pages\templates"):
    page_name = page.split("_")[0]
    page_value = page.split("_")[0]

    pages[page_name] = page_value




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
            ],
        ),

        dcc.Store(id="script-status-store", data={"running": False, "finished": False}),
        dcc.Interval(id="script-status-interval", interval=1000, n_intervals=0),
        dcc.Store(id="neurokit-progress-store", data={}),
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
    fibro_hdf_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    fibro_hdf_file = os.path.join(fibro_hdf_file, f'{project}_mindware_results.parquet')

    results = []

    if os.path.exists(fibro_hdf_file):
        results = pl.read_parquet(fibro_hdf_file).to_pandas().to_dict('records')
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

# Callback for checking for updates
@callback(
    Output("neurokit-update-container", "children"),
    [Input("check-update-button-neurokit", "n_clicks")],
    State("project-dropdown-neurokit", "value"),
    prevent_initial_call=True
)
def check_for_update(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    csv_file = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\data\mindwareData'
    csv_file = os.path.join(csv_file, f'{project}_UpdatedData.csv')

    if not os.path.exists(csv_file):
        return "No updates available for Fibro project"

    data_df = pl.read_csv(csv_file)

    project_df = data_df.filter(pl.col('project') == project, pl.col('status') == 'New')


    if len(project_df) == 0:
        return "No updates available for Fibro project"

    elif len(project_df) > 0:    
        last_update = project_df['last_updated'].max()

        return f"New data available for Fibro project, last updated on {last_update}"
    
    else:
        raise PreventUpdate

# Callback for updating the data
@callback(
    Output("script-output-container", "children"),
    [Input("update-button-neurokit", "n_clicks")],
    State("project-dropdown-neurokit", "value"),
    prevent_initial_call=True
)
def update_data(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    try:
        if project == "fibro":
            # Define the command to run the script
            script_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\scripts\results_generating_fibro_update_polars.py'

        if platform.system() == "Windows":
            command = f'start cmd /c python "{script_path}"'
        else:
            command = f'python3 "{script_path}"'

        # Run the script in a new CMD window
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return "The update is running in the background, please wait for the window to close automatically."
    except Exception as e:
        return f"Error running update script: {str(e)}"
    




@callback(
    Output("script-output-container", "children", allow_duplicate=True),
    [Input("run-script-button", "n_clicks")],
    State("project-dropdown-neurokit", "value"),
    prevent_initial_call=True
)
def run_analysis_script(n_clicks, project):
    if n_clicks is None:
        raise PreventUpdate

    try:

        param = project
        # Define the command to run the script
        script_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\scripts\results_generating.py'
    
        if platform.system() == "Windows":
            command = f'start cmd /c python "{script_path}" {param}'
        else:
            command = f'python3 "{script_path}" {param}'  # Adjust this for non-Windows systems if needed

        # Run the script in a new CMD window
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        # Optionally wait for the process to complete and capture output
        # stdout, stderr = process.communicate()


        return "The analysis running in the background, please wait for the window to close automatically."

    except Exception as e:
        return f"Error running script: {str(e)}"
    


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
    json_path = r'C:\Users\PsyLab-6028\Desktop\RoeeProj\pages\templates'
    project_json_path = os.path.join(json_path, f'{project}_template.json')
    with open(project_json_path, 'r') as file:
        template = json.load(file)
        file.close()
    output_folder = template['output']['path']
    print(output_folder)
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
    


    
