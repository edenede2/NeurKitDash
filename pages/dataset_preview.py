from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import polars as pl
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objs as go
import dash
import sys
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pages.data_manager import get_data_manager

from pages.paths import project_names

dash.register_page(__name__, name="Dataset Preview", order=5)

pages = {name: name for name in project_names()}


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


def _subject_table(manager):
    subject_ids = manager.get_all_subjects()
    if os.path.isfile(manager.events_parquet_path):
        events = (pl.scan_parquet(manager.events_parquet_path)
                  .filter(pl.col("subject_id").is_in(subject_ids))
                  .select("subject_id", "event_name", "duration_minutes", "done")
                  .with_columns(pl.col("duration_minutes").cast(pl.Float64),
                                pl.col("done").cast(pl.Boolean)).collect())
    else:
        events = pl.DataFrame(schema={"subject_id": pl.String, "event_name": pl.String,
                                      "duration_minutes": pl.Float64, "done": pl.Boolean})

    missing = sorted(set(subject_ids) - set(events["subject_id"]))
    if missing:
        subjects = pl.DataFrame({"subject_id": missing}).with_columns(
            event_name=pl.lit("No events processed"),
            duration_minutes=pl.lit(None, dtype=pl.Float64),
            done=pl.lit(False),
        )
        events = pl.concat([events, subjects])
    return events


@callback(
    Output("preview-table-container", "children"),
    Input("submit-button", "n_clicks"),
    State("project-preview-dropdown", "value")
)
def update_table(n_clicks, project):
    if n_clicks is None:
        raise PreventUpdate

    try:
        manager = get_data_manager(project)
        data_df = _subject_table(manager)
        if data_df.is_empty():
            return html.H2("No subject data found. Import recordings first.")

        subjects_df = data_df.with_row_index("Index").to_pandas()
        total_subjects = subjects_df["subject_id"].nunique()

        columns_def = [
            {"headerName": "Index", "field": "Index", "checkboxSelection": True},
            {"headerName": "Subject ID", "field": "subject_id"},
            {"headerName": "Event Name", "field": "event_name"},
            {"headerName": "Duration (minutes)", "field": "duration_minutes"},
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
                                  id={"type": "remove-button", "index": 1},
                                  color="danger",
                                  style={"margin": "10px"})

        return html.Div([
            html.H2(f"Dataset Preview for {project.title()}"),
            html.P(f"Total unique subjects: {total_subjects}"),
            sub_table,
            remove_button
        ])

    except Exception as e:
        return html.Div([
            html.H2("Error loading data"),
            html.P(f"Error: {str(e)}")
        ], style={"color": "red"})


@callback(
    Output({"type": "sub-table", "index": MATCH}, "rowData"),
    Input({"type": "remove-button", "index": MATCH}, "n_clicks"),
    State({"type": "sub-table", "index": ALL}, "selectedRows"),
    State("project-preview-dropdown", "value")
)
def remove_subject(n_clicks, selected_rows, project):
    if n_clicks is None or not selected_rows[0]:
        raise PreventUpdate

    try:
        data_manager = get_data_manager(project)
        selected_rows = selected_rows[0]
        
        # Get unique subject IDs to remove
        selected_subjects = list(set([row['subject_id'] for row in selected_rows]))

        # Remove subjects using data manager
        removed_count = 0
        for subject_id in selected_subjects:
            if data_manager.remove_subject(subject_id):
                removed_count += 1

        print(f"Removed {removed_count} subjects")

        return _subject_table(data_manager).with_row_index("Index").to_dicts()
    except Exception:
        logger.exception("Could not remove subjects from project %s", project)
        return dash.no_update


@callback(
    Output("preview-pie-chart-container", "children"),
    Input("submit-button", "n_clicks"),
    State("project-preview-dropdown", "value")
)
def update_pie_chart(n_clicks, project):
    if n_clicks is None:
        raise PreventUpdate

    try:
        data_manager = get_data_manager(project)

        # Check for data in the same way as the main table
        data_df = None
        if os.path.exists(data_manager.events_parquet_path):
            try:
                data_df = pl.read_parquet(data_manager.events_parquet_path)
                if data_df.height == 0:
                    data_df = None
            except:
                data_df = None

        if data_df is None:
            return html.H3("No data available for pie chart")

        # Calculate completion statistics
        completion_stats = (
            data_df
            .group_by('done')
            .count()
            .to_pandas()
        )

        if completion_stats.empty:
            return html.H3("No data available for pie chart")

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Completed' if done else 'Pending' for done in completion_stats['done']],
            values=completion_stats['count'],
            hole=0.3
        )])

        fig.update_layout(
            title="Project Completion Status",
            annotations=[dict(text='Status', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return dcc.Graph(figure=fig)
        
    except Exception as e:
        return html.Div(f"Error creating pie chart: {str(e)}", style={"color": "red"})


@callback(
    Output("preview-details-container", "children"),
    Input("submit-button", "n_clicks"),
    State("project-preview-dropdown", "value")
)
def update_details(n_clicks, project):
    if n_clicks is None:
        raise PreventUpdate

    try:
        data_manager = get_data_manager(project)

        # Check for data in the same way as the main table
        data_df = None
        if os.path.exists(data_manager.events_parquet_path):
            try:
                data_df = pl.read_parquet(data_manager.events_parquet_path)
                if data_df.height == 0:
                    data_df = None
            except:
                data_df = None

        if data_df is None:
            return html.H3("No data available for details")

        # Calculate summary statistics
        total_subjects = data_df['subject_id'].n_unique()
        total_events = data_df.height
        completed_events = data_df.filter(pl.col('done') == True).height
        avg_duration = data_df['duration_minutes'].mean()

        # Calculate completion percentage safely
        completion_pct = (completed_events / total_events * 100) if total_events > 0 else 0

        # Event type distribution
        event_distribution = (
            data_df
            .group_by('event_name')
            .count()
            .sort('count', descending=True)
            .to_pandas()
        )

        details = html.Div([
            html.H3("Dataset Details"),
            html.P(f"Total Subjects: {total_subjects}"),
            html.P(f"Total Events: {total_events}"),
            html.P(f"Completed Events: {completed_events} ({completion_pct:.1f}%)"),
            html.P(f"Average Event Duration: {avg_duration:.2f} minutes" if avg_duration and avg_duration == avg_duration else "Average Event Duration: N/A"),
            html.Hr(),
            html.H4("Event Type Distribution"),
            dash_table.DataTable(
                data=event_distribution.to_dict('records'),
                columns=[
                    {"name": "Event Name", "id": "event_name"},
                    {"name": "Count", "id": "count"}
                ],
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ])

        return details

    except Exception as e:
        return html.Div(f"Error loading details: {str(e)}", style={"color": "red"})
