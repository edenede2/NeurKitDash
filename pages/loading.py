from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
import os
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import pickle as pkl
import h5py
import numpy as np
# import dash_core_components as dcc
from flask import Flask
import neurokit2 as nk
import base64
import json
import io
import re
import dash
import sys
import logging
from time import perf_counter

from pages.data_manager import DataManager, get_data_manager, process_raw_mindware_data
from pages.loading_logger import LoadingLogger
from pages.drive_import import subject_folders, find_recording, read_recording
from pages.paths import TEMPLATES_PATH, DATA_PATH, project_names

logger = logging.getLogger(__name__)

dash.register_page(__name__, name='Loading', order=2)

pages = {name: name for name in project_names()}


layout = html.Div(
    [
        html.Div("Subject Data Loading from Google Drive", style={"fontSize": 40, 'textAlign': 'center'}),
        html.Hr(),
        html.Div("In this page you can load the data of the subjects from the Google Drive folders. The system will automatically detect new subjects and process their data.", style={"fontSize": 20, 'textAlign': 'center'}),
        html.Hr(),
        html.Div("Select Project", style={"fontSize": 20, 'textAlign': 'center'}),

        dcc.Dropdown(
            id='project-dropdown-loading',
            options=[
                {'label': key, 'value': value} for key, value in pages.items()
            ],
            style={"color": "black"},
            value='fibro'
        ),

        html.Div(id='project-markdown-loading', style={"fontSize": 20, 'textAlign': 'center'}),
        
        # Radio button for overwriting the data or not
        html.Div(
            dcc.RadioItems(
                id='overwrite-data',
                options=[
                    {'label': 'Append data', 'value': 'append'},
                    {'label': 'Skip data', 'value': 'skip'}
                ],
                value='append',
                labelStyle={'display': 'inline-block'}
            )
        ),

        dbc.Container(
            [
            html.Div("Load Data from Google Drive Folders", style={"fontSize": 20, 'textAlign': 'center'}),
            
            html.Hr(),
            html.P("The system will automatically scan the configured Google Drive folders for new subject data and process it.",
                   style={"textAlign": "center", "fontSize": 16}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Preview Google Drive", id="preview-drive-button", color="info", size="lg",
                              style={"margin": "10px", "width": "250px"})
                ], width=6),
                dbc.Col([
                    dbc.Button("Scan and Load New Subjects", id="scan-drive-button", color="primary", size="lg",
                              style={"margin": "10px", "width": "250px"})
                ], width=6)
            ], justify="center"),

            # Google Drive Preview Section
            dcc.Loading(
                html.Div(id='drive-preview-container', children=[]),
                overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white"},
                custom_spinner=html.H2(["Scanning Google Drive...", dbc.Spinner(color="info")])
            ),

            # Detailed File Preview Modal
            dbc.Modal([
                dbc.ModalHeader("Subject File Details"),
                dbc.ModalBody(id="file-details-modal-body"),
                dbc.ModalFooter([
                    dbc.Button("Close", id="close-file-modal", className="ms-auto", n_clicks=0)
                ])
            ], id="file-details-modal", is_open=False, size="lg"),

            html.Hr(),

            html.Div("Migrate from HDF5 Format", style={"fontSize": 20, 'textAlign': 'center'}),
            html.P("If you have existing HDF5 data files, migrate them to the new Parquet/CSV format.",
                   style={"textAlign": "center", "fontSize": 16}),

            dbc.Button("Migrate HDF5 Data", id="migrate-hdf5-button", color="secondary", size="lg",
                      style={"margin": "20px", "width": "300px"}),

            dcc.Loading(
                [dcc.ConfirmDialog(
                id='confirm',
                message='',
                ),
                html.Div(id='loading-result', children="")],

                overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white", "filter": "blur(2px)"},
                custom_spinner=html.H2(["Processing subject data...", dbc.Spinner(color="primary")])
                )

            
            ]
        ),
        dbc.Container(
            [
                html.Div("Check for updates in the Google Drive folders", style={"fontSize": 20, 'textAlign': 'center'}),
                html.Hr(),
                dbc.Button("Check for updates", id="check-updates", color="primary", className="mr-1", style={"margin": "10px"}),
                dcc.Loading(
                    [dcc.ConfirmDialog(
                        id='update-confirm',
                        message='',
                    )],
                    overlay_style={"visibility": "visible", "opacity": 0.5, "background": "white", "filter": "blur(2px)"},
                    custom_spinner=html.H2(["Checking for updates...", dbc.Spinner(color="primary")])
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Subjects Found"),
                        dbc.ModalBody(
                            [
                                dcc.Checklist(id='subjects-checklist'),
                                dbc.Button("Load Selected Subjects", id='load-selected-subjects', color="primary", className="mr-1", style={"margin": "10px"}),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-modal", className="ml-auto")
                        ),
                    ],
                    id="modal",
                    is_open=False,
                ),
                dcc.ConfirmDialog(
                    id='no-new-subjects-dialog',
                    message='No new subjects found.',
                )
            ]
            
        ),

        dcc.Store(id='uploaded-data-store'),
        dcc.Store(id='uploaded-event-store'),
        dcc.Store(id='subjects-to-load-store'),
        dcc.Store(id='subjects-preview-store')  # Store for Google Drive preview data
    ]
)

@callback(
    [Output('drive-preview-container', 'children'),
     Output('subjects-preview-store', 'data')],
    Input('preview-drive-button', 'n_clicks'),
    State('project-dropdown-loading', 'value'),
    prevent_initial_call=True
)
def preview_google_drive(n_clicks, project):
    """Preview candidates without opening recordings or traversing loaded subjects."""
    if n_clicks is None or not project:
        raise PreventUpdate

    start_time = perf_counter()

    try:
        # Load project configuration
        template_path = os.path.join(TEMPLATES_PATH, f"{project}_template.json")
        with open(template_path, 'r') as f:
            project_config = json.load(f)
        
        input_path = project_config['input']['path']
        
        if not input_path:
            return dbc.Alert("No input path configured for this project.", color="warning"), {}

        if not os.path.exists(input_path):
            return dbc.Alert([
                html.H5("Google Drive Path Not Accessible", className="alert-heading"),
                html.P(f"Path: {input_path}"),
                html.Hr(),
                html.P("Please ensure Google Drive is synced and accessible.")
            ], color="danger"), {}

        existing_subjects = set(get_data_manager(project).get_all_subjects())
        folders = subject_folders(input_path)
        existing_folders = sorted(folders.keys() & existing_subjects)
        new_folders = sorted(folders.keys() - existing_subjects)
        total_folders = len(folders)
        subjects_info = []
        new_ready = 0

        for item in new_folders:
            try:
                data_folder_type, signal_path, events_path = find_recording(folders[item])
                txt_count = int(bool(signal_path)) + int(bool(events_path))
                if signal_path:
                    status, status_color = "new", "info"
                    new_ready += 1
                elif data_folder_type:
                    status, status_color = "empty_folder", "warning"
                else:
                    status, status_color = "no_data_folder", "secondary"

                subjects_info.append({
                    'subject_id': item,
                    'has_data_folder': data_folder_type is not None,
                    'data_folder_type': data_folder_type,
                    'status': status,
                    'status_color': status_color,
                    'is_existing': False,
                    'txt_files': txt_count
                })
            except Exception as e:
                subjects_info.append({
                    'subject_id': item,
                    'has_data_folder': False,
                    'status': 'error',
                    'status_color': 'danger',
                    'is_existing': False
                })

        # Add existing subjects (no detailed scan needed - they're processed)
        for item in existing_folders:
            subjects_info.append({
                'subject_id': item,
                'has_data_folder': True,
                'status': 'already_loaded',
                'status_color': 'success',
                'is_existing': True
            })

        elapsed = perf_counter() - start_time
        print(f"[FAST] Preview completed in {elapsed:.3f}s")

        # Sort: new ready first, then other new, then existing
        status_order = {'new': 0, 'incomplete': 1, 'empty_folder': 2, 'no_data_folder': 3, 'error': 4, 'already_loaded': 5}
        subjects_info.sort(key=lambda x: (status_order.get(x['status'], 99), x['subject_id']))

        print(f"[INFO] Valid subjects ready to process: {new_ready}")

        # Create summary cards - optimized counts
        new_with_data = len([s for s in subjects_info if s.get('status') == 'new'])
        mindware_count = len([s for s in subjects_info if s.get('data_folder_type') == 'MINDWARE'])
        hrv_count = len([s for s in subjects_info if s.get('data_folder_type') == 'HRV'])
        already_loaded_count = len([s for s in subjects_info if s.get('status') == 'already_loaded'])

        # Summary cards - focused on NEW subjects prominently
        summary_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(new_with_data), className="card-title text-primary", style={'fontWeight': 'bold'}),
                        html.P("🆕 New Ready to Load", className="card-text")
                    ])
                ], color="primary", outline=True, style={'border-width': '2px'})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(len(new_folders)), className="card-title"),
                        html.P([
                            "New in Google Drive",
                            html.Br(),
                            html.Small(f"MW: {mindware_count}, HRV: {hrv_count}", className="text-muted")
                        ], className="card-text")
                    ])
                ], color="info", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(already_loaded_count), className="card-title text-success"),
                        html.P("✓ Already Loaded", className="card-text")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(total_folders), className="card-title"),
                        html.P("Total Folders", className="card-text")
                    ])
                ], color="light", outline=True)
            ], width=3)
        ], className="mb-4")

        # Create simplified table - focused on NEW subjects
        if subjects_info:
            table_rows = []
            # Show new subjects first (up to 30), then a few existing
            new_subjects_list = [s for s in subjects_info if not s.get('is_existing')][:30]
            existing_sample = [s for s in subjects_info if s.get('is_existing')][:5]
            display_subjects = new_subjects_list + existing_sample

            for i, subject in enumerate(display_subjects):
                status_badge = dbc.Badge(
                    subject['status'].title().replace('_', ' '),
                    color=subject['status_color'],
                    className="me-1"
                )
                
                # Data folder info
                folder_type = subject.get('data_folder_type', '-')
                folder_display = f"📁 {folder_type}" if folder_type and folder_type != '-' else "❌"
                
                # TXT files count (for new subjects)
                txt_info = str(subject.get('txt_files', '-')) if not subject.get('is_existing') else "✓"
                
                table_rows.append(html.Tr([
                    html.Td(subject['subject_id'], style={'fontWeight': 'bold'} if subject['status'] == 'new' else {}),
                    html.Td(status_badge),
                    html.Td(folder_display),
                    html.Td(txt_info),
                    html.Td(
                        dbc.Button("📋 Details",
                                 id={"type": "view-files-btn", "index": i},
                                 color="outline-secondary",
                                 size="sm") if not subject.get('is_existing') else "-"
                    )
                ], style={'backgroundColor': '#e7f3ff'} if subject['status'] == 'new' else {}))

            subjects_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Subject ID"),
                        html.Th("Status"),
                        html.Th("Data Folder"),
                        html.Th("TXT Files"),
                        html.Th("Actions")
                    ])
                ]),
                html.Tbody(table_rows)
            ], striped=True, bordered=True, hover=True, responsive=True, size="sm")

            # Table note
            if len(new_folders) > 30:
                table_note = html.P(f"Showing {len(new_subjects_list)} of {len(new_folders)} new subjects + {len(existing_sample)} existing.",
                                  className="text-muted small")
            else:
                table_note = html.P(f"Scanned in {elapsed:.2f}s", className="text-muted small")
        else:
            subjects_table = dbc.Alert("No subject folders found in the Google Drive path.", color="warning")
            table_note = ""

        # Simplified legend
        legend = dbc.Card([
            dbc.CardHeader("Status Legend"),
            dbc.CardBody([
                dbc.Badge("New", color="info", className="me-2"), " Ready to load (not in system yet)",
                html.Br(),
                dbc.Badge("Already Loaded", color="success", className="me-2"), " Already processed",
                html.Br(),
                dbc.Badge("Incomplete", color="warning", className="me-2"), " Missing required files",
                html.Br(),
                dbc.Badge("No Data Folder", color="secondary", className="me-2"), " No MINDWARE/HRV folder",
                html.Hr(className="my-2"),
                html.Small("Expected: subject_folder/MINDWARE/*.txt or subject_folder/HRV/*.txt", className="text-muted")
            ])
        ], className="mb-3")

        # Build the return content
        content_children = [
            html.H4(f"Google Drive Preview - {project.title()} Project", className="mb-3"),
            html.P(f"📁 Path: {input_path}", className="text-muted small"),
            summary_cards,
            legend,
            html.H5("📋 Subject Details (New subjects first)", className="mb-3"),
            subjects_table,
            table_note
        ]

        return html.Div(content_children), {'subjects_info': subjects_info, 'project': project, 'input_path': input_path}

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_alert = dbc.Alert([
            html.H5("Error Scanning Google Drive", className="alert-heading"),
            html.P(f"Error: {str(e)}"),
            html.Hr(),
            html.P("Please check the console for more details.")
        ], color="danger")
        return error_alert, {}


@callback(
    [Output('file-details-modal', 'is_open'),
     Output('file-details-modal-body', 'children')],
    [Input({"type": "view-files-btn", "index": ALL}, 'n_clicks'),
     Input('close-file-modal', 'n_clicks')],
    [State('file-details-modal', 'is_open'),
     State('subjects-preview-store', 'data')],
    prevent_initial_call=True
)
def toggle_file_modal(view_clicks, close_click, is_open, preview_data):
    """Show the export files selected by the same rules as the import."""
    trigger = dash.ctx.triggered_id
    if trigger == "close-file-modal":
        return False, ""
    if not trigger or not any(view_clicks) or not preview_data:
        raise PreventUpdate

    subjects = preview_data["subjects_info"]
    subject = subjects[trigger["index"]]["subject_id"]
    folder = os.path.join(preview_data["input_path"], subject)
    try:
        kind, signal_path, markers_path = find_recording(folder)
        if not signal_path:
            return True, dbc.Alert(f"No signal export in {folder}", color="warning")

        frame, parser = read_recording(signal_path, nrows=5)
        return True, html.Div([
            html.H5(f"{subject} ({kind})"),
            html.P(f"Signal file: {signal_path}"),
            html.P(f"Event file: {markers_path}" if markers_path else "No event file found."),
            html.Small(f"First {len(frame)} samples ({parser})"),
            dbc.Table.from_dataframe(frame, striped=True, responsive=True, size="sm"),
        ])
    except (OSError, ValueError):
        logger.exception("Could not read files for subject %s", subject)
        return True, dbc.Alert(f"Could not preview files in {folder}. See the app log.", color="warning")



@callback(
    [Output('loading-result', 'children'),
     Output('confirm', 'message'),
     Output('confirm', 'displayed')],
    [Input('scan-drive-button', 'n_clicks'),
     Input('migrate-hdf5-button', 'n_clicks')],
    [State('project-dropdown-loading', 'value')],
    prevent_initial_call=True
)
def handle_data_operations(scan_clicks, migrate_clicks, project):
    """Handle scanning for new subjects and HDF5 migration"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if button_id == 'scan-drive-button':
            # Scan Google Drive and load new subjects
            result = scan_and_load_subjects(project)
            return result, "", False

        elif button_id == 'migrate-hdf5-button':
            # Migrate HDF5 data
            result = migrate_hdf5_data(project)
            return result, "", False

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return "", error_msg, True

    raise PreventUpdate


def _stage_recording(data_manager, project_config, subject, folder, report):
    discovery_start = perf_counter()
    folder_type, signal_path, marker_path = find_recording(folder)
    report.logger.debug("Discovery for %s in %.3fs", subject, perf_counter() - discovery_start)
    if not signal_path:
        return None

    report.log_file_discovery("signal", signal_path)
    if marker_path:
        report.log_file_discovery("events", marker_path)

    start = perf_counter()
    signal_frame, parser = read_recording(signal_path)
    marker_frame = None
    if marker_path:
        marker_frame = pd.read_csv(marker_path, sep="\t")
    report.log_processing_step("Read TSV", True, {
        "format": parser, "samples": len(signal_frame), "seconds": round(perf_counter() - start, 3)
    })

    start = perf_counter()
    signals, events, attributes = process_raw_mindware_data(
        signal_frame, marker_frame, project_config, subject
    )
    if not len(signals.get("signal", ())) and not len(signals.get("gsc", ())):
        raise ValueError(f"No configured ECG or GSC channel in {os.path.basename(signal_path)}")
    report.log_processing_step("Process channels and events", True, {
        "folder": folder_type, "seconds": round(perf_counter() - start, 3), "events": len(events)
    })
    for channel, field in (("ECG", "signal"), ("GSC/EDA", "gsc"), ("Motion", "motion")):
        samples = len(signals.get(field, ()))
        report.log_signal_detection(channel, bool(samples), {"samples": samples})
    report.log_data_quality({"rpeaks_count": len(attributes.get("rpeaks", ())), "events_count": len(events)})

    start = perf_counter()
    records = data_manager.stage_subject(subject, signals, events, attributes)
    report.log_processing_step("Stage Parquet and metadata", True, {
        "seconds": round(perf_counter() - start, 3)
    })
    return records


def _commit_staged_subjects(manager, pending, report):
    if not pending:
        return
    start = perf_counter()
    try:
        completed = manager.commit_subjects(pending)
    except Exception as error:
        report.log_error("Could not publish the batch. Run the import again to retry.", error)
        completed = set()
    for subject in pending:
        if subject in completed:
            report.finish_subject_processing("success", subject_id=subject)
        else:
            report.finish_subject_processing("failed", "Could not publish data. Retry the import.", subject_id=subject)
    report.logger.info("Published %d of %d subjects in %.3fs",
                       len(completed), len(pending), perf_counter() - start)
    pending.clear()


def scan_and_load_subjects(project, selected_subjects=None):
    """Import only new subject folders and publish metadata in batches."""
    if project not in project_names():
        return html.Div("Select a project in Loading.", style={"color": "red"})
    report = LoadingLogger(project)
    started = perf_counter()
    try:
        with open(TEMPLATES_PATH / f"{project}_template.json", encoding="utf-8") as file:
            config = json.load(file)
        input_path = config["input"]["path"]
        if not input_path or not os.path.isdir(input_path):
            return html.Div("Set an accessible input folder in Project Settings.", style={"color": "red"})
        manager = get_data_manager(project)
        if not manager.write_lock.acquire(blocking=False):
            return html.Div("A data operation is already running for this project. Try again when it finishes.")
        try:
            folders = subject_folders(input_path)
            if selected_subjects is not None:
                folders = {name: folders[name] for name in selected_subjects if name in folders}
            existing = folders.keys() & set(manager.get_all_subjects())
            candidates = sorted(folders.keys() - existing)
            report.logger.info("Scanned %d folders in %.3fs (%d already loaded, %d candidates)",
                               len(folders), perf_counter() - started, len(existing), len(candidates))
            pending = {}
            for subject in candidates:
                report.start_subject_processing(subject, folders[subject])
                try:
                    records = _stage_recording(manager, config, subject, folders[subject], report)
                    if records is None:
                        report.finish_subject_processing("skipped", "No signal export found")
                    else:
                        pending[subject] = records
                        report.current_subject = None
                except Exception as error:
                    report.log_error("Could not import subject", error)
                    report.finish_subject_processing("failed", str(error))
                if len(pending) >= 16:
                    _commit_staged_subjects(manager, pending, report)
            _commit_staged_subjects(manager, pending, report)
        finally:
            manager.write_lock.release()

        stats = report.session_stats
        failed = len(stats["subjects_failed"])
        report_path = report.save_session_report()
        elapsed = perf_counter() - started
        report.logger.info("Import finished in %.3fs", elapsed)
        return html.Div([
            html.P(f"Loaded {len(stats['subjects_processed'])}, already loaded {len(existing)}, "
                   f"skipped {len(stats['subjects_skipped'])}, failed {failed} ({elapsed:.1f}s)."),
            html.Small(f"Report: {report_path}"),
        ], style={"color": "darkorange" if failed else "green"})
    except Exception as error:
        report.log_error("Import failed", error)
        return html.Div(f"Import failed: {error}. See the loading log in {report.log_dir}.",
                        style={"color": "red"})
    finally:
        report.close()


def migrate_hdf5_data(project):
    """Migrate existing HDF5 data to Parquet/CSV format"""
    try:
        data_manager = get_data_manager(project)

        # Look for HDF5 files to migrate
        data_path = DATA_PATH
        hdf5_file = os.path.join(data_path, f"{project}_mindware.hdf5")

        if not os.path.exists(hdf5_file):
            return html.Div(f"No HDF5 file found for project {project}",
                          style={"color": "orange", "fontSize": 18})

        # Perform migration
        success = data_manager.migrate_from_hdf5(hdf5_file)

        if success:
            migrated_subjects = data_manager.get_all_subjects()
            result_text = f"Migration completed successfully! Migrated {len(migrated_subjects)} subjects."
            color = "green"
        else:
            result_text = "Migration failed. Please check the logs for details."
            color = "red"

        return html.Div(result_text, style={"color": color, "fontSize": 18, "textAlign": "center"})

    except Exception as e:
        return html.Div(f"Error during migration: {str(e)}",
                       style={"color": "red", "fontSize": 18})

@callback(
    Output('project-markdown-loading', 'children'),
    Input('project-dropdown-loading', 'value'),
    prevent_initial_call=True
)
def display_project_info(project):
    """Display project configuration info"""
    if not project:
        return ""

    try:
        # Load project configuration
        template_path = os.path.join(TEMPLATES_PATH, f"{project}_template.json")
        with open(template_path, 'r') as f:
            project_config = json.load(f)

        # Get data manager to check existing subjects
        data_manager = get_data_manager(project)
        existing_subjects = data_manager.get_all_subjects()

        input_path = project_config.get('input', {}).get('path', 'Not configured')
        sampling_rate = project_config.get('parameters', {}).get('sampling_rate', 500)
        channels = project_config.get('channel', [])

        info = html.Div([
            html.H4(f"Project: {project.title()}"),
            html.P(f"Input Path: {input_path}"),
            html.P(f"Sampling Rate: {sampling_rate} Hz"),
            html.P(f"Channels: {', '.join(channels)}"),
            html.P(f"ExistingSubjects: {len(existing_subjects)}"),
            html.Hr(),
        ])

        return info

    except Exception as e:
        return html.Div(f"Error loading project info: {str(e)}",
                       style={"color": "red"})

@callback(
    Output('subjects-checklist', 'options'),
    Output('modal', 'is_open'),
    Output('no-new-subjects-dialog', 'displayed'),
    Input('check-updates', 'n_clicks'),
    State('project-dropdown-loading', 'value'),
    prevent_initial_call=True
)
def check_google_drive(n_clicks, project):
    if n_clicks is None:
        raise PreventUpdate
    
    # Function to check Google Drive folder for new subjects
    new_subjects = check_new_subjects_in_drive(project)

    print(f'New subjects: {new_subjects}')
    
    if not new_subjects:
        return [], False, True
    
    subject_options = [{'label': subject, 'value': subject} for subject in new_subjects]
    return subject_options, True, False





def check_new_subjects_in_drive(project):
    try:
        with open(TEMPLATES_PATH / f"{project}_template.json", encoding="utf-8") as file:
            input_path = json.load(file)["input"]["path"]
        if not input_path:
            return []
        existing = set(get_data_manager(project).get_all_subjects())
        return sorted(name for name, path in subject_folders(input_path).items()
                      if name not in existing and find_recording(path)[1])
    except (OSError, ValueError, KeyError):
        logger.exception("Could not check the input folder for project %s", project)
        return []


@callback(
    Output('update-confirm', 'message'),
    Output('update-confirm', 'displayed'),
    Input('load-selected-subjects', 'n_clicks'),
    State('subjects-checklist', 'value'),
    State('project-dropdown-loading', 'value'),
    prevent_initial_call=True
)
def load_selected_subjects(n_clicks, selected_subjects, project):
    if n_clicks is None:
        raise PreventUpdate

    if not selected_subjects:
        return "No subjects selected for loading.", True
    
    # Process the selected subjects
    message = process_selected_subjects(selected_subjects, project)
    return message, True

def process_selected_subjects(selected_subjects, project):
    result = scan_and_load_subjects(project, selected_subjects)
    parts = result.children
    return " ".join(part.children for part in parts) if isinstance(parts, list) else parts


def detect_events(event_list, events_df, new_events_data, skip_indices, blocks_shorts=None):
    for event_key, event in event_list:
        start_flag = event['start']
        end_flag = event['end']
        classification = event['name']
        events_df['Event Type'] = [e.split(":")[1] if ":" in e else e for e in events_df['Event Type']]
        if blocks_shorts is not None:
            events_df['Name'] = [blocks_shorts[e] if e in blocks_shorts.keys() else e for e in events_df['Name']]
            events_df['Event Type'] = [events_df.loc[idx, 'Name'] if re.search('UDP', e) else e for idx, e in enumerate(events_df['Event Type'])]
        start_indices = events_df[(events_df['Event Type'] == start_flag) & (~events_df.index.isin(skip_indices))].index.tolist()
        end_indices = events_df[(events_df['Event Type'] == end_flag) & (~events_df.index.isin(skip_indices))].index.tolist()
        start_indices.sort()
        end_indices.sort()
        for start_idx in start_indices:
            for end_idx in end_indices:
                if start_idx < end_idx:
                    start_time = events_df.loc[start_idx, 'Time']
                    end_time = events_df.loc[end_idx, 'Time']
                    if not any((e['Start_Flag'] == start_flag and e['End_Flag'] == end_flag and
                                e['Start_Time'] == start_time and e['End_Time'] == end_time) for e in new_events_data):
                        new_events_data.append({
                            'Classification': classification,
                            'Start_Flag': start_flag,
                            'End_Flag': end_flag,
                            'Start_Time': start_time,
                            'End_Time': end_time
                        })
                    break
        
def mark_special_events(events_df, events_mapping, special_type):
    special_events = [event for event in events_mapping.items() if event[1].get(special_type)]
    special_flags = set()
    for _, event in special_events:
        start_flag = event['start']
        end_flag = event['end']
        special_flags.add((start_flag, end_flag))
    marked_indices = []
    flagged_events_count = {}
    for start_flag, end_flag in special_flags:
        start_indices = events_df[events_df['Event Type'] == start_flag].index.tolist()
        end_indices = events_df[events_df['Event Type'] == end_flag].index.tolist()
        start_indices.sort()
        end_indices.sort()
        pairs_count = min(len(start_indices), len(end_indices))
        flagged_events_count[(start_flag, end_flag)] = pairs_count
        for start_idx in start_indices:
            for end_idx in end_indices:
                if start_idx < end_idx:
                    marked_indices.append((start_idx, end_idx))
                    break
    adjusted_indices = []
    for (start_flag, end_flag), count in flagged_events_count.items():
        mapping_count = sum([1 for event in events_mapping.items() if event[1].get(special_type) and event[1]['start'] == start_flag and event[1]['end'] == end_flag])
        regular_event_count = count - mapping_count
        if regular_event_count < 0:
            return []
        if regular_event_count > 0:
            events_mapping_order = [event[0] for event in events_mapping.items() if event[1]['start'] == start_flag and event[1]['end'] == end_flag]
            events_mapping_order = [event.split("_")[1] for event in events_mapping_order]
            regular_event_number = [event[0] for event in events_mapping.items() if event[1]['start'] == start_flag and event[1]['end'] == end_flag and not event[1].get(special_type)]
            regular_event_number = [event.split("_")[1] for event in regular_event_number]
            events_mask = [1 if event in regular_event_number else 0 for event in events_mapping_order]
            for idx, (start_idx, end_idx) in enumerate(marked_indices):
                if events_mask[idx] == 0:
                    adjusted_indices.append(start_idx)
                    adjusted_indices.append(end_idx)
    return adjusted_indices

def save_to_hdf5(fibro_file, subject, data_df, events_df, project):

    templates_path = TEMPLATES_PATH
    project_template = f'{project}_template.json'
    with open(os.path.join(templates_path, project_template)) as f:
        json_data = json.load(f)

        events = json_data['events']
        sampling_rate = json_data['parameters']['sampling_rate']
        channels = json_data['channel']
        if json_data.get('blocks_shorts'):
            blocks_shorts = json_data['blocks_shorts']
        else:
            blocks_shorts = None

        f.close()

    signal = data_df.filter(regex='Bio').values.flatten()
    time = data_df['Time (s)'].values.flatten()

    x_axis = data_df.filter(regex='X Axis').values.flatten()
    y_axis = data_df.filter(regex='Y Axis').values.flatten()

    try:
        z_axis = data_df.filter(regex='Z Axis').values.flatten()
        x_axis = [x**2 for x in x_axis]
        y_axis = [y**2 for y in y_axis]
        z_axis = [z**2 for z in z_axis]
        motion = np.sqrt(x_axis + y_axis + z_axis)
        motion = [m * 0.01 for m in motion]
        if len(motion) / len(time) == 3:
            motion = motion[::3]
        elif len(motion) / len(time) == 2:
            motion = motion[::2]


        
        print(f"Motion data loaded successfully")
    except KeyError:
        x_axis = [x**2 for x in x_axis]
        y_axis = [y**2 for y in y_axis]

        motion = np.sqrt(x_axis + y_axis)   
        motion = [m * 0.01 for m in motion]
        motion = motion[::2]
        print(f"Motion data loaded successfully")

    events_df['Event Type'] = [x.split(':')[1] for x in events_df['Event Type']]
    binah_marked_indices = mark_special_events(events_df, events, 'Binah')
    korro_marked_indices = mark_special_events(events_df, events, 'Korro')

    new_events_data = []

    non_special_events = [event for event in events.items() if not event[1].get('Binah') and not event[1].get('Korro')]
    if binah_marked_indices or korro_marked_indices:
        detect_events(non_special_events, events_df, new_events_data, binah_marked_indices + korro_marked_indices, blocks_shorts)

        binah_events = [event for event in events.items() if event[1].get('Binah')]
        korro_events = [event for event in events.items() if event[1].get('Korro')]
        detect_events(binah_events, events_df, new_events_data, [], blocks_shorts)
        detect_events(korro_events, events_df, new_events_data, [], blocks_shorts)
    else:
        detect_events(events.items(), events_df, new_events_data, [], blocks_shorts)

    new_events_df = pd.DataFrame(new_events_data)
    events_data = new_events_df.to_dict(orient='records')

    events_columns = [{'name': i, 'id': i, 'editable': True} for i in events_df.columns]

    if signal.size == 0:
        print(f"No ecg signal found for subject {subject}")

        events_df['avg_motion'] = np.nan

        events_data = {event['Classification']: event for event in events_data}

        for key, value in events_data.items():
            if 'Start_Time' not in value or 'End_Time' not in value:
                continue
            start_time = value['Start_Time']
            end_time = value['End_Time']

            time = np.array(time)

            differences_start_time = np.abs(time - start_time)
            min_index_start = np.argmin(differences_start_time)
            closest_start_time = time[min_index_start]

            start_index = np.where(time == closest_start_time)[0][0]

            differences_end_time = np.abs(time - end_time)
            min_index_end = np.argmin(differences_end_time)
            closest_end_time = time[min_index_end]

            end_index = np.where(time == closest_end_time)[0][0]

            average_motion = round(np.mean(motion[start_index:end_index]), 2)
            events_data[key]['avg_motion'] = average_motion


    else:
        cleaned_signal = nk.ecg_clean(signal, sampling_rate=500)

        info, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=500, correct_artifacts=True)
        rpeaks_indices = rpeaks['ECG_R_Peaks'].tolist()

        ectopic_beats = rpeaks['ECG_fixpeaks_ectopic']
        longshort_beats = rpeaks['ECG_fixpeaks_longshort']
        false_negatives = rpeaks['ECG_fixpeaks_missed']
        false_positives = rpeaks['ECG_fixpeaks_extra']

        quality = nk.ecg_quality(cleaned_signal, sampling_rate=500)

        quality_rpeaks = quality[rpeaks_indices]

        peaks_color = ['yellow' if quality_rpeak<0.6 else 'red' for quality_rpeak in quality_rpeaks]

        rpeaks_indices_bad_quality = [rpeaks_indices[i] for i in range(len(quality_rpeaks)) if quality_rpeaks[i] < 0.6]

        peaks_color_binary = [1 if quality_rpeak<0.6 else 0 for quality_rpeak in quality_rpeaks]

        columns_to_add = ['avg_quality', 'avg_motion', 'corrected_rpeaks_percent']

        for column in columns_to_add:
            events_df[column] = np.nan

        events_data = {event['Classification']: event for event in events_data}

        for key, value in events_data.items():

            if 'Start_Time' not in value or 'End_Time' not in value:
                continue
            start_time = value['Start_Time']
            end_time = value['End_Time']

            time = np.array(time)

            differences_start_time = np.abs(time - start_time)
            min_index_start = np.argmin(differences_start_time)
            closest_start_time = time[min_index_start]

            start_index = np.where(time == closest_start_time)[0][0]

            differences_end_time = np.abs(time - end_time)
            min_index_end = np.argmin(differences_end_time)
            closest_end_time = time[min_index_end]

            end_index = np.where(time == closest_end_time)[0][0]

            rpeaks_time = [time[rpeak] for rpeak in rpeaks_indices]

            rpeaks_time = np.array(rpeaks_time)

            differences_start_time_rpeaks = np.abs(rpeaks_time - start_time)
            min_index_start_rpeaks = np.argmin(differences_start_time_rpeaks)
            closest_start_time_rpeaks = rpeaks_time[min_index_start_rpeaks]

            start_index_rpeaks = np.where(rpeaks_time == closest_start_time_rpeaks)[0][0]

            differences_end_time_rpeaks = np.abs(rpeaks_time - end_time)
            min_index_end_rpeaks = np.argmin(differences_end_time_rpeaks)
            closest_end_time_rpeaks = rpeaks_time[min_index_end_rpeaks]

            end_index_rpeaks = np.where(rpeaks_time == closest_end_time_rpeaks)[0][0]

            quality_value = round(np.mean(quality_rpeaks[start_index_rpeaks:end_index_rpeaks]), 2)

            average_motion = round(np.mean(motion[start_index:end_index]), 2)

            rpeaks_indices_range = [start_index_rpeaks, end_index_rpeaks]

            ectopic_beats_value = len([peak for peak in ectopic_beats if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
            longshort_beats_value = len([peak for peak in longshort_beats if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
            false_negatives_value = len([peak for peak in false_negatives if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])
            false_positives_value = len([peak for peak in false_positives if peak<=rpeaks_indices_range[1] and peak>=rpeaks_indices_range[0]])

            artifacts = len([peak for peak in peaks_color_binary[start_index_rpeaks:end_index_rpeaks] if peak == 1])

            if artifacts > 0 and len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks]) != 0:
                artifacts_percent = round((artifacts / len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks])) * 100, 2)
            else:
                artifacts_percent = 0.00

            corrected_rpeaks = ectopic_beats_value + longshort_beats_value + false_negatives_value + false_positives_value

            if corrected_rpeaks > 0:
                print(f"!!!!!!!!!!!!!!!!!!!!!Currected rpeaks: {corrected_rpeaks}!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks]) != 0:
                corrected_rpeaks_percent = round((corrected_rpeaks / len(rpeaks_indices[start_index_rpeaks:end_index_rpeaks])) * 100, 2)
            else:
                corrected_rpeaks_percent = 0

            events_data[key]['avg_quality'] = quality_value
            events_data[key]['artifacts_percent'] = artifacts_percent
            events_data[key]['avg_motion'] = average_motion
            events_data[key]['corrected_rpeaks_percent'] = corrected_rpeaks_percent

    if 'GSC' in json_data['channel'] and data_df.filter(regex='GSC').values.any():
        gsc = data_df.filter(regex='GSC').values.flatten()
        
        gsc_cleaned = nk.eda_clean(gsc, sampling_rate=500)
        print(f"GSC cleaned successfully")

        gsc_phasic = nk.eda_phasic(gsc_cleaned, sampling_rate=500, method='highpass')
        print(f"GSC phasic calculated successfully")

        _, gsc_peaks = nk.eda_peaks(gsc_cleaned, sampling_rate=500)
        print(f"GSC peaks calculated successfully")

        if signal.size != 0:


            stored_signal = {
                'signal': signal,
                'gsc': gsc,
                'eda_tonic': gsc_phasic['EDA_Tonic'].to_list(),
                'eda_phasic': gsc_phasic['EDA_Phasic'].to_list(),
                'time': time,
                'motion': motion
            }

            signal_attributes = {
                'rpeaks': rpeaks_indices,
                'yellow_peaks': rpeaks_indices_bad_quality,
                'ectopic_beats': ectopic_beats,
                'longshort_beats': longshort_beats,
                'false_negatives': false_negatives,
                'false_positives': false_positives
            }

            gsc_attributes = {
                'SCR_Onsets': gsc_peaks['SCR_Onsets'].tolist(),
                'SCR_Peaks': gsc_peaks['SCR_Peaks'].tolist(),
                'SCR_Height': gsc_peaks['SCR_Height'].tolist(),
                'SCR_Amplitude': gsc_peaks['SCR_Amplitude'].tolist(),
                'SCR_RiseTime': gsc_peaks['SCR_RiseTime'].tolist(),
                'SCR_Recovery': gsc_peaks['SCR_Recovery'].tolist(),
                'SCR_RecoveryTime': gsc_peaks['SCR_RecoveryTime'].tolist()
            }

            # signal_attributes.update(gsc_attributes)

        else:

            stored_signal = {
                'gsc': gsc,
                'eda_tonic': gsc_phasic['EDA_Tonic'].to_list(),
                'eda_phasic': gsc_phasic['EDA_Phasic'].to_list(),
                'time': time,
                'motion': motion
            }

            gsc_attributes = {
                'SCR_Onsets': gsc_peaks['SCR_Onsets'].tolist(),
                'SCR_Peaks': gsc_peaks['SCR_Peaks'].tolist(),
                'SCR_Height': gsc_peaks['SCR_Height'].tolist(),
                'SCR_Amplitude': gsc_peaks['SCR_Amplitude'].tolist(),
                'SCR_RiseTime': gsc_peaks['SCR_RiseTime'].tolist(),
                'SCR_Recovery': gsc_peaks['SCR_Recovery'].tolist(),
                'SCR_RecoveryTime': gsc_peaks['SCR_RecoveryTime'].tolist()
            }

            signal_attributes = {
            }
    else:
        stored_signal = {
            'signal': signal,
            'time': time,
            'motion': motion
        }

        gsc_attributes = {
        }

        signal_attributes = {
            'rpeaks': rpeaks_indices,
            'yellow_peaks': rpeaks_indices_bad_quality,
            'ectopic_beats': ectopic_beats,
            'longshort_beats': longshort_beats,
            'false_negatives': false_negatives,
            'false_positives': false_positives
        }
    for key, value in signal_attributes.items():
        print(f"Key: {key}, Length: {len(value)}, Type: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"Key: {key}, Shape: {value.shape}")

    for key, value in stored_signal.items():
        print(f"Key: {key}, Length: {len(value)}, Type: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"Key: {key}, Shape: {value.shape}")

    signal_dataset = pd.DataFrame(stored_signal)
    subject_folder_name = '/' + subject + '/data'
    subject_group = f'{subject}/data'
    rpeaks_group = f'{subject}/rpeaks'

    file_name = f'pages/data/mindwareData/{project}_mindware.hdf5'

    with pd.HDFStore(file_name, mode='a') as store:
        # Check if the subject group already exists in the HDF5 file
        if subject_folder_name in store.keys():
            del store[subject]
            print(f"Subject group {subject_group} removed successfully")
            store.close()
        else:
            print(f"Subject group {subject_group} does not exist in the HDF5 file")
            store.close()

    signal_dataset.to_hdf(file_name, key=subject_group, mode='a', format='fixed', complevel=9, complib='blosc')

    if signal_attributes != {}:
        rpeaks_dataset = pd.DataFrame(rpeaks_indices)

        rpeaks_dataset.to_hdf(file_name, key=rpeaks_group, mode='a', format='fixed', complevel=9, complib='blosc')

    with pd.HDFStore(file_name, mode='a') as store:

        data_group = store[f'{subject}/data']

        if signal_attributes != {}:
            signal_attributes.pop('rpeaks')
            store.get_storer(f'{rpeaks_group}').attrs.signal_attributes = json.dumps(signal_attributes)

        if gsc_attributes != {}:
            store.get_storer(f'{subject_group}').gsc_attributes = json.dumps(gsc_attributes)

        store.get_storer(f'{subject_group}').attrs.events_data = json.dumps(events_data)

        store.close()

    finish_message, df = save_events_data(file_name, subject, project)

    return finish_message, df
    


def save_parquet_file(file_name, selected_subjects, project, events_df):
    
    events_df.to_parquet(file_name.replace('.hdf5', '.parquet'), index=False)
    print(f"events_data dataset saved successfully in {file_name}")
        

def save_events_data(file_name, subject, project, mode='append'):
    


    # if 'events_data' not in fibro_hdf.keys():
    #     if file_name.replace('.hdf5', '.parquet') in os.listdir('pages/data'):
    #         events_df = pd.read_parquet(file_name.replace('.hdf5', '.parquet'))

    #     else:
    #         print(f"Creating events_data dataset in {file_name}")
    #         events_df = pd.DataFrame(columns=['Id', 'event_name', 'done', 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality', 'duration(minutes)'])
    #         first_row = {'Id': 'Id', 'event_name': 'event_name', 'done': False, 'artifacts_percentage_%': 0.0, 'corrected_rpeaks_%': 0.0, 'avg_motion': 0.0, 'avg_quality': 0.0, 'duration(minutes)': 0.0}
    #         events_df = pd.concat([events_df, pd.DataFrame([first_row])])
    #         fibro_hdf.close()
    #         print(f"events_data dataset created successfully in {file_name}")
    #         # done_bool = events_df['done']
    #         # done_str = events_df['done'].astype(str)
    #         # events_df['done'] = done_str
    #         # events_df.to_hdf(file_name, key='events_data', mode='a', format='table')
    #         # events_df_parquet = events_df.copy()
    #         # events_df_parquet['done'] = done_bool
    #         events_df.to_parquet(file_name.replace('.hdf5', '.parquet'), index=False)
    #         print(f"events_data dataset saved successfully in {file_name}")
    # else:
    #     try:
    #         events_df = pd.read_hdf(file_name, key='events_data')
    #     except:
    #         events_df = pd.DataFrame(columns=['Id', 'event_name', 'done', 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality', 'duration(minutes)'])    
    # print(f"events_data dataset loaded successfully from {file_name}")

    # if mode == 'overwrite':
    #     events_df = events_df[events_df['Id'] != subject]
    
    events_df = pd.DataFrame(columns=['Id', 'event_name', 'done', 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality', 'duration(minutes)'])

    fibro_hdf = h5py.File(file_name, 'r+')


    for group in fibro_hdf.keys():
        if mode == 'skip':
            if group == subject:
                break
            else:
                continue
        for key in fibro_hdf[group].keys():
            if 'events_data' in fibro_hdf[group][key].attrs.keys():
                for event, value in json.loads(fibro_hdf[group][key].attrs['events_data'].decode('utf-8')).items():
                    
                    sub_id = group
                    event_name = value['Classification']
                    if 'artifacts_percent' in value.keys():
                        artifacts_percentage = value['artifacts_percent']
                    else:
                        artifacts_percentage = 0

                    if 'corrected_rpeaks_percent' in value.keys():
                        corrected_rpeaks = value['corrected_rpeaks_percent']
                    else:
                        corrected_rpeaks = 0

                    if 'avg_motion' in value.keys():
                        avg_motion = value['avg_motion']
                    else:
                        avg_motion = 0

                    if 'avg_quality' in value.keys():
                        avg_quality = value['avg_quality']
                    else:
                        avg_quality = 0
                    start_time = value['Start_Time']
                    end_time = value['End_Time']
                    duration = end_time - start_time
                    duration = duration/60
                    new_row = {'Id': sub_id, 'event_name': event_name, 'done': False, 'artifacts_percentage_%': artifacts_percentage, 'corrected_rpeaks_%': corrected_rpeaks, 'avg_motion': avg_motion, 'avg_quality': avg_quality, 'duration(minutes)': duration}
                    if 'done' not in events_df.columns:
                        events_df['done'] = False
                    
                    events_df = pd.concat([events_df, pd.DataFrame([new_row])])

                event_name = 'Total'
                filtered_df = events_df[events_df['Id'] == group]

                # Convert the 'artifacts_percentage_%', 'corrected_rpeaks_%', 'avg_motion', 'avg_quality' and 'duration(minutes)' columns to float
                filtered_df['artifacts_percentage_%'] = filtered_df['artifacts_percentage_%'].astype(float)
                filtered_df['corrected_rpeaks_%'] = filtered_df['corrected_rpeaks_%'].astype(float)
                filtered_df['avg_motion'] = filtered_df['avg_motion'].astype(float)
                filtered_df['avg_quality'] = filtered_df['avg_quality'].astype(float)
                filtered_df['duration(minutes)'] = filtered_df['duration(minutes)'].astype(float)
                if 'done' not in filtered_df.columns:
                    filtered_df['done'] = False
                
                filtered_df['done'] = False

                aggregated_values = filtered_df.agg({
                    'artifacts_percentage_%': 'mean',
                    'corrected_rpeaks_%': 'mean',
                    'avg_motion': 'mean',
                    'avg_quality': 'mean',
                    'duration(minutes)': 'sum'
                }).to_dict()
                

                # round the values
                aggregated_values['artifacts_percentage_%'] = round(aggregated_values['artifacts_percentage_%'], 2)
                aggregated_values['corrected_rpeaks_%'] = round(aggregated_values['corrected_rpeaks_%'], 2)
                aggregated_values['avg_motion'] = round(aggregated_values['avg_motion'], 2)
                aggregated_values['avg_quality'] = round(aggregated_values['avg_quality'], 2)
                aggregated_values['duration(minutes)'] = round(aggregated_values['duration(minutes)'], 2)

                new_row_mean = {'Id': sub_id, 'event_name': event_name, 'done': False, 'artifacts_percentage_%': aggregated_values['artifacts_percentage_%'], 'corrected_rpeaks_%': aggregated_values['corrected_rpeaks_%'], 'avg_motion': aggregated_values['avg_motion'], 'avg_quality': aggregated_values['avg_quality'], 'duration(minutes)': aggregated_values['duration(minutes)']}
                
                if 'done' not in events_df.columns:
                    events_df['done'] = False

                events_df = events_df.dropna(subset=['Id'])

                events_df = pd.concat([events_df, pd.DataFrame([new_row_mean])])
                
                break
            break

    fibro_hdf.close()

    # events_df = events_df.reset_index(drop=True)

    # # Add the events_df as a new dataset in the hdf file
    # # Convert the 'duration' column to string
    # events_df['duration(minutes)'] = events_df['duration(minutes)'].astype(str)

    
    # # Remove the 'events_data' dataset if it already exists
    # with pd.HDFStore(file_name, mode='a') as store:
    #     if 'events_data' in store.keys():
    #         store.remove('events_data')
    #     store.close()

    # # events_df['done'] = str(events_df['done'])
    # # Now try saving to HDF5 again
    # # events_df.to_hdf(file_name, key='events_data', mode='a', format='table')
    # # events_df_parquet = events_df.copy()
    # # events_df_parquet['done'] = events_df_parquet['done'].astype(bool)
    # events_df.to_parquet(file_name.replace('.hdf5', '.parquet'), index=False)


    

    return f'Data for {subject} saved successfully to {project}_mindware.hdf5', events_df



@callback(
    Output("modal", "is_open", allow_duplicate=True),
    [Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


