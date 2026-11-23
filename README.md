# Mindware App

Mindware App runs locally with Plotly Dash. It imports Mindware exports from a Google Drive folder available through Google Drive for desktop or from any local folder with the same layout.

## Run locally

Use Python 3.11. On Windows, run these commands from the repository directory in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py --browser
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050). For the Windows desktop window, double-click `Mindware app.bat` or run `python app.py` in the virtual environment.

## Import and analyze

1. In **Project Settings**, choose the project and set **Input Path** to the synced folder containing the subject folders. Confirm the sampling rate, channels, and event marker names, then save.
2. In **Loading**, use **Preview Google Drive** to check the folder and **Scan and Load New Subjects** to import new recordings. Place TXT exports in each subject's `MINDWARE` or `HRV` subfolder. An event TXT file is optional for importing the signal and required for event-based analysis.
3. Use **Signal Processing Page** to inspect an event. In **NeuroKit Values New**, load the event names, select a filter if needed, and use **Run All Subjects**. Reload the results table after the analysis finishes.

Imported data and results live in `pages/data/mindwareData/`. To continue with an existing installation, copy this data folder and any custom project templates from `pages/templates/`. Per-subject Parquet and legacy project-wide Parquet are read automatically. For legacy HDF5 imports, place the file at `pages/data/mindwareData/<project>_mindware.hdf5` and use **Migrate HDF5 Data** on the Loading page. The legacy **Events Editing** page uses the HDF5 path in its project settings.

## Debugging

Run `python app.py --debug` in the virtual environment to enable Dash debugging on localhost and view tracebacks in the terminal. The Loading screen shows a report path. Its JSON report records the time spent reading TXT files, processing signals and events, and staging Parquet for each subject. NeuroKit displays the analysis status and log path. Both workflows write logs under `pages/logs/`.

To run an analysis directly with diagnostics:

```powershell
.\.venv\Scripts\python.exe -m pages.scripts.results_generating_smart fibro --debug
```
