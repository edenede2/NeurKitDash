"""Discover Mindware files and select the parser before reading a recording."""

import os
import re

import pandas as pd


def subject_folders(input_path):
    with os.scandir(input_path) as entries:
        return {entry.name: entry.path for entry in entries
                if "test" not in entry.name.lower() and entry.is_dir()}


def find_recording(subject_path):
    """Return folder type, signal path and optional marker path."""
    with os.scandir(subject_path) as entries:
        folders = {entry.name.upper(): entry.path for entry in entries
                   if entry.name.upper() in {"MINDWARE", "HRV"} and entry.is_dir()}
    # Also support older layouts that store the exports directly in the subject folder.
    for folder_type, folder_path in (*folders.items(), ("Subject folder", subject_path)):
        with os.scandir(folder_path) as entries:
            files = sorted((entry for entry in entries
                            if entry.name.lower().endswith(".txt") and entry.is_file()),
                           key=lambda entry: entry.name.casefold())
        event_files = [entry for entry in files if any(
            word in entry.name.lower() for word in ("event", "marker", "trigger"))]
        signal_files = [entry for entry in files if entry not in event_files and not any(
            word in entry.name.lower() for word in ("signals.txt", "processed", "output", "result"))]
        if signal_files:
            signal = next((entry for entry in signal_files if entry.name.lower().endswith("data.txt")),
                          next((entry for entry in signal_files if any(
                              word in entry.name.lower() for word in ("mindware", "bio", "hrv"))),
                               signal_files[0]))
            events = next((entry for entry in event_files if entry.name.lower().endswith("events.txt")),
                          event_files[0] if event_files else None)
            if events is None:
                events = next((entry for entry in signal_files if entry is not signal), None)
            return folder_type, signal.path, events.path if events else None
    return next(iter(folders), None), None, None


def read_recording(path, *, nrows=None):
    """Read a TSV with headers on the first line or following a metadata line."""
    with open(path, "rb") as recording:
        first = recording.readline().decode("utf-8-sig", errors="replace").strip()
        second = recording.readline().decode("utf-8", errors="replace").strip()
        if not first:
            raise ValueError("The recording is empty")
        channels = r"time|timestamp|bio|ecg|ekg|gs[rc]|eda|conductance|motion"
        first_is_metadata = bool(re.match(r"(?i)^\s*(?:sample|sampling)\s*rate\b", first))
        skip = first_is_metadata or (
            not re.search(channels, first, re.I) and bool(re.search(channels, second, re.I))
        )
        recording.seek(0)
        frame = pd.read_csv(recording, sep="\t", skiprows=int(skip), low_memory=False, nrows=nrows)
    if frame.empty:
        raise ValueError("The recording has headers but no samples")
    return frame, "metadata + header" if skip else "header"
