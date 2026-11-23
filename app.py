import argparse
import logging

import dash
from dash import html
from flask import Flask
import dash_bootstrap_components as dbc


flask_app = Flask(__name__)

app = dash.Dash(__name__, server=flask_app, external_stylesheets=[dbc.themes.SUPERHERO], use_pages=True)

sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page['name'], className='ms-2'),
            ],
            href=page['path'],
            active="exact",
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className="bg-light",
)

app.layout = dbc.Container([
    dbc.Row(
        [
        dbc.Col(
            [
            html.Img(src=app.get_asset_url('EmbeddedImage.jpg'), style={'width': '100%'})
            ], width=2),
        dbc.Col(
            [
            html.H1("Mindware App"),
            html.Hr()
            ], width=8, style={'textAlign': 'left'})
        
    ]),
    dbc.Row(
        [
            dbc.Col([
                sidebar
            ], width=2),
            dbc.Col([
                dash.page_container
            ], width=10)
        ]
    )
], fluid=True)



def main():
    parser = argparse.ArgumentParser(description="Run Mindware App locally")
    parser.add_argument("--browser", action="store_true", help="Serve Dash for a regular browser")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debugging in browser mode")
    parser.add_argument("--port", type=int, default=8050, help="Local port in browser mode")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.browser or args.debug:
        app.run(host="127.0.0.1", port=args.port, debug=args.debug, use_reloader=False)
        return

    import webview
    webview.create_window("Mindware App", flask_app, width=800, height=600)
    webview.start()


if __name__ == "__main__":
    main()
