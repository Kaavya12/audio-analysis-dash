
import numpy as np
from keras.models import load_model
from PIL import Image
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import pickle
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

#figure = go.Figure(go.Scatter(name="Model", x=top50_results['year'], y=top50_results['rank']))

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap",
    "../assets/styles.css"
]

app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)

server = app.server

app.css.config.serve_locally = False
app.config.suppress_callback_exceptions = True

nav = html.Nav(id='nav', children=[
    html.Div([
        dcc.Link(page["name"], href=page['path'])    
        for page in dash.page_registry.values()
    ])
])

app.layout = html.Div(id='app-main', children=[
    nav,
    dash.page_container
]
)

if __name__ == "__main__":
    app.run_server()