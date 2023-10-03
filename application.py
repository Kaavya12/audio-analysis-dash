import dash
from dash import html
from dash import dcc

#figure = go.Figure(go.Scatter(name="Model", x=top50_results['year'], y=top50_results['rank']))

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap",
    "../assets/styles.css"
]

app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)

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

server = app.server