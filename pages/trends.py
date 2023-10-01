import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import pandas as pd

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap",
    "/assets/trends_styles.css",
    "/assets/styles.css",
    
]

dash.register_page(__name__, path='/')


header = html.Div(
    id="app-header",
    children=[
        html.H1("Genre Classification", id='header-title'),
        html.H2("A display of the evolving trend of popular music over the last 50 years", id='header-text')
    ]
)

about_text = dcc.Markdown ("""
This app has two components, built around the use of a genre classification model. The model was trained on data obtained from the data provided at (to be filled). Audio features in the dataset have been extracted using the librosa library. The model was trained using the tensorflow and keras libraries, and involves an ANN architecture.  

For the first component of this app, the model was used to determine the genre of the top 50 songs of the last 50 years (as per Billboard rankings). Audio clips were first found for these songs, and then the features of interest were extracted using librosa on my end. The same has been presented here in the form of various charts.   

For the second component, I have presented the model for public use, wherein you, the user can upload any audio file, and the model will provide a predicted genre for the same.
""")

intro = html.Div(
    id="about",
    children=[
        html.H2("About this app"),
        html.P(about_text, id='about-text')
    ]
)

landing = html.Div(id='landing', children=[header,
        html.Br(),
        intro],
)

def createFigs():
    top50_results = pd.read_csv("data/top_50_predicted_data_mod10_v2.csv")
    top50_results['display_genre'] = np.where(top50_results['top_predicted_genre']=="Experimental", top50_results['second_predicted_genre'], top50_results['top_predicted_genre'])
    top50_results['display_genre'] = np.where(top50_results['display_genre']=="Electronic", top50_results['third_predicted_genre'], top50_results['display_genre'])
    top50_results = top50_results.sort_values(['year', 'display_genre'])
    
    years = list(range(1973,2023))
    yAxes = pd.Series(list(range(1,51))*50)
    
    rankPlot = px.scatter(top50_results, 
                          x='year', y='rank', color='display_genre',
                          hover_data=['year', 'rank', 'singer', 'song', 'top_predicted_genre', 'second_predicted_genre'])
    
    rankPlot.update_traces(marker={'size': 8})
    rankPlot.update_layout(
        xaxis=dict(
            title_text="Year"
        ),
        yaxis=dict(
            title_text="Rank"
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=18,
            font_family="Poppins"
        ),
        font=dict(
            family="Poppins",
            size=20,
        ), legend_title="Genre"
    )
    rankPlot.layout.legend.itemsizing = 'constant'
    
    config = {'responsive': True}
    rankPlot.layout._config= config
    
    rankFig = dcc.Graph(id='rank-plot', figure=rankPlot, style={'min-height': '600px', 'height': '50vh'})

    catPlot = px.scatter(top50_results, 
                         x='year', y=yAxes, color='display_genre',
                         hover_data=['year', 'rank', 'singer', 'song', 'top_predicted_genre', 'second_predicted_genre'])
    catPlot.update_traces(marker={'size': 8})
    catPlot.update_layout(
        xaxis=dict(
            title_text="Year"
        ),
        yaxis=dict(
            title_text="Number"
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=18,
            font_family="Poppins"
        ),
        font=dict(
            family="Poppins",
            size=20,
        ), legend_title="Genre"
    )
    catPlot.layout.legend.itemsizing = 'constant'
    
    catFig = dcc.Graph(id='cat-plot', figure=catPlot, style={'min-height': '600px', 'height': '50vh'})
    
    return rankFig, catFig

rankFig, catFig = createFigs()

layout = html.Div(
    id="main",
    children=[
        landing,
        rankFig,
        catFig
    ]
)