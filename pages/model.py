import numpy as np
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import io, base64
import librosa
from dash import callback
import warnings
from scipy import stats
import tensorflow as tf
import joblib

#figure = go.Figure(go.Scatter(name="Model", x=top50_results['year'], y=top50_results['rank']))

interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
pipe, enc = joblib.load("models/pipe_10.joblib"), joblib.load("models/enc_10.jobilb")

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap",
    "/assets/model_styles.css",
    "/assets/styles.css"
]

dash.register_page(__name__)

header = html.Div(
    id="app-header-model",
    children=[
        html.H1("Use the model yourself!"),
    ]
)

upload = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(id="upload-data-div", children = [
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        # Allow multiple files to be uploaded
        multiple=False
    ),
])

def columns():
    feature_sizes = dict(chroma_cens=12,
                            tonnetz=6, mfcc=20,
                            zcr=1,
                            spectral_centroid=1,
                            spectral_contrast=7,)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    return columns.sort_values()

def compute_features(x, sr):
    features = pd.Series(index=columns(), dtype=np.float32)
    warnings.filterwarnings('error', module='librosa')
        

    def feature_stats(name, values):
        features.loc[(name, 'mean')] = np.mean(values, axis=1)
        features.loc[(name, 'std')] = np.std(values, axis=1)
        features.loc[(name, 'skew')] = stats.skew(values, axis=1)
        features.loc[(name, 'kurtosis')] = stats.kurtosis(values, axis=1)
        features.loc[(name, 'median')] = np.median(values, axis=1)
        features.loc[(name, 'min')] = np.min(values, axis=1)
        features.loc[(name, 'max')] = np.max(values, axis=1)

    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                n_bins=7*12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)
    del cqt
    
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    del x

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)
    del f

    return (features)

def find_genre(y, sr):
    features = compute_features(y,sr)
    columns = ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'zcr', 'tonnetz']
    features = features.loc[columns]
    transposed_df = pd.DataFrame(features.values.reshape(1, -1),
                              columns=features.index)
    features = pipe.transform(transposed_df)
    features = np.array(features, dtype=np.float32)

    input_shape = input_details[0]['index']
    interpreter.set_tensor(input_shape, features)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    preds = np.argsort(output_data.reshape(-1))
    features = None
    return enc.inverse_transform(preds)[::-1]

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    file = io.BytesIO(decoded)
    try:
        if 'mp3' in filename or 'wav' in filename:
            y, sr = librosa.load(file)
            results = [
                html.H2("Top 5 Predicted Genres")
            ]
            genres = find_genre(y,sr)
            y = None
            for genre in genres:
                results.append(html.P(genre)),#, style=genre_styles[genre]))
            results = html.Div(children=results[:6], id='results_div')
            output = html.Div(["File accepted!", html.Br()], id="output-div")
        else:
            output = html.Div(["Please upload an MP3 or WAV File"], id="output-div")
            results = html.Br()
            
    except Exception as e:
        #print(e)
        return html.Div(id='error-div', children=[ html.Br(), 
            'There was an error processing this file. Please try reuploading or upload a different file if that doesn\'t work. Sorry for the inconvenience!'
        ])

    return html.Div(id="result-div", children=[
        html.H3(f"File Received:"),
        html.H3(f"{filename}"),
        output,
        html.Br(),  # horizontal line
        results
    ])
    
@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(content, name, date):
    if content is not None:
        children = [
            parse_contents(content, name, date)]
        return children


layout = html.Div(
    id="model-main",
    children=[
        header,
        upload,
        html.Div(id='output-data-upload')
    ]
)

