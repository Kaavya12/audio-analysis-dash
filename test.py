"""import numpy as np
import pandas as pd

top50_results = pd.read_csv("top_50_predicted_data_mod9.csv")
top50_results['display_genre'] = np.where(top50_results['top_predicted_genre']=="Experimental", top50_results['second_predicted_genre'], top50_results['top_predicted_genre'])
top50_results = top50_results.sort_values(['year', 'display_genre'])
top50_results.to_csv('top_50_predicted_data_mod9_v2.csv')"""

import librosa
import pandas as pd
import warnings
from scipy import stats
import numpy as np
import tensorflow as tf
import joblib

def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
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

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    del x

    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    feature_stats('chroma_stft', f)

    f = librosa.feature.rms(S=stft)
    feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)

    return (features)


model = tf.keras.models.load_model("models/model_10.h5")
pipe, enc = joblib.load("models/pipe_10.joblib"), joblib.load("models/enc_10.jobilb")

def find_genre(y, sr):
    features = compute_features(y,sr)
    columns = ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'zcr', 'tonnetz']
    features = features.loc[columns]
    transposed_df = pd.DataFrame(features.values.reshape(1, -1),
                              columns=features.index)
    features = pipe.transform(transposed_df)
    preds = model.predict(features)[0]
    preds = np.argsort(preds)
    return enc.inverse_transform(preds)[::-1]

y, sr = librosa.load("/Users/kaavyamahajan/Desktop/fma_small/122/122533.mp3")
print(find_genre(y, sr))