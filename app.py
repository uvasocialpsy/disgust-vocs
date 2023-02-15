# --- PACKAGES ---
import pandas as pd
from pathlib import Path
import scipy.io.wavfile as wavfile
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.io as pio

# --- DASBOARD SETUP ---
st.set_page_config(layout="wide", initial_sidebar_state="auto")
pio.templates.default = "plotly"

# --- PATH SETTINGS ---
THIS_DIR = Path.cwd()
AUDIO_DIR = THIS_DIR / 'disgust-audios'

# --- DATASET ---
@st.cache_data()
def import_data():
    df = pd.read_csv(THIS_DIR / 'disgust_all.csv')
    return df
df = import_data()
#------------------------------------------------------------------------------------------------------------------------

# --- APP TITLE ---
st.title("Welcome to Disgust Vocalizations Dashboard App")
st.markdown("***")

# --- APP LAYOUT ---
layout_1, layout_2 = st.columns(2)

# --- LAYOUT 1 ---
with layout_1:
    # Filters
    filter_1, filter_2 = st.columns(2)
    with filter_1:
        feature_filter_1 = st.radio('Select feature type 👇🏻',
                                       options = ['Categorical', 'Numerical'])
    with filter_2:
        if feature_filter_1 == 'Categorical':
            categorical_filter_1 = st.selectbox('Select feature 👇🏻',
                                        options = ['Disgust category',
                                                    'Noise level',
                                                    'Gender',
                                                    'Age',
                                                    'Linguistic group',
                                                    'Confidence in disgust category',
                                                    'Experience valence',
                                                    'Intensity of expression'])
        else:
            numerical_filter_1 = st.selectbox('Select feature 👇🏻',
                                            options = list(df.columns[22:110]))
    # Visualizastion
    if feature_filter_1 == 'Categorical':
        fig_1_categorical = px.scatter(df, x = 'umap_1', y = 'umap_2', color = categorical_filter_1, hover_name = 'AudioID',
                         title = '2-D Acoustic Features Embeddings')
        fig_1_categorical.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'}, margin = {'t':40, 'l':40, 'r':40, 'b':40})
        selected_points = plotly_events(fig_1_categorical, click_event = True, hover_event = False)
        st.markdown("Please select a point (audio) above 👆🏽 and press play button below to listen 👇🏻")
        try:
            st.audio('disgust-audios/'+df[(df['umap_1'] == selected_points[0]["x"]) & (df['umap_2'] == selected_points[0]["y"])]['AudioID'].reset_index(drop = True)[0] + '.wav')
        except:
            st.audio('disgust-audios/a_ca_001_01.wav')
    else:
        fig_1_numerical = px.scatter(df, x = 'umap_1', y = 'umap_2', color = numerical_filter_1, hover_name = 'AudioID',
                         title = '2-D Acoustic Features Embeddings')
        fig_1_numerical.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'}, margin = {'t':40, 'l':40, 'r':40, 'b':40})
        selected_points = plotly_events(fig_1_numerical, click_event = True, hover_event = False)
        st.markdown("Please select a point (audio) above 👆🏽 and press play button below to listen 👇🏻")
        try:
            st.audio('disgust-audios/'+df[(df['umap_1'] == selected_points[0]["x"]) & (df['umap_2'] == selected_points[0]["y"])]['AudioID'].reset_index(drop = True)[0] + '.wav')
        except:
            st.audio('disgust-audios/a_ca_001_01.wav')

# --- LAYOUT 2 ---
with layout_2:
    # Filters
    filter_1, filter_2 = st.columns(2)
    with filter_1:
        feature_filter_2 = st.radio('Select feature type 👇🏻',
                                       options = ['Categorical', 'Numerical'], index = 1)
    with filter_2:
        if feature_filter_2 == 'Categorical':
            categorical_filter_2 = st.selectbox('Select feature 👇🏻',
                                        options = ['Disgust category',
                                                    'Noise level',
                                                    'Gender',
                                                    'Age',
                                                    'Linguistic group',
                                                    'Confidence in disgust category',
                                                    'Experience valence',
                                                    'Intensity of expression'], key = 'second_categorical')
        else:
            numerical_filter_2 = st.selectbox('Select feature 👇🏻',
                                            options = list(df.columns[22:110]))
    # Visualizastion
    if feature_filter_2 == 'Categorical':
        fig_2_categorical = px.scatter(df, x = 'umap_1', y = 'umap_2', color = categorical_filter_2, hover_name = 'AudioID',
                         title = '2-D Acoustic Features Embeddings')
        fig_2_categorical.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'}, margin = {'t':40, 'l':40, 'r':40, 'b':40})
        selected_points = plotly_events(fig_2_categorical, click_event = True, hover_event = False, key = 'plotly_key')
        st.markdown("Please select a point (audio) above 👆🏽 and press play button below to listen 👇🏻")
        try:
            st.audio('disgust-audios/'+df[(df['umap_1'] == selected_points[0]["x"]) & (df['umap_2'] == selected_points[0]["y"])]['AudioID'].reset_index(drop = True)[0] + '.wav')
        except:
            st.audio('disgust-audios/a_ca_001_01.wav')        
    else:
        fig_2_numerical = px.scatter(df, x = 'umap_1', y = 'umap_2', color = numerical_filter_2, hover_name = 'AudioID',
                         title = '2-D Acoustic Features Embeddings')
        fig_2_numerical.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'}, margin = {'t':40, 'l':40, 'r':40, 'b':40})
        selected_points = plotly_events(fig_2_numerical, click_event = True, hover_event = False)
        st.markdown("Please select a point (audio) above 👆🏽 and press play button below to listen 👇🏻")
        try:
            st.audio('disgust-audios/'+df[(df['umap_1'] == selected_points[0]["x"]) & (df['umap_2'] == selected_points[0]["y"])]['AudioID'].reset_index(drop = True)[0] + '.wav')
        except:
            st.audio('disgust-audios/a_ca_001_01.wav')