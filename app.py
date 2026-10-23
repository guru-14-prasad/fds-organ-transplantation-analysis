from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import f_oneway, kruskal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="OrganScope", page_icon=None, layout="wide")

RANDOM_STATE = 42
APP_DIR = Path(__file__).resolve().parent
DATA_FILES = [
    APP_DIR / "final organ dataset.csv",
    APP_DIR / "final organ dataset(3).csv",
    APP_DIR / "final_organ_clean.csv",
    APP_DIR / "organ_donation_cleaned_tableau.csv",
]


st.markdown(
    """
    <style>
    .stApp { background: #081225; color: #eef5ff; }
    .block-container { padding-top: 1.2rem; max-width: 1500px; }
    [data-testid="stSidebar"] { background: #0e1b30; }
    .hero {
        padding: 26px 30px;
        border-radius: 18px;
        border: 1px solid #285078;
        background: linear-gradient(120deg, #102b4d, #18285a);
        margin-bottom: 20px;
    }
    .hero h1 { color: white; margin: 0 0 8px 0; }
    .hero p { color: #c9dbef; margin: 0; font-size: 1.05rem; }
    div[data-testid="stMetric"] {
        background: #102038;
        border: 1px solid #29425f;
        border-radius: 14px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path_string, modified_time):
    data = pd.read_csv(path_string)
    data.columns = [str(column).strip() for column in data.columns]
    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].replace(r"^\s*$", np.nan, regex=True)
        data[column] = data[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    for column in data.columns:
        if column not in data.select_dtypes(include=["object", "string"]).columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def find_data_file():
    return next((file for file in DATA_FILES if file.exists()), None)


DATA_PATH = find_data_file()
if DATA_PATH is None:
    st.error("Dataset not found. Place final organ dataset.csv beside app.py.")
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()


def sampled(data, limit=6000):
    if len(data) <= limit:
        return data
    return data.sample(limit, random_state=RANDOM_STATE)


def show_chart(figure, key):
    figure.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=25))
    st.plotly_chart(figure, use_container_width=True, key=key)
