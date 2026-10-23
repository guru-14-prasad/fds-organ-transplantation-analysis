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
    header[data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
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


def number_columns(candidates):
    return [column for column in candidates if column in df.columns]


st.markdown(
    """
    <div class="hero">
        <h1>OrganScope</h1>
        <p>Organ donation and transplantation analysis through visual exploration, project questions, machine learning, and prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = ["EDA Visuals", "Analysis Questions", "ML Models", "Predict"]
page = st.selectbox("Section", pages, label_visibility="collapsed")


if page == "EDA Visuals":
    st.title("EDA Visuals")
    st.caption("Inspect distributions, data quality, organ activity, survival, and correlations.")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Records", f"{len(df):,}")
    metric_2.metric("Countries", f"{df['Country'].nunique():,}" if "Country" in df else "N/A")
    metric_3.metric("Organs", f"{df['Organ'].nunique():,}" if "Organ" in df else "N/A")
    metric_4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    distributions, quality, relationships = st.tabs(["Distributions", "Data quality", "Relationships"])
    with distributions:
        fields = st.multiselect(
            "Numeric fields",
            numeric_cols,
            default=number_columns(["Total_Donors", "Transplants", "Waiting_List", "Donor_Rate_PMP"]),
        )
        chart_columns = st.columns(2)
        for index, field in enumerate(fields):
            values = df[field].dropna()
            if len(values):
                with chart_columns[index % 2]:
                    show_chart(
                        px.histogram(
                            sampled(pd.DataFrame({field: values})),
                            x=field,
                            nbins=35,
                            marginal="box",
                            title=f"Distribution of {field}",
                        ),
                        f"distribution_{field}",
                    )
        if fields:
            st.dataframe(df[fields].describe().T, use_container_width=True)

    with quality:
        quality_table = pd.DataFrame(
            {
                "Field": df.columns,
                "Missing": df.isna().sum().values,
                "Missing %": (df.isna().mean().values * 100).round(2),
                "Unique values": [df[column].nunique(dropna=True) for column in df.columns],
            }
        ).sort_values("Missing", ascending=False)
        show_chart(
            px.bar(quality_table.head(18), x="Field", y="Missing", text="Missing", title="Missing values by field"),
            "missing_values",
        )
        st.dataframe(quality_table, use_container_width=True, hide_index=True)

    with relationships:
        left, right = st.columns(2)
        with left:
            if {"Organ", "Transplants"}.issubset(df.columns):
                organ_totals = df.groupby("Organ")["Transplants"].sum().reset_index()
                show_chart(
                    px.bar(organ_totals, x="Organ", y="Transplants", text_auto=".2s", title="Transplants by organ"),
                    "eda_transplants_by_organ",
                )
        with right:
            if {"Organ", "Survival_Rate_5Y_Pct"}.issubset(df.columns):
                survival = df.groupby("Organ")["Survival_Rate_5Y_Pct"].mean().reset_index()
                show_chart(
                    px.bar(survival, x="Organ", y="Survival_Rate_5Y_Pct", text_auto=".2f", title="Mean five-year survival"),
                    "eda_survival_by_organ",
                )
        if len(numeric_cols) > 1:
            show_chart(
                px.imshow(
                    df[numeric_cols].corr(),
                    text_auto=".2f",
                    aspect="auto",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                    title="Numeric correlation matrix",
                ),
                "eda_correlation",
            )


elif page == "Analysis Questions":
    st.title("Analysis Questions")
    st.caption("Select one of the project questions to display its visual analysis.")
    questions = {
        "Q1. Donor volume by continent": "donor_volume",
        "Q2. Top countries by donor rate": "country_rate",
        "Q3. Donor rate by continent": "continent_rate",
        "Q4. Living versus deceased donors": "donor_mix",
        "Q5. Transplants by organ": "organ_transplants",
        "Q6. Waiting list by organ": "organ_waiting",
        "Q7. Donors versus transplants": "donor_transplants",
        "Q8. Donor rate versus waiting list": "rate_waiting",
        "Q9. One-year survival by organ": "survival_1y",
        "Q10. Five-year survival by organ": "survival_5y",
        "Q11. One-year versus five-year survival": "survival_compare",
        "Q12. Living-donor share by organ": "living_share",
        "Q13. Annual transplant trend": "annual_trend",
        "Q14. Highest waiting-list-to-transplant ratio": "backlog",
        "Q15. Eligibility versus living-donor share": "eligibility",
    }
    selected_question = st.selectbox("Project question", list(questions))
    question_key = questions[selected_question]

    if question_key == "donor_volume" and {"Continent", "Total_Donors"}.issubset(df.columns):
        data = df.groupby("Continent")["Total_Donors"].sum().reset_index()
        show_chart(px.bar(data, x="Continent", y="Total_Donors", text_auto=".2s", title=selected_question), "q1")
    elif question_key == "country_rate" and {"Country", "Donor_Rate_PMP"}.issubset(df.columns):
        data = df.groupby("Country")["Donor_Rate_PMP"].median().nlargest(15).reset_index().sort_values("Donor_Rate_PMP")
        show_chart(px.bar(data, x="Donor_Rate_PMP", y="Country", orientation="h", text_auto=".2f", title=selected_question), "q2")
    elif question_key == "continent_rate" and {"Continent", "Donor_Rate_PMP"}.issubset(df.columns):
        data = df.groupby("Continent")["Donor_Rate_PMP"].mean().reset_index()
        show_chart(px.bar(data, x="Continent", y="Donor_Rate_PMP", text_auto=".2f", title=selected_question), "q3")
    elif question_key == "donor_mix" and {"Organ", "Living_Donors", "Deceased_Donors"}.issubset(df.columns):
        data = df.groupby("Organ")[["Living_Donors", "Deceased_Donors"]].sum().reset_index().melt("Organ", var_name="Donor type", value_name="Donors")
        show_chart(px.bar(data, x="Organ", y="Donors", color="Donor type", barmode="stack", title=selected_question), "q4")
    elif question_key == "organ_transplants" and {"Organ", "Transplants"}.issubset(df.columns):
        data = df.groupby("Organ")["Transplants"].sum().reset_index()
        show_chart(px.bar(data, x="Organ", y="Transplants", text_auto=".2s", title=selected_question), "q5")
    elif question_key == "organ_waiting" and {"Organ", "Waiting_List"}.issubset(df.columns):
        data = df.groupby("Organ")["Waiting_List"].sum().reset_index()
        show_chart(px.bar(data, x="Organ", y="Waiting_List", text_auto=".2s", title=selected_question), "q6")
    elif question_key == "donor_transplants" and {"Total_Donors", "Transplants", "Organ"}.issubset(df.columns):
        data = sampled(df[["Total_Donors", "Transplants", "Organ"]].dropna())
        show_chart(px.scatter(data, x="Total_Donors", y="Transplants", color="Organ", title=selected_question), "q7")
    elif question_key == "rate_waiting" and {"Donor_Rate_PMP", "Waiting_List", "Continent"}.issubset(df.columns):
        data = sampled(df[["Donor_Rate_PMP", "Waiting_List", "Continent"]].dropna())
        show_chart(px.scatter(data, x="Donor_Rate_PMP", y="Waiting_List", color="Continent", log_y=True, title=selected_question), "q8")
    elif question_key in {"survival_1y", "survival_5y"} and "Organ" in df.columns:
        field = "Survival_Rate_1Y_Pct" if question_key == "survival_1y" else "Survival_Rate_5Y_Pct"
        if field in df:
            data = df.groupby("Organ")[field].mean().reset_index()
            show_chart(px.bar(data, x="Organ", y=field, text_auto=".2f", title=selected_question), "q_survival")
    elif question_key == "survival_compare" and {"Organ", "Survival_Rate_1Y_Pct", "Survival_Rate_5Y_Pct"}.issubset(df.columns):
        data = df.groupby("Organ")[ ["Survival_Rate_1Y_Pct", "Survival_Rate_5Y_Pct"] ].mean().reset_index().melt("Organ", var_name="Measure", value_name="Percent")
        show_chart(px.bar(data, x="Organ", y="Percent", color="Measure", barmode="group", text_auto=".2f", title=selected_question), "q11")
    elif question_key == "living_share" and {"Organ", "Living_Donor_Pct"}.issubset(df.columns):
        data = df.groupby("Organ")["Living_Donor_Pct"].mean().reset_index()
        show_chart(px.bar(data, x="Organ", y="Living_Donor_Pct", text_auto=".2f", title=selected_question), "q12")
    elif question_key == "annual_trend" and {"Year", "Transplants"}.issubset(df.columns):
        data = df.groupby("Year")["Transplants"].sum().reset_index()
        show_chart(px.line(data, x="Year", y="Transplants", markers=True, title=selected_question), "q13")
    elif question_key == "backlog" and {"Country", "Organ", "Waiting_List", "Transplants"}.issubset(df.columns):
        data = df.copy()
        data["Backlog Ratio"] = np.where(data["Transplants"] > 0, data["Waiting_List"] / data["Transplants"], np.nan)
        data = data.dropna(subset=["Backlog Ratio"]).nlargest(15, "Backlog Ratio")
        show_chart(px.bar(data.sort_values("Backlog Ratio"), x="Backlog Ratio", y="Country", color="Organ", orientation="h", text_auto=".1f", title=selected_question), "q14")
        st.dataframe(data[["Country", "Organ", "Waiting_List", "Transplants", "Backlog Ratio"]], use_container_width=True, hide_index=True)
    elif question_key == "eligibility" and {"Can_Living_Donate", "Organ", "Living_Donor_Pct"}.issubset(df.columns):
        data = df.groupby(["Organ", "Can_Living_Donate"])["Living_Donor_Pct"].mean().reset_index()
        show_chart(px.bar(data, x="Organ", y="Living_Donor_Pct", color="Can_Living_Donate", barmode="group", title=selected_question), "q15")
    else:
        st.info("The selected question needs fields that are not available in this dataset.")


elif page == "ML Models":
    st.title("ML Models")
    st.caption("Train and compare regression and classification models on the current dataset.")
    regression_features = number_columns(["Population_M", "Total_Donors", "Deceased_Rate_PMP", "Waiting_List"])
    classification_features = [column for column in numeric_cols if column not in {"Year"}]
    tab_regression, tab_classification = st.tabs(["Regression", "Classification"])

    with tab_regression:
        st.write("Target: Transplants")
        if len(regression_features) >= 2 and "Transplants" in df:
            train = df[regression_features + ["Transplants"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["Transplants"])
            X = train[regression_features]
            y = pd.to_numeric(train["Transplants"], errors="coerce")
            valid = y.notna()
            X, y = X.loc[valid], y.loc[valid]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
            model = st.selectbox("Regression model", ["Random Forest", "Linear Regression"], key="model_regression")
            if st.button("Run regression", type="primary"):
                estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
                if model == "Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    estimator = LinearRegression()
                pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
                pipeline.fit(x_train, y_train)
                prediction = pipeline.predict(x_test)
                st.session_state["regression_pipeline"] = pipeline
                a, b, c = st.columns(3)
                a.metric("R²", f"{r2_score(y_test, prediction):.4f}")
                b.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, prediction)):,.2f}")
                c.metric("MAE", f"{mean_absolute_error(y_test, prediction):,.2f}")
        else:
            st.warning("Required regression fields are unavailable.")

    with tab_classification:
        st.write("Target: Can_Living_Donate")
        if "Can_Living_Donate" in df and classification_features:
            train = df[classification_features + ["Can_Living_Donate"]].dropna(subset=["Can_Living_Donate"])
            X = train[classification_features]
            y = train["Can_Living_Donate"].astype(str)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
            model = st.selectbox("Classification model", ["Random Forest", "Logistic Regression"], key="model_classification")
            if st.button("Run classification", type="primary"):
                estimator = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
                if model == "Logistic Regression":
                    estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
                pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
                pipeline.fit(x_train, y_train)
                prediction = pipeline.predict(x_test)
                a, b = st.columns(2)
                a.metric("Accuracy", f"{accuracy_score(y_test, prediction):.4f}")
                b.metric("Macro-F1", f"{f1_score(y_test, prediction, average='macro'):.4f}")
                st.dataframe(pd.crosstab(y_test, prediction, rownames=["Actual"], colnames=["Predicted"]), use_container_width=True)
        else:
            st.warning("The classification target is unavailable.")


elif page == "Predict":
    st.title("Predict")
    st.caption("Generate an exploratory transplant-volume prediction from four numeric inputs.")
    features = number_columns(["Population_M", "Total_Donors", "Deceased_Rate_PMP", "Waiting_List"])
    if "Transplants" not in df or len(features) < 2:
        st.warning("Required prediction fields are unavailable.")
    else:
        train = df[features + ["Transplants"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["Transplants"])
        X = train[features]
        y = pd.to_numeric(train["Transplants"], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        predictor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        predictor.fit(X, y)
        left, right = st.columns(2)
        with left:
            inputs = {}
            for feature in features:
                values = X[feature].dropna()
                inputs[feature] = st.number_input(feature, min_value=float(values.min()), max_value=float(values.max()), value=float(values.median()), key=f"input_{feature}")
            predict = st.button("Generate prediction", type="primary", use_container_width=True)
        with right:
            if predict:
                value = float(predictor.predict(pd.DataFrame([inputs]))[0])
                median = float(y.median())
                st.metric("Predicted transplants", f"{value:,.1f}", delta=f"{value - median:+,.1f} vs median")
                st.caption(f"Dataset median: {median:,.1f}. This is an exploratory model output.")
                st.warning("Do not use this prediction for clinical or policy decisions.")
            else:
                st.info("Enter the values and select Generate prediction.")

