from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    st.caption("These questions match the notebook exactly.")
    questions = {
        "Q1. What is the structure and coverage of the dataset, and how complete is it?": "structure",
        "Q2. How many records are inconsistent or implausible, and how should they be handled?": "quality",
        "Q3. Which continents have the highest and lowest donor activity per million population?": "continent_rates",
        "Q4. Which organs contribute most to total transplants and donors?": "organ_contribution",
        "Q5. Which countries lead in donor performance, and is the ranking consistent across organs?": "country_ranking",
        "Q6. How have donor and transplant activity changed from 2010 to 2025?": "trend",
        "Q7. How much do health systems rely on living versus deceased donors, and does this depend on the organ?": "donor_mix",
        "Q8. Where is the gap between organ demand and supply largest?": "demand_supply",
        "Q9. What factors are associated with longer average waiting times?": "waiting_factors",
        "Q10. How do one-year and five-year survival differ by organ, and which organ shows the steepest long-term decline?": "survival",
        "Q11. Does higher transplant volume go with better survival outcomes?": "volume_survival",
        "Q12. Is the difference in mean donor rate across continents statistically significant?": "continent_test",
        "Q13. Do organs that allow living donation have significantly higher survival than organs that rarely do?": "eligibility_survival",
        "Q14. Which variables best explain total transplant activity?": "regression_question",
        "Q15. Can country-organ records be grouped into distinct performance profiles?": "clustering_question",
    }
    selected_question = st.selectbox("Project question", list(questions))
    question_key = questions[selected_question]

    if question_key == "structure":
        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", f"{len(df.columns):,}")
        c.metric("Countries", f"{df['Country'].nunique():,}" if "Country" in df else "N/A")
        d.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        coverage = pd.DataFrame({"Dimension": ["Country", "Organ", "Continent", "Year"], "Distinct values": [df[c].nunique() if c in df else 0 for c in ["Country", "Organ", "Continent", "Year"]]})
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        missing = df.isna().sum().sort_values(ascending=False).reset_index(name="Missing values").rename(columns={"index": "Field"})
        show_chart(px.bar(missing[missing["Missing values"] > 0], x="Field", y="Missing values", title="Q1 — Missing Values by Field"), "q1_missing")
        if {"Year", "Continent"}.issubset(df.columns): show_chart(px.histogram(df, x="Year", color="Continent", title="Q1 — Record Coverage by Year and Continent"), "q1_coverage")
    elif question_key == "quality":
        checks = []
        checks.append(("Exact duplicate rows", int(df.duplicated().sum())))
        if {"Country", "Year", "Organ"}.issubset(df.columns): checks.append(("Duplicate Country-Year-Organ rows", int(df.duplicated(["Country", "Year", "Organ"], keep=False).sum())))
        if "Year" in df: checks.append(("Years outside 2010–2025", int((~df["Year"].between(2010, 2025)).sum())))
        if "Donor_Rate_PMP" in df:
            q1, q3 = df["Donor_Rate_PMP"].quantile([.25, .75]); iqr = q3 - q1
            checks.append(("Donor-rate IQR outliers", int(((df["Donor_Rate_PMP"] < q1 - 1.5 * iqr) | (df["Donor_Rate_PMP"] > q3 + 1.5 * iqr)).sum())))
        st.dataframe(pd.DataFrame(checks, columns=["Audit", "Count"]), use_container_width=True, hide_index=True)
        if "Donor_Rate_PMP" in df: show_chart(px.box(df, y="Donor_Rate_PMP", points="outliers", title=selected_question), "q2")
    elif question_key == "continent_rates" and {"Continent", "Donor_Rate_PMP", "Deceased_Rate_PMP"}.issubset(df.columns):
        data = df.melt(id_vars=["Continent"], value_vars=["Donor_Rate_PMP", "Deceased_Rate_PMP"], var_name="Rate type", value_name="Rate per million").dropna()
        show_chart(px.box(data, x="Continent", y="Rate per million", color="Rate type", points="outliers", title=selected_question), "q3")
    elif question_key == "organ_contribution" and {"Organ", "Transplants", "Total_Donors"}.issubset(df.columns):
        data = df.groupby("Organ").agg(Transplants=("Transplants", "sum"), Donors=("Total_Donors", "sum")).reset_index().sort_values("Transplants", ascending=False)
        data["Cumulative transplant share (%)"] = 100 * data["Transplants"].cumsum() / data["Transplants"].sum()
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Bar(x=data["Organ"], y=data["Transplants"], name="Transplants"), secondary_y=False)
        figure.add_trace(go.Scatter(x=data["Organ"], y=data["Cumulative transplant share (%)"], name="Cumulative share %", mode="lines+markers"), secondary_y=True)
        figure.update_layout(title="Q4 — Organ Transplant Pareto Analysis", xaxis_title="Organ")
        figure.update_yaxes(title_text="Transplants", secondary_y=False)
        figure.update_yaxes(title_text="Cumulative share (%)", range=[0, 105], secondary_y=True)
        show_chart(figure, "q4")
    elif question_key == "country_ranking" and {"Country", "Donor_Rate_PMP"}.issubset(df.columns):
        overall = df.groupby("Country")["Donor_Rate_PMP"].median().nlargest(10).rename("Median rate").reset_index(); overall["Scope"] = "Overall"
        parts = [overall]
        for organ_name in ["Kidney", "Liver"]:
            if "Organ" in df:
                subset = df[df["Organ"].astype(str).str.casefold() == organ_name.casefold()]
                if not subset.empty:
                    organ = subset.groupby("Country")["Donor_Rate_PMP"].median().nlargest(10).rename("Median rate").reset_index(); organ["Scope"] = organ_name; parts.append(organ)
        data = pd.concat(parts, ignore_index=True).sort_values("Median rate")
        show_chart(px.bar(data, x="Median rate", y="Country", color="Scope", orientation="h", barmode="group", title=selected_question), "q5")
    elif question_key == "trend" and {"Year", "Total_Donors", "Transplants"}.issubset(df.columns):
        data = df.groupby("Year")[["Total_Donors", "Transplants"]].sum().reset_index().melt("Year", var_name="Measure", value_name="Total")
        show_chart(px.line(data, x="Year", y="Total", color="Measure", markers=True, title=selected_question), "q6")
    elif question_key == "donor_mix" and {"Organ", "Living_Donors", "Deceased_Donors"}.issubset(df.columns):
        data = df.groupby("Organ")[["Living_Donors", "Deceased_Donors"]].sum().reset_index().melt("Organ", var_name="Donor type", value_name="Donors")
        show_chart(px.bar(data, x="Organ", y="Donors", color="Donor type", barmode="stack", title=selected_question), "q7")
    elif question_key == "demand_supply" and {"Country", "Organ", "Waiting_List", "Transplants"}.issubset(df.columns):
        data = df.copy(); data["Waiting-list / transplant ratio"] = np.where(data["Transplants"] > 0, data["Waiting_List"] / data["Transplants"], np.nan)
        data = data.dropna(subset=["Waiting-list / transplant ratio"]).nlargest(15, "Waiting-list / transplant ratio")
        show_chart(px.bar(data.sort_values("Waiting-list / transplant ratio"), x="Waiting-list / transplant ratio", y="Country", color="Organ", orientation="h", text_auto=".1f", title="Q8 — 15 Highest-Pressure Country-Organ Records"), "q8")
        st.dataframe(data[["Country", "Organ", "Waiting_List", "Transplants", "Waiting-list / transplant ratio"]], use_container_width=True, hide_index=True)
    elif question_key == "waiting_factors" and {"Waiting_List", "Population_M", "Donor_Rate_PMP", "Transplants"}.issubset(df.columns):
        fields = ["Waiting_List", "Population_M", "Donor_Rate_PMP", "Transplants"]
        corr = df[fields].corr()
        show_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Q9 — Pearson Correlation Matrix"), "q9")
    elif question_key == "survival" and {"Organ", "Survival_Rate_1Y_Pct", "Survival_Rate_5Y_Pct"}.issubset(df.columns):
        data = df.groupby("Organ")[["Survival_Rate_1Y_Pct", "Survival_Rate_5Y_Pct"]].mean().reset_index().melt("Organ", var_name="Measure", value_name="Percent")
        show_chart(px.bar(data, x="Organ", y="Percent", color="Measure", barmode="group", text_auto=".2f", title="Q10 — One-Year vs Five-Year Survival by Organ"), "q10")
        decline_data = df.assign(Decline=df["Survival_Rate_1Y_Pct"] - df["Survival_Rate_5Y_Pct"])
        decline = decline_data.groupby("Organ")["Decline"].mean().reset_index(name="Mean decline")
        show_chart(px.bar(decline.sort_values("Mean decline"), x="Mean decline", y="Organ", orientation="h", text_auto=".2f", title="Q10 — Survival Decline (Percentage Points) by Organ"), "q10_decline")
    elif question_key == "volume_survival" and {"Transplants", "Survival_Rate_5Y_Pct", "Organ"}.issubset(df.columns):
        data = sampled(df[["Transplants", "Survival_Rate_1Y_Pct", "Survival_Rate_5Y_Pct", "Organ"]].dropna())
        data = data[data["Transplants"] > 0]
        for survival_field in ["Survival_Rate_1Y_Pct", "Survival_Rate_5Y_Pct"]:
            figure = px.scatter(data, x="Transplants", y=survival_field, color="Organ", log_x=True, title=f"Q11 — Transplant Volume vs {survival_field} (log x-axis)")
            show_chart(figure, f"q11_{survival_field}")
            st.caption(f"Pearson r = {data['Transplants'].corr(data[survival_field]):.3f} | Spearman rho = {data['Transplants'].corr(data[survival_field], method='spearman'):.3f}")
    elif question_key == "continent_test" and {"Continent", "Donor_Rate_PMP"}.issubset(df.columns):
        groups = {name: group["Donor_Rate_PMP"].dropna().values for name, group in df.groupby("Continent") if group["Donor_Rate_PMP"].notna().sum() >= 2}
        statistic, p_value = kruskal(*groups.values())
        a, b = st.columns(2); a.metric("Kruskal–Wallis H", f"{statistic:.4f}"); b.metric("p-value", f"{p_value:.6g}")
        st.info("Reject H0 at α = 0.05." if p_value < .05 else "Fail to reject H0 at α = 0.05.")
        pairwise = []
        names = list(groups)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                _, raw_p = __import__("scipy").stats.mannwhitneyu(groups[names[i]], groups[names[j]], alternative="two-sided")
                pairwise.append((names[i], names[j], raw_p))
        pairwise.sort(key=lambda row: row[2]); m = len(pairwise)
        holm = []; running = 0
        for rank, (name_a, name_b, raw_p) in enumerate(pairwise):
            adjusted = min(1, (m - rank) * raw_p); running = max(running, adjusted); holm.append((name_a, name_b, raw_p, running))
        st.dataframe(pd.DataFrame(holm, columns=["Continent A", "Continent B", "Raw p", "Holm-adjusted p"]), use_container_width=True, hide_index=True)
        show_chart(px.box(df, x="Continent", y="Donor_Rate_PMP", points="outliers", title=selected_question), "q12")
    elif question_key == "eligibility_survival" and {"Can_Living_Donate", "Survival_Rate_5Y_Pct"}.issubset(df.columns):
        data = df[["Can_Living_Donate", "Survival_Rate_5Y_Pct"]].dropna()
        show_chart(px.box(data, x="Can_Living_Donate", y="Survival_Rate_5Y_Pct", points="outliers", title=selected_question), "q13")
        st.caption("The notebook finds six eligibility labels in this dataset, so the two-group Mann–Whitney assumption is not met without recoding.")
    elif question_key == "regression_question":
        features = ["Population_M", "Total_Donors", "Deceased_Rate_PMP", "Waiting_List"]
        if set(features + ["Transplants"]).issubset(df.columns):
            from sklearn.linear_model import LinearRegression
            train = df[features + ["Transplants"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["Transplants"])
            X, y = train[features], pd.to_numeric(train["Transplants"], errors="coerce")
            valid = y.notna(); X, y = X.loc[valid], y.loc[valid]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE)
            models = {"Linear Regression": LinearRegression(), "Random Forest Regressor": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)}
            fitted = {}; results = []
            for name, estimator in models.items():
                pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
                pipeline.fit(x_train, y_train); prediction = pipeline.predict(x_test); fitted[name] = pipeline
                results.append({"Model": name, "R2": r2_score(y_test, prediction), "MAE": mean_absolute_error(y_test, prediction), "RMSE": np.sqrt(mean_squared_error(y_test, prediction))})
            result_table = pd.DataFrame(results).sort_values("RMSE")
            st.dataframe(result_table.round(4), use_container_width=True, hide_index=True)
            show_chart(px.bar(result_table.melt(id_vars="Model", value_vars=["R2", "MAE", "RMSE"], var_name="Metric", value_name="Score"), x="Model", y="Score", color="Metric", barmode="group", title="Q14 — Regression Model Comparison"), "q14_models")
            best = result_table.iloc[0]["Model"]; prediction = fitted[best].predict(x_test)
            show_chart(px.scatter(x=y_test, y=prediction, labels={"x": "Actual transplants", "y": "Predicted transplants"}, title=f"Q14 — Actual vs Predicted ({best})"), "q14_actual_predicted")
            residuals = y_test - prediction
            figure = px.scatter(x=prediction, y=residuals, labels={"x": "Predicted", "y": "Residual"}, title="Q14 — Residual Diagnostic")
            figure.add_hline(y=0, line_dash="dash")
            show_chart(figure, "q14_residuals")
        else:
            st.info("The notebook regression predictors are not all available in this dataset.")
    elif question_key == "clustering_question":
        st.info("The notebook retains this clustering question but does not execute the clustering model in its reduced two-model version. The ML Models section contains the executed predictive models.")
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

