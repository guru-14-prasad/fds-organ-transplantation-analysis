from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import kruskal, f_oneway, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, silhouette_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="OrganScope | Donation Analytics", layout="wide")

APP_DIR = Path(__file__).resolve().parent
# Try the preferred filename first, then common filenames used in this project.
DATA_CANDIDATES = [
    APP_DIR / "final organ dataset(3).csv",
    APP_DIR / "final organ dataset.csv",
    APP_DIR / "final_organ_clean.csv",
    APP_DIR / "final_organ_dataset.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])
RANDOM_STATE = 42
PLOT_SAMPLE = 6000

st.markdown("""
<style>
.stApp {background:#081225;color:#f3f7ff}
[data-testid="stSidebar"] {background:#101b30}
.block-container {padding-top:1.3rem;padding-bottom:2rem}
.hero {padding:26px 30px;border:1px solid #24486e;border-radius:20px;
background:linear-gradient(120deg,#102a4c,#123a59);margin-bottom:18px}
.hero h1 {color:#fff;margin-bottom:8px}
.hero p {color:#c4d8ef;margin:0}
div[data-testid="stMetric"] {background:#111f34;border:1px solid #293b55;padding:14px 16px;border-radius:14px}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(path, modified):
    data = pd.read_csv(path)
    data.columns = [str(c).strip() for c in data.columns]
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].replace(r"^\s*$", np.nan, regex=True)
        data[col] = data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    for col in data.select_dtypes(exclude=["object", "string"]).columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Put the CSV in the same folder as app.py.")
    st.stop()
try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read dataset: {exc}")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

def chart(fig, key=None):
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=25), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, key=key)

def sample(data, limit=PLOT_SAMPLE):
    return data if len(data) <= limit else data.sample(limit, random_state=RANDOM_STATE)

def safe_mean(col):
    return pd.to_numeric(df[col], errors="coerce").mean() if col in df else np.nan

PAGES = [
    "Overview", "Data Explorer", "Country & Organ Analysis",
    "Trends & Survival", "Statistical Tests", "Regression Models",
    "Classification Models", "Clustering", "About Project"
]

st.markdown('<div class="nav-strip"></div>', unsafe_allow_html=True)
_, nav_column = st.columns([4.8, 1.4], vertical_alignment="center")
with nav_column:
    page = st.selectbox("Navigate", PAGES, label_visibility="collapsed", key="top_navigation")

if page == "Overview":
    st.markdown("""<div class="hero">
    <p style="letter-spacing:2px;font-weight:700;color:#60e0ee">GLOBAL HEALTH • DATA SCIENCE</p>
    <h1>Global Organ Donation & Transplantation</h1>
    <p>Explore donation rates, transplant activity, waiting-list pressure, and patient survival across countries, organs, continents, and years.</p>
    </div>""", unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Records", f"{len(df):,}")
    a2 = df["Country"].nunique() if "Country" in df else 0
    b.metric("Countries", f"{a2:,}")
    c.metric("Organs", f"{df['Organ'].nunique():,}" if "Organ" in df else "N/A")
    d.metric("Years covered", f"{int(df['Year'].min())}–{int(df['Year'].max())}" if "Year" in df and df["Year"].notna().any() else "N/A")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total transplants", f"{df['Transplants'].sum():,.0f}" if "Transplants" in df else "N/A")
    m2.metric("Total donors", f"{df['Total_Donors'].sum():,.0f}" if "Total_Donors" in df else "N/A")
    m3.metric("Mean donor rate / million", f"{safe_mean('Donor_Rate_PMP'):.2f}" if "Donor_Rate_PMP" in df else "N/A")
    m4.metric("Duplicate rows", f"{df.duplicated().sum():,}")
    st.subheader("Dataset preview")
    st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    left,right = st.columns(2)
    with left:
        if {"Organ","Transplants"}.issubset(df.columns):
            organ = df.groupby("Organ", dropna=False)["Transplants"].sum().sort_values(ascending=False).reset_index()
            chart(px.bar(organ, x="Organ", y="Transplants", title="Total transplants by organ", text_auto=".2s"))
    with right:
        if {"Continent","Donor_Rate_PMP"}.issubset(df.columns):
            cont = df.groupby("Continent", dropna=False)["Donor_Rate_PMP"].median().reset_index()
            chart(px.bar(cont, x="Continent", y="Donor_Rate_PMP", title="Median donor rate by continent", text_auto=".2f"))
    st.subheader("Data quality overview")
    quality = pd.DataFrame({"Feature":df.columns, "Type":[str(df[c].dtype) for c in df.columns],
        "Missing":[int(df[c].isna().sum()) for c in df.columns],
        "Missing %":(df.isna().mean().values*100).round(2),
        "Unique":[int(df[c].nunique(dropna=True)) for c in df.columns]})
    chart(px.bar(quality.sort_values("Missing",ascending=False).head(18), x="Feature", y="Missing", title="Fields with most missing values", text="Missing"))
    with st.expander("Full data-quality table"):
        st.dataframe(quality, use_container_width=True, hide_index=True)
    if len(numeric_cols)>1:
        chart(px.imshow(df[numeric_cols].corr(), text_auto=".2f", aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Numeric correlation heatmap"))

elif page == "Data Explorer":
    st.title("Data Explorer")
    feature = st.selectbox("Choose a field", ["All fields"] + df.columns.tolist())
    if feature == "All fields":
        t1,t2,t3 = st.tabs(["Numeric distributions","Category counts","Missing values & outliers"])
        with t1:
            selected = st.multiselect("Numeric fields", numeric_cols, default=numeric_cols[:min(5,len(numeric_cols))])
            cols = st.columns(2)
            for i,col in enumerate(selected):
                with cols[i%2]:
                    vals = pd.to_numeric(df[col],errors="coerce").dropna()
                    if len(vals):
                        chart(px.histogram(sample(pd.DataFrame({col:vals})), x=col, nbins=35, marginal="box", title=f"Distribution: {col}"), key=f"hist_{col}")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        with t2:
            selected = st.multiselect("Categorical fields", categorical_cols, default=categorical_cols[:min(4,len(categorical_cols))])
            for col in selected:
                counts=df[col].fillna("Missing").astype(str).value_counts().head(20).rename_axis("Category").reset_index(name="Count")
                chart(px.bar(counts,x="Category",y="Count",title=f"Top categories: {col}",text_auto=True),key=f"cat_{col}")
        with t3:
            missing=pd.DataFrame({"Feature":df.columns,"Missing":df.isna().sum().values,"Missing %":(df.isna().mean().values*100).round(2)})
            st.dataframe(missing.sort_values("Missing",ascending=False),use_container_width=True,hide_index=True)
            out=[]
            for col in numeric_cols:
                s=df[col].dropna()
                if len(s)>=4:
                    q1,q3=s.quantile([.25,.75]); iqr=q3-q1
                    out.append({"Feature":col,"IQR outliers":int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())})
            if out: st.dataframe(pd.DataFrame(out).sort_values("IQR outliers",ascending=False),use_container_width=True,hide_index=True)
    else:
        s=df[feature]
        if pd.api.types.is_numeric_dtype(s):
            chart(px.histogram(sample(pd.DataFrame({feature:s.dropna()})),x=feature,nbins=40,marginal="box",title=f"Distribution: {feature}"))
            st.dataframe(s.describe().to_frame("Summary"))
        else:
            counts=s.fillna("Missing").astype(str).value_counts().head(40).rename_axis("Value").reset_index(name="Count")
            chart(px.bar(counts,x="Value",y="Count",title=f"Category frequency: {feature}",text_auto=True))
            st.dataframe(counts,use_container_width=True,hide_index=True)

elif page == "Country & Organ Analysis":
    st.title("Country & Organ Analysis")
    if not {"Country","Organ"}.issubset(df.columns):
        st.warning("Country and Organ columns are required for this page.")
    else:
        countries=sorted(df["Country"].dropna().unique())
        organs=sorted(df["Organ"].dropna().unique())
        continents=sorted(df["Continent"].dropna().unique()) if "Continent" in df else []
        c1,c2,c3=st.columns(3)
        country_filter=c1.multiselect("Countries",countries,default=[])
        organ_filter=c2.multiselect("Organs",organs,default=organs)
        continent_filter=c3.multiselect("Continents",continents,default=[])
        view=df.copy()
        if country_filter: view=view[view["Country"].isin(country_filter)]
        if organ_filter: view=view[view["Organ"].isin(organ_filter)]
        if continent_filter and "Continent" in view: view=view[view["Continent"].isin(continent_filter)]
        left,right=st.columns(2)
        with left:
            if "Donor_Rate_PMP" in view:
                top=view.groupby("Country")["Donor_Rate_PMP"].mean().sort_values(ascending=False).head(15).reset_index()
                chart(px.bar(top,x="Donor_Rate_PMP",y="Country",orientation="h",title="Top countries by mean donor rate"))
        with right:
            if "Transplants" in view:
                organ_total=view.groupby("Organ")["Transplants"].sum().reset_index()
                chart(px.pie(organ_total,names="Organ",values="Transplants",hole=.45,title="Transplant share by organ"))
        if {"Organ","Living_Donor_Pct","Deceased_Donor_Pct"}.issubset(view.columns):
            living=view.groupby("Organ")[["Living_Donor_Pct","Deceased_Donor_Pct"]].mean().reset_index().melt(id_vars="Organ",var_name="Donor type",value_name="Mean percent")
            chart(px.bar(living,x="Organ",y="Mean percent",color="Donor type",barmode="stack",title="Living vs deceased donor mix"))
        st.subheader("Filtered records")
        st.dataframe(view,use_container_width=True,hide_index=True)
        st.download_button("Download filtered CSV",view.to_csv(index=False).encode("utf-8"),"organ_donation_filtered.csv","text/csv")

elif page == "Trends & Survival":
    st.title("Trends & Survival")
    if "Year" in df and "Transplants" in df:
        group_options=[c for c in ["Organ","Continent"] if c in df]
        group=st.selectbox("Group trend by",["Overall"]+group_options)
        trend=df.dropna(subset=["Year"]).groupby(["Year"]+([group] if group!="Overall" else []),dropna=False)["Transplants"].sum().reset_index()
        chart(px.line(trend,x="Year",y="Transplants",color=group if group!="Overall" else None,markers=True,title="Annual transplant trend"))
    s1,s2=st.columns(2)
    with s1:
        if {"Organ","Survival_Rate_1Y_Pct"}.issubset(df.columns):
            surv=df.groupby("Organ")[["Survival_Rate_1Y_Pct","Survival_Rate_5Y_Pct"]].mean(numeric_only=True).reset_index() if "Survival_Rate_5Y_Pct" in df else df.groupby("Organ")["Survival_Rate_1Y_Pct"].mean().reset_index()
            chart(px.bar(surv,x="Organ",y=[c for c in ["Survival_Rate_1Y_Pct","Survival_Rate_5Y_Pct"] if c in surv],barmode="group",title="Survival rate by organ"))
    with s2:
        if {"Transplants","Survival_Rate_5Y_Pct"}.issubset(df.columns):
            chart(px.scatter(sample(df[["Transplants","Survival_Rate_5Y_Pct","Organ"]].dropna()),x="Transplants",y="Survival_Rate_5Y_Pct",color="Organ",log_x=True,title="Transplant volume vs 5-year survival"))

elif page == "Statistical Tests":
    st.title("Statistical Tests")
    st.caption("Tests use available non-missing observations. Interpret associations as evidence, not proof of causation.")
    if {"Continent","Donor_Rate_PMP"}.issubset(df.columns):
        st.subheader("Donor-rate differences between continents")
        groups=[g["Donor_Rate_PMP"].dropna().values for _,g in df.groupby("Continent") if g["Donor_Rate_PMP"].notna().sum()>1]
        if len(groups)>=2:
            test=st.selectbox("Test",["Kruskal–Wallis (robust non-parametric)","One-way ANOVA"],key="stat_test")
            stat,p=(kruskal(*groups) if test.startswith("Kruskal") else f_oneway(*groups))
            st.write("**H₀:** donor-rate distributions/means do not differ between continents.")
            st.write("**H₁:** at least one continent differs.")
            a,b=st.columns(2); a.metric("Test statistic",f"{stat:.4f}"); b.metric("p-value",f"{p:.6g}")
            st.success("Reject H₀ at α = 0.05." if p<.05 else "Fail to reject H₀ at α = 0.05.")
            chart(px.box(df.dropna(subset=["Continent","Donor_Rate_PMP"]),x="Continent",y="Donor_Rate_PMP",points="outliers",title="Donor rate by continent"))
    if {"Can_Living_Donate","Survival_Rate_5Y_Pct"}.issubset(df.columns):
        st.subheader("Living-donation eligibility and 5-year survival")
        tmp=df[["Can_Living_Donate","Survival_Rate_5Y_Pct"]].dropna()
        labels=tmp["Can_Living_Donate"].astype(str).str.lower()
        groups=[tmp.loc[labels.isin(["yes","true","1"]), "Survival_Rate_5Y_Pct"].values,
                tmp.loc[labels.isin(["no","false","0"]), "Survival_Rate_5Y_Pct"].values]
        if all(len(g)>1 for g in groups):
            stat,p=mannwhitneyu(groups[0],groups[1],alternative="two-sided")
            a,b=st.columns(2); a.metric("Mann–Whitney U",f"{stat:.2f}"); b.metric("p-value",f"{p:.6g}")
            st.write("**H₀:** the two groups have the same survival-rate distribution.")
            st.write("**H₁:** the distributions differ.")
            st.success("Statistically significant at α = 0.05." if p<.05 else "Not statistically significant at α = 0.05.")
            chart(px.box(tmp.assign(Eligibility=labels.values),x="Eligibility",y="Survival_Rate_5Y_Pct",title="5-year survival by living-donation eligibility"))

elif page == "Regression Models":
    st.title("Regression Model Comparison & Prediction")
    st.caption("Train multi-model regression pipelines, monitor execution status, and generate real-time predictions with operational demand classification.")
    
    targets = [c for c in ["Transplants", "Total_Donors", "Waiting_List", "Survival_Rate_5Y_Pct"] if c in numeric_cols]
    if not targets:
        st.warning("No suitable numeric target columns found.")
    else:
        target = st.selectbox("Target variable", targets, key="reg_target_select")
        candidates = [c for c in numeric_cols if c != target]
        default_candidates = [c for c in ["Population_M", "Total_Donors", "Deceased_Rate_PMP", "Waiting_List", "Donor_Rate_PMP"] if c in candidates][:5]
        selected = st.multiselect("Input features", candidates, default=default_candidates, key="reg_features_multiselect")
        
        row_options = [500, 1000, 2000, 5000, 10000]
        desired_rows = min(5000, max(500, len(df)))
        default_rows = min(row_options, key=lambda option: abs(option - desired_rows))
        max_rows = st.select_slider(
            "Maximum rows for model comparison",
            options=row_options,
            value=default_rows,
            help="Caps the number of rows used to keep model training responsive.",
            key="reg_max_rows"
        )
        algos = st.multiselect(
            "Algorithms",
            ["Linear Regression", "Random Forest"],
            default=["Linear Regression", "Random Forest"],
            key="reg_algos"
        )
        
        col_btn, _ = st.columns([1.5, 4])
        with col_btn:
            run_btn = st.button("🚀 Run regression comparison & evaluate", type="primary", use_container_width=True)
            
        if run_btn:
            if not selected or not algos:
                st.error("Select at least one feature and algorithm.")
            else:
                work = df[selected + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
                X = work[selected]
                y = pd.to_numeric(work[target], errors="coerce")
                valid = y.notna()
                X, y = X.loc[valid], y.loc[valid]
                
                if len(X) > max_rows:
                    idx = X.sample(max_rows, random_state=RANDOM_STATE).index
                    X, y = X.loc[idx], y.loc[idx]
                    
                if len(X) < 10 or y.nunique() < 2:
                    st.error("Not enough usable data to train models.")
                else:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                    numpipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
                    pre = ColumnTransformer([("num", numpipe, selected)])
                    
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)
                    }
                    
                    rows = []
                    predictions = {}
                    fitted_pipes = {}
                    
                    with st.status("Training regression models and running predictions...", expanded=True) as status_box:
                        st.write(f"⚙️ Data partitioned: **{len(Xtr):,}** training rows, **{len(Xte):,}** test rows.")
                        for i, name in enumerate(algos):
                            st.write(f"🔄 Training & evaluating **{name}**...")
                            try:
                                pipe = Pipeline([("prep", pre), ("model", models[name])])
                                pipe.fit(Xtr, ytr)
                                pred = pipe.predict(Xte)
                                predictions[name] = pred
                                fitted_pipes[name] = pipe
                                r2 = float(r2_score(yte, pred))
                                rmse = float(np.sqrt(mean_squared_error(yte, pred)))
                                mae = float(mean_absolute_error(yte, pred))
                                rows.append({
                                    "Algorithm": name,
                                    "Prediction Status": "Success (Evaluated)",
                                    "R²": r2,
                                    "RMSE": rmse,
                                    "MAE": mae
                                })
                                st.write(f"✅ **{name}** — R²: `{r2:.3f}`, RMSE: `{rmse:,.2f}`")
                            except Exception as exc:
                                rows.append({
                                    "Algorithm": name,
                                    "Prediction Status": f"Failed ({str(exc)[:50]})",
                                    "R²": np.nan,
                                    "RMSE": np.nan,
                                    "MAE": np.nan
                                })
                                st.write(f"❌ **{name}** failed: {str(exc)[:100]}")
                        status_box.update(label="Prediction Status: Successfully Completed", state="complete", expanded=False)
                    
                    results = pd.DataFrame(rows)
                    
                    # Store run details in session state for persistence across tab/input interactions
                    st.session_state["regression_run"] = {
                        "target": target,
                        "selected": selected,
                        "algos": algos,
                        "results": results,
                        "predictions": predictions,
                        "fitted_pipes": fitted_pipes,
                        "yte": yte,
                        "q25": float(work[target].quantile(0.25)),
                        "q50": float(work[target].quantile(0.50)),
                        "q75": float(work[target].quantile(0.75)),
                        "q90": float(work[target].quantile(0.90)),
                        "min_val": float(work[target].min()),
                        "max_val": float(work[target].max()),
                        "feature_stats": {
                            f: {
                                "min": float(work[f].min()),
                                "max": float(work[f].max()),
                                "median": float(work[f].median()),
                                "q75": float(work[f].quantile(0.75))
                            } for f in selected
                        }
                    }
                    st.session_state.pop("last_interactive_prediction", None)

        # Check if saved results exist in session state
        run_data = st.session_state.get("regression_run")
        if run_data is not None:
            if run_data["target"] != target or set(run_data["selected"]) != set(selected):
                st.info("⚠️ Active settings (target or features) changed since last run. Click **'Run regression comparison & evaluate'** above to retrain and update models.")
            else:
                results = run_data["results"]
                predictions = run_data["predictions"]
                fitted_pipes = run_data["fitted_pipes"]
                yte = run_data["yte"]
                validres = results.dropna(subset=["R²"]) if "R²" in results else pd.DataFrame()
                
                if not validres.empty:
                    best = validres.sort_values("R²", ascending=False).iloc[0]
                    
                    st.markdown("---")
                    # Headline KPI summary metrics
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Prediction Status", "Completed", delta=f"{len(validres)}/{len(algos)} models ready")
                    k2.metric("Best Algorithm", best["Algorithm"], delta=f"R² = {best['R²']:.3f}")
                    k3.metric("Top R² Score", f"{best['R²']:.3f}")
                    k4.metric("Lowest MAE", f"{best['MAE']:,.2f}")
                    
                    tab_models, tab_diag, tab_predict = st.tabs([
                        "📊 Model Comparison & Status",
                        "📈 Residual Diagnostics",
                        "🔮 Interactive Predictor & Status Assessment"
                    ])
                    
                    with tab_models:
                        st.subheader("Model Performance & Prediction Status Leaderboard")
                        st.dataframe(results, use_container_width=True, hide_index=True)
                        chart(px.bar(validres, x="Algorithm", y="R²", title="Model R² comparison", text_auto=".3f", color="Algorithm"), key="reg_r2_chart")
                    
                    with tab_diag:
                        st.subheader("Model Residual Analysis")
                        chosen = st.selectbox("Select model for diagnostics", list(predictions.keys()), key="diag_model_select")
                        if chosen in predictions:
                            diag = pd.DataFrame({"Actual": yte.to_numpy(), "Predicted": predictions[chosen]})
                            diag["Residual"] = diag["Actual"] - diag["Predicted"]
                            x_col, yc_col = st.columns(2)
                            with x_col:
                                chart(px.scatter(diag, x="Actual", y="Predicted", title=f"{chosen}: Actual vs Predicted", trendline="ols"), key="diag_scatter")
                            with yc_col:
                                chart(px.histogram(diag, x="Residual", title=f"{chosen}: Residual Distribution", nbins=30), key="diag_hist")
                                
                    with tab_predict:
                        st.subheader(f"Predict {target} & Assess Operational Status")
                        st.caption("Input feature parameters to generate instant predictions and assess whether the activity level represents routine, balanced, high, or critical operational status.")
                        
                        p_left, p_right = st.columns([1.1, 1.4], gap="large")
                        
                        with p_left:
                            st.markdown("#### 1. Choose Model & Input Parameters")
                            model_names = list(fitted_pipes.keys())
                            best_idx = model_names.index(best["Algorithm"]) if best["Algorithm"] in model_names else 0
                            inf_model = st.selectbox("Predictor Model", model_names, index=best_idx, key="interactive_model_select")
                            
                            preset = st.radio(
                                "Quick Presets",
                                ["Custom Input", "Dataset Median (Balanced)", "High Volume (75th Percentile)"],
                                horizontal=True,
                                key="preset_radio"
                            )
                            
                            user_inputs = {}
                            stats = run_data["feature_stats"]
                            for f in selected:
                                f_stat = stats[f]
                                f_min = f_stat["min"]
                                f_max = f_stat["max"]
                                f_med = f_stat["median"]
                                f_q75 = f_stat["q75"]
                                
                                if preset == "Dataset Median (Balanced)":
                                    default_val = f_med
                                elif preset == "High Volume (75th Percentile)":
                                    default_val = f_q75
                                else:
                                    default_val = f_med
                                    
                                step = (f_max - f_min) / 100.0 if f_max > f_min else 1.0
                                step = round(step, 2) if step < 1 else round(step)
                                if step == 0:
                                    step = 0.01
                                    
                                user_inputs[f] = st.number_input(
                                    f"{f} (Range: {f_min:,.1f} to {f_max:,.1f})",
                                    value=float(default_val),
                                    step=float(step),
                                    key=f"num_{f}_{preset}"
                                )
                                
                            predict_btn = st.button("⚡ Generate Prediction & Assess Status", type="primary", use_container_width=True)
                            
                        # Compute prediction either on button click or default initial load
                        if predict_btn or "last_interactive_prediction" not in st.session_state:
                            input_df = pd.DataFrame([user_inputs])
                            try:
                                pred_val = float(fitted_pipes[inf_model].predict(input_df)[0])
                                st.session_state["last_interactive_prediction"] = {
                                    "val": pred_val,
                                    "model": inf_model,
                                    "inputs": user_inputs
                                }
                            except Exception as pred_err:
                                st.error(f"Prediction calculation error: {pred_err}")
                                
                        with p_right:
                            st.markdown("#### 2. Prediction Outcome & Status Assessment")
                            last_pred = st.session_state.get("last_interactive_prediction")
                            if last_pred is not None:
                                pval = last_pred["val"]
                                q25 = run_data["q25"]
                                q50 = run_data["q50"]
                                q75 = run_data["q75"]
                                q90 = run_data["q90"]
                                min_v = run_data["min_val"]
                                max_v = run_data["max_val"]
                                
                                is_survival = "survival" in target.lower()
                                
                                if is_survival:
                                    if pval < q25:
                                        status_title = "Lower Survival Tier (Elevated Risk)"
                                        status_class = "status-critical"
                                        status_desc = f"Predicted survival rate ({pval:.1f}%) is in the lower quartile (≤ {q25:.1f}%). Indicates elevated post-transplant patient vulnerability."
                                        badge_color = "#f87171"
                                    elif pval < q75:
                                        status_title = "Standard / Moderate Survival Outcome"
                                        status_class = "status-moderate"
                                        status_desc = f"Predicted survival rate ({pval:.1f}%) aligns with national interquartile benchmark ({q25:.1f}%–{q75:.1f}%). Expected post-transplant success."
                                        badge_color = "#4ade80"
                                    elif pval < q90:
                                        status_title = "High Survival Outcome"
                                        status_class = "status-low"
                                        status_desc = f"Predicted survival rate ({pval:.1f}%) is in the 75th–90th percentile range. Strong clinical recovery tier."
                                        badge_color = "#5ee7ee"
                                    else:
                                        status_title = "Optimal / Benchmark Survival Outcome"
                                        status_class = "status-low"
                                        status_desc = f"Predicted survival rate ({pval:.1f}%) reaches the top 10% (≥ {q90:.1f}%). Exceptional clinical survival profile."
                                        badge_color = "#5ee7ee"
                                else:
                                    if pval < q25:
                                        status_title = "Low Volume / Routine Demand"
                                        status_class = "status-low"
                                        status_desc = f"Predicted {target} ({pval:,.1f}) is in the lower quartile (≤ {q25:,.1f}). Operational capacity is under low demand pressure."
                                        badge_color = "#5ee7ee"
                                    elif pval < q75:
                                        status_title = "Moderate / Balanced Activity"
                                        status_class = "status-moderate"
                                        status_desc = f"Predicted {target} ({pval:,.1f}) falls within typical interquartile bounds ({q25:,.1f}–{q75:,.1f}). Stable, balanced operational level."
                                        badge_color = "#4ade80"
                                    elif pval < q90:
                                        status_title = "High Activity / Elevated Demand"
                                        status_class = "status-high"
                                        status_desc = f"Predicted {target} ({pval:,.1f}) reaches the upper quartile ({q75:,.1f}–{q90:,.1f}). High transplant/donor volume or waiting list pressure."
                                        badge_color = "#facc15"
                                    else:
                                        status_title = "Critical Pressure / Intensive Load"
                                        status_class = "status-critical"
                                        status_desc = f"Predicted {target} ({pval:,.1f}) exceeds the 90th percentile (≥ {q90:,.1f}). Critical operational strain or acute organ allocation demand."
                                        badge_color = "#f87171"
                                        
                                diff_vs_med = pval - q50
                                pct_vs_med = (diff_vs_med / q50 * 100) if q50 != 0 else 0
                                
                                sm1, sm2 = st.columns(2)
                                sm1.metric(f"Predicted {target}", f"{pval:,.1f}", delta=f"{pct_vs_med:+.1f}% vs Median ({q50:,.1f})")
                                sm2.metric("Inference Engine", last_pred["model"], delta=status_title.split("/")[0].strip())
                                
                                st.markdown(f"""
                                <div class="status-card {status_class}">
                                    <h4>📌 Prediction Status: {status_title}</h4>
                                    <p>{status_desc}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                bounds_check = []
                                for feat_name, feat_val in last_pred["inputs"].items():
                                    f_min = run_data["feature_stats"][feat_name]["min"]
                                    f_max = run_data["feature_stats"][feat_name]["max"]
                                    if feat_val < f_min or feat_val > f_max:
                                        bounds_check.append(f"{feat_name} ({feat_val:,.1f} ∉ [{f_min:,.1f}, {f_max:,.1f}])")
                                        
                                if bounds_check:
                                    st.warning(f"⚠️ **Input Reliability Status:** Extrapolation Warning — input features exceed historical training range: {', '.join(bounds_check)}.")
                                else:
                                    st.success("✅ **Input Reliability Status:** In-Distribution — all input features fall within verified historical training boundaries.")
                                    
                                gauge_max = max(max_v, pval * 1.15) if max_v > 0 else 100
                                gauge_min = min(min_v, pval * 0.85) if min_v < 0 else 0
                                
                                gauge_fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=pval,
                                    number={"valueformat": ",.1f", "font": {"color": "#ffffff", "size": 32}},
                                    title={"text": f"Prediction Position on Historical {target} Scale", "font": {"color": "#eaf3ff", "size": 15}},
                                    gauge={
                                        "axis": {"range": [gauge_min, gauge_max], "tickcolor": "#9fb6cf"},
                                        "bar": {"color": badge_color, "thickness": 0.28},
                                        "bgcolor": "#0d1a2d",
                                        "borderwidth": 1,
                                        "bordercolor": "#244768",
                                        "steps": [
                                            {"range": [gauge_min, q25], "color": "#0d2847"},
                                            {"range": [q25, q75], "color": "#0e352b"},
                                            {"range": [q75, q90], "color": "#42300c"},
                                            {"range": [q90, gauge_max], "color": "#47151b"}
                                        ],
                                        "threshold": {
                                            "line": {"color": "#ffffff", "width": 3},
                                            "thickness": 0.8,
                                            "value": q50
                                        }
                                    }
                                ))
                                gauge_fig.update_layout(
                                    height=250,
                                    margin=dict(l=25, r=25, t=40, b=15),
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)"
                                )
                                st.plotly_chart(gauge_fig, use_container_width=True, key="pred_gauge_chart")
                                st.caption("Gauge bands: Dark Blue (≤ 25th %ile: Low) | Green (25–75th %ile: Balanced) | Amber (75–90th %ile: Elevated) | Red (≥ 90th %ile: Critical). White line = Historical Median.")
elif page == "Classification Models":
    st.title("Classification Model Comparison")
    st.caption("Compare Logistic Regression vs. Random Forest Classifier on living-donor eligibility status (`Can_Living_Donate`).")
    
    target = "Can_Living_Donate"
    if target not in df.columns:
        st.warning(f"Target column '{target}' not found in dataset.")
    else:
        cls_base = df.loc[df[target].notna()].copy()
        y_raw = cls_base[target].astype(str).str.strip().str.capitalize()
        valid = y_raw.notna() & y_raw.ne("") & y_raw.str.lower().ne("nan")
        cls_base = cls_base.loc[valid].copy()
        y_clean = y_raw.loc[valid]
        
        excluded_leakage = [target, "Living_Eligible", "Organ", "Country", "Continent"]
        candidates = [c for c in cls_base.columns if c not in excluded_leakage and pd.api.types.is_numeric_dtype(cls_base[c])]
        
        default_sel = [c for c in ["Population_M", "Deceased_Donors", "Living_Donors", "Total_Donors", "Donor_Rate_PMP", "Deceased_Rate_PMP", "Transplants", "Living_Donor_Pct", "Deceased_Donor_Pct", "Waiting_List", "Survival_Rate_5Y_Pct"] if c in candidates]
        
        col_c1, col_c2 = st.columns([2.5, 1.5])
        with col_c1:
            cls_features = st.multiselect("Input Predictor Features (Leakage-Aware)", candidates, default=default_sel, key="cls_features_select")
        with col_c2:
            cls_algos = st.multiselect("Classification Algorithms", ["Logistic Regression", "Random Forest"], default=["Logistic Regression", "Random Forest"], key="cls_algos_select")
            
        c_btn, _ = st.columns([1.5, 4])
        with c_btn:
            run_cls_btn = st.button("🚀 Run classification comparison", type="primary", use_container_width=True, key="run_cls_btn")
            
        if run_cls_btn:
            if not cls_features or not cls_algos:
                st.error("Select at least one feature and one classification algorithm.")
            elif y_clean.nunique() < 2:
                st.error("Classification requires at least two distinct target classes.")
            else:
                work_cls = cls_base[cls_features].dropna(how="all")
                v_idx = work_cls.index.intersection(y_clean.index)
                X_c = work_cls.loc[v_idx]
                y_c = y_clean.loc[v_idx]
                
                Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_c, y_c, test_size=0.2, random_state=RANDOM_STATE, stratify=y_c)
                
                prep_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
                pre_cls = ColumnTransformer([("num", prep_pipe, cls_features)])
                
                cls_model_dict = {
                    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
                }
                
                cls_rows = []
                cls_preds = {}
                cls_fitted = {}
                
                with st.status("Training classifiers and evaluating prediction metrics...", expanded=True) as cls_status:
                    st.write(f"⚙️ Data split: **{len(Xtr_c):,}** training samples, **{len(Xte_c):,}** test samples.")
                    for name in cls_algos:
                        st.write(f"🔄 Training **{name}**...")
                        try:
                            pipe = Pipeline([("prep", pre_cls), ("model", cls_model_dict[name])])
                            pipe.fit(Xtr_c, ytr_c)
                            pred = pipe.predict(Xte_c)
                            cls_preds[name] = pred
                            cls_fitted[name] = pipe
                            
                            acc = float(accuracy_score(yte_c, pred))
                            f1 = float(f1_score(yte_c, pred, average="macro"))
                            prec = float(precision_score(yte_c, pred, average="macro", zero_division=0))
                            rec = float(recall_score(yte_c, pred, average="macro", zero_division=0))
                            
                            cls_rows.append({
                                "Algorithm": name,
                                "Prediction Status": "Success (Evaluated)",
                                "Accuracy": acc,
                                "Macro-F1": f1,
                                "Precision": prec,
                                "Recall": rec
                            })
                            st.write(f"✅ **{name}** — Accuracy: `{acc*100:.1f}%`, Macro-F1: `{f1:.3f}`")
                        except Exception as exc:
                            cls_rows.append({
                                "Algorithm": name,
                                "Prediction Status": f"Failed ({str(exc)[:50]})",
                                "Accuracy": np.nan,
                                "Macro-F1": np.nan,
                                "Precision": np.nan,
                                "Recall": np.nan
                            })
                            st.write(f"❌ **{name}** failed: {str(exc)[:100]}")
                    cls_status.update(label="Prediction Status: Successfully Completed", state="complete", expanded=False)
                    
                cls_results = pd.DataFrame(cls_rows)
                
                st.session_state["classification_run"] = {
                    "results": cls_results,
                    "preds": cls_preds,
                    "fitted": cls_fitted,
                    "yte": yte_c,
                    "features": cls_features,
                    "algos": cls_algos,
                    "feature_stats": {
                        f: {
                            "min": float(X_c[f].min()),
                            "max": float(X_c[f].max()),
                            "median": float(X_c[f].median()),
                            "q75": float(X_c[f].quantile(0.75))
                        } for f in cls_features
                    }
                }
                st.session_state.pop("last_interactive_cls_pred", None)
                
        # Render persisted classification results
        c_run = st.session_state.get("classification_run")
        if c_run is not None:
            if set(c_run["features"]) != set(cls_features):
                st.info("⚠️ Active feature selection changed. Click **'Run classification comparison'** to retrain models.")
            else:
                c_results = c_run["results"]
                c_preds = c_run["preds"]
                c_fitted = c_run["fitted"]
                yte_c = c_run["yte"]
                valid_cls = c_results.dropna(subset=["Accuracy"]) if "Accuracy" in c_results else pd.DataFrame()
                
                if not valid_cls.empty:
                    best_cls = valid_cls.sort_values("Macro-F1", ascending=False).iloc[0]
                    
                    st.markdown("---")
                    # Metric cards
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Prediction Status", "Completed", delta=f"{len(valid_cls)}/{len(cls_algos)} models ready")
                    k2.metric("Best Classifier", best_cls["Algorithm"], delta=f"F1 = {best_cls['Macro-F1']:.3f}")
                    k3.metric("Highest Accuracy", f"{best_cls['Accuracy']*100:.1f}%")
                    k4.metric("Best Macro-F1", f"{best_cls['Macro-F1']:.3f}")
                    
                    t_leader, t_matrix, t_infer = st.tabs([
                        "📊 Classifier Leaderboard & Metrics",
                        "📉 Confusion Matrix & Diagnostics",
                        "🔮 Interactive Eligibility Status Predictor"
                    ])
                    
                    with t_leader:
                        st.subheader("Model Performance Leaderboard")
                        st.dataframe(c_results, use_container_width=True, hide_index=True)
                        chart(px.bar(valid_cls, x="Algorithm", y=["Accuracy", "Macro-F1"], barmode="group", title="Accuracy vs Macro-F1 Comparison"), key="cls_bar_chart")
                        
                    with t_matrix:
                        st.subheader("Confusion Matrix & Diagnostics")
                        chosen_cls = st.selectbox("Select Model for Diagnostic Matrix", list(c_preds.keys()), key="diag_cls_select")
                        if chosen_cls in c_preds:
                            cm = confusion_matrix(yte_c, c_preds[chosen_cls])
                            unique_labels = sorted(yte_c.unique())
                            fig_cm = px.imshow(cm, x=unique_labels, y=unique_labels, text_auto=True, color_continuous_scale="Blues", title=f"{chosen_cls}: Confusion Matrix")
                            chart(fig_cm, key="cm_plot")
                            
                    with t_infer:
                        st.subheader("Interactive Living Donor Eligibility Inference")
                        st.caption("Input clinical indicators to predict living donation feasibility (`Can_Living_Donate`).")
                        
                        col_in, col_out = st.columns([1.1, 1.4], gap="large")
                        with col_in:
                            inf_cls_model = st.selectbox("Predictor Model", list(c_fitted.keys()), key="cls_inf_model_select")
                            c_inputs = {}
                            c_stats = c_run["feature_stats"]
                            for f in cls_features:
                                f_st = c_stats[f]
                                f_min, f_max, f_med = f_st["min"], f_st["max"], f_st["median"]
                                step = (f_max - f_min) / 100.0 if f_max > f_min else 1.0
                                step = round(step, 2) if step < 1 else round(step)
                                if step == 0: step = 0.01
                                c_inputs[f] = st.number_input(f"{f} (Range: {f_min:,.1f}–{f_max:,.1f})", value=float(f_med), step=float(step), key=f"cin_{f}")
                                
                            predict_cls_btn = st.button("⚡ Predict Eligibility Status", type="primary", use_container_width=True, key="pred_cls_btn")
                            
                        if predict_cls_btn or "last_interactive_cls_pred" not in st.session_state:
                            input_df_cls = pd.DataFrame([c_inputs])
                            try:
                                pred_label = str(c_fitted[inf_cls_model].predict(input_df_cls)[0])
                                prob_vals = c_fitted[inf_cls_model].predict_proba(input_df_cls)[0]
                                prob_map = dict(zip(c_fitted[inf_cls_model].classes_, prob_vals))
                                st.session_state["last_interactive_cls_pred"] = {
                                    "label": pred_label,
                                    "probs": prob_map,
                                    "model": inf_cls_model
                                }
                            except Exception as c_err:
                                st.error(f"Prediction error: {c_err}")
                                
                        with col_out:
                            last_c_pred = st.session_state.get("last_interactive_cls_pred")
                            if last_c_pred is not None:
                                plabel = last_c_pred["label"]
                                probs = last_c_pred["probs"]
                                is_yes = plabel.lower() in ["yes", "true", "1"]
                                
                                if is_yes:
                                    s_title = f"Eligible for Living Donation ({plabel})"
                                    s_class = "status-moderate"
                                    s_desc = "Clinical profile is compatible with living donor programs (e.g. Living Kidney or Partial Liver). Paired and altruistic donations are clinically viable."
                                    s_icon = "🟢"
                                else:
                                    s_title = f"Deceased-Only Status ({plabel})"
                                    s_class = "status-critical"
                                    s_desc = "Living donation is not viable under these parameters. Organ transplants must originate from deceased donor registries."
                                    s_icon = "🔴"
                                    
                                st.markdown(f"""
                                <div class="status-card {s_class}">
                                    <h4>{s_icon} Prediction Status: {s_title}</h4>
                                    <p>{s_desc}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("##### Prediction Class Probabilities")
                                pcols = st.columns(len(probs))
                                for i, (cname, pval) in enumerate(probs.items()):
                                    with pcols[i]:
                                        st.metric(f"Probability ({cname})", f"{pval*100:.1f}%")
                                        
                                st.info("ℹ️ **Clinical Interpretation Note:** Observational machine learning model exercise for data science coursework. Not for direct medical or triage decisions.")


elif page == "Clustering":
    st.title("Country–Organ Performance Clustering")
    st.caption("Identify distinct country-organ performance archetypes using K-Means clustering.")
    features = [c for c in ["Donor_Rate_PMP", "Transplants", "Waiting_List", "Survival_Rate_5Y_Pct"] if c in numeric_cols]
    selected = st.multiselect("Features", features, default=features)
    k = st.slider("Number of clusters (k)", 2, 8, 3)
    max_rows = st.slider("Maximum rows (performance)", 200, 3000, 1500, step=100)
    st.info("Algorithm: **K-Means Clustering**")
    if st.button("Run K-Means clustering", type="primary"):
        if len(selected) < 2:
            st.error("Select at least two numeric features.")
        else:
            work = df[selected].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(df[selected].median(numeric_only=True)).dropna()
            if len(work) > max_rows:
                work = work.sample(max_rows, random_state=RANDOM_STATE)
            if len(work) <= k:
                st.error("Not enough rows for this number of clusters.")
            else:
                scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
                X = scaler.fit_transform(work)
                model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
                labels = model.fit_predict(X)
                result = work.copy()
                result["Cluster"] = labels.astype(str)
                if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
                    st.metric("Silhouette score", f"{silhouette_score(X, labels, sample_size=min(1500, len(labels)), random_state=RANDOM_STATE):.3f}")
                st.dataframe(result.groupby("Cluster")[selected].mean().round(2), use_container_width=True)
                pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
                viz = pd.DataFrame({"PC1": pca[:, 0], "PC2": pca[:, 1], "Cluster": labels.astype(str)})
                chart(px.scatter(viz, x="PC1", y="PC2", color="Cluster", title="Clusters projected onto PCA components"))
                st.download_button("Download cluster assignments", result.to_csv(index=False).encode("utf-8"), "organ_cluster_assignments.csv", "text/csv")

else:
    st.title("About OrganScope")
    st.markdown("""
    **Project:** Global Organ Donation & Transplantation Analytics

    **Purpose**
    - Understand donation and transplant patterns across countries, organs, continents, and years.
    - Explore donor rates, waiting-list pressure, survival outcomes, and data quality.
    - Compare regression models and discover performance profiles through clustering.
    - Use statistical tests to assess whether observed group differences are statistically significant.

    **Methodology:** Data cleaning and exploration → visual analysis → hypothesis testing → regression comparison → clustering.

    **Important interpretation notes**
    - Correlation does not prove causation.
    - `Avg_Wait_Years_US` is explicitly US-labelled in the dataset; do not interpret it as every country's waiting time.
    - Model metrics depend on the selected target, features, sample size, and train/test split.
    """)


# Enhanced top navigation and visual polish
st.markdown("""
<style>
:root {
    --ink: #eaf3ff;
    --muted: #9fb6cf;
    --panel: #10223a;
    --panel-2: #0d1a2d;
    --line: #244768;
    --accent: #5ee7ee;
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #173758 0, #081225 38%, #060d1a 100%);
    color: var(--ink);
}

[data-testid="stHeader"] {
    display: none;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1500px;
    padding-top: 0.35rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1b30 0%, #091324 100%);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] h2 {
    color: var(--accent);
    letter-spacing: .04em;
}

.nav-strip {
    height: 8px;
    margin-bottom: 8px;
    border-bottom: 1px solid #244768;
}

div[data-testid="stSelectbox"] {
    margin: 0 0 18px 0;
}

div[data-testid="stSelectbox"] > div > div {
    background: #10233b;
    border: 1px solid #3b7395;
    border-radius: 11px;
    color: #fff;
    min-height: 46px;
    box-shadow: 0 6px 18px #0003;
    transition: border-color .15s ease, box-shadow .15s ease;
}

div[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--accent);
    box-shadow: 0 8px 22px #1d9db755;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stRadio"] label {
    font-size: 0.92rem;
    color: var(--muted);
    font-weight: 600;
}

/* Prediction status cards */
.status-card {
    border-radius: 14px;
    padding: 18px 22px;
    margin: 14px 0;
    border: 1px solid var(--line);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.status-card h4 {
    margin: 0 0 6px 0;
    font-size: 1.12rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-card p {
    margin: 0;
    font-size: 0.93rem;
    color: var(--muted);
    line-height: 1.45;
}
.status-low {
    background: linear-gradient(135deg, rgba(14, 52, 92, 0.7), rgba(12, 38, 66, 0.9));
    border-color: #2b74b3;
}
.status-low h4 { color: #5ee7ee; }
.status-moderate {
    background: linear-gradient(135deg, rgba(15, 66, 52, 0.7), rgba(11, 46, 36, 0.9));
    border-color: #26997b;
}
.status-moderate h4 { color: #4ade80; }
.status-high {
    background: linear-gradient(135deg, rgba(79, 56, 12, 0.7), rgba(56, 39, 8, 0.9));
    border-color: #d49524;
}
.status-high h4 { color: #facc15; }
.status-critical {
    background: linear-gradient(135deg, rgba(82, 22, 31, 0.7), rgba(59, 15, 22, 0.9));
    border-color: #e04859;
}
.status-critical h4 { color: #f87171; }

[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--panel), var(--panel-2));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 10px 26px #0003;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #fff;
}

.stButton > button,
.stDownloadButton > button {
    border: 1px solid #3b7395;
    border-radius: 10px;
    background: linear-gradient(135deg, #1b5473, #214b87);
    color: #fff;
    font-weight: 700;
    transition: transform .15s ease, box-shadow .15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 7px 18px #1d9db755;
    border-color: var(--accent);
}

[data-testid="stExpander"],
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    background: #0d1b2e88;
}

[data-baseweb="input"],
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {
    background: #10233b;
    border-color: #2a5274;
    border-radius: 9px;
    color: #fff;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    background: #10223a;
    border-radius: 9px 9px 0 0;
    padding: 8px 14px;
    color: var(--muted);
}

.stTabs [aria-selected="true"] {
    background: #1b4d6b;
    color: #fff;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

h1, h2, h3 {
    letter-spacing: -.02em;
}
</style>
""", unsafe_allow_html=True)

