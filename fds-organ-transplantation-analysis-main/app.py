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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

st.set_page_config(page_title="OrganScope | Donation Analytics", page_icon="🫀", layout="wide")

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

with st.sidebar:
    st.markdown("## 🫀 OrganScope")
    st.caption("Global Organ Donation & Transplantation Analytics")
    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "Overview", "Data Explorer", "Country & Organ Analysis",
        "Trends & Survival", "Statistical Tests", "Regression Models",
        "Clustering", "About Project"
    ])
    st.markdown("---")
    st.success(f"{len(df):,} records · {len(df.columns)} fields")
    st.caption(f"Dataset: {DATA_PATH.name}")

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
            test=st.radio("Test",["Kruskal–Wallis (robust non-parametric)","One-way ANOVA"],horizontal=True)
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
    st.title("Regression Model Comparison")
    targets=[c for c in ["Transplants","Total_Donors","Waiting_List","Survival_Rate_5Y_Pct"] if c in numeric_cols]
    if not targets:
        st.warning("No suitable numeric target columns found.")
    else:
        target=st.selectbox("Target variable",targets)
        candidates=[c for c in numeric_cols if c!=target]
        selected=st.multiselect("Input features",candidates,default=[c for c in ["Population_M","Total_Donors","Deceased_Rate_PMP","Waiting_List","Donor_Rate_PMP"] if c in candidates][:5])
        row_options = [500, 1000, 2000, 5000, 10000]
        # select_slider's value must be one of its options (not the exact row count).
        desired_rows = min(5000, max(500, len(df)))
        default_rows = min(row_options, key=lambda option: abs(option - desired_rows))
        max_rows = st.select_slider(
            "Maximum rows for model comparison",
            options=row_options,
            value=default_rows,
            help="Caps the number of rows used to keep model training responsive."
        )
        algos=st.multiselect("Algorithms",["Linear Regression","Ridge","Lasso","Decision Tree","Random Forest","Gradient Boosting"],default=["Linear Regression","Ridge","Decision Tree","Random Forest"])
        if st.button("Run regression comparison",type="primary"):
            if not selected or not algos: st.error("Select at least one feature and algorithm.")
            else:
                work=df[selected+[target]].replace([np.inf,-np.inf],np.nan).dropna(subset=[target])
                X=work[selected]; y=pd.to_numeric(work[target],errors="coerce")
                valid=y.notna(); X,y=X.loc[valid],y.loc[valid]
                if len(X)>max_rows:
                    idx=X.sample(max_rows,random_state=RANDOM_STATE).index; X,y=X.loc[idx],y.loc[idx]
                if len(X)<10 or y.nunique()<2: st.error("Not enough usable data to train models.")
                else:
                    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE)
                    numpipe=Pipeline([("imputer",SimpleImputer(strategy="median"))])
                    pre=ColumnTransformer([("num",numpipe,selected)])
                    models={"Linear Regression":LinearRegression(),"Ridge":Ridge(),"Lasso":Lasso(max_iter=3000),
                            "Decision Tree":DecisionTreeRegressor(max_depth=10,random_state=RANDOM_STATE),
                            "Random Forest":RandomForestRegressor(n_estimators=100,max_depth=15,n_jobs=-1,random_state=RANDOM_STATE),
                            "Gradient Boosting":GradientBoostingRegressor(random_state=RANDOM_STATE)}
                    rows=[]; predictions={}
                    bar=st.progress(0); status=st.empty()
                    for i,name in enumerate(algos):
                        try:
                            pipe=Pipeline([("prep",pre),("model",models[name])])
                            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); predictions[name]=pred
                            rows.append({"Algorithm":name,"R²":r2_score(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred)),"MAE":mean_absolute_error(yte,pred)})
                        except Exception as exc: rows.append({"Algorithm":name,"Error":str(exc)[:200]})
                        bar.progress((i+1)/len(algos)); status.write(f"Evaluated {name}")
                    results=pd.DataFrame(rows); st.dataframe(results,use_container_width=True,hide_index=True)
                    validres=results.dropna(subset=["R²"]) if "R²" in results else pd.DataFrame()
                    if not validres.empty:
                        best=validres.sort_values("R²",ascending=False).iloc[0]
                        st.success(f"Best model by R²: {best['Algorithm']} (R² = {best['R²']:.3f})")
                        chart(px.bar(validres,x="Algorithm",y="R²",title="Model R² comparison",text_auto=".3f"))
                        chosen=st.selectbox("Residual diagnostics model",list(predictions))
                        diag=pd.DataFrame({"Actual":yte.to_numpy(),"Predicted":predictions[chosen]})
                        diag["Residual"]=diag["Actual"]-diag["Predicted"]
                        x,yc=st.columns(2)
                        with x: chart(px.scatter(diag,x="Actual",y="Predicted",title="Actual vs predicted"))
                        with yc: chart(px.histogram(diag,x="Residual",title="Residual distribution",nbins=30))

elif page == "Clustering":
    st.title("Country–Organ Performance Clustering")
    features=[c for c in ["Donor_Rate_PMP","Transplants","Waiting_List","Survival_Rate_5Y_Pct"] if c in numeric_cols]
    selected=st.multiselect("Features",features,default=features)
    k=st.slider("Number of clusters",2,8,3)
    max_rows=st.slider("Maximum rows (performance)",200,3000,1500,step=100)
    algo=st.radio("Algorithm",["K-Means","Agglomerative Clustering"],horizontal=True)
    if st.button("Run clustering",type="primary"):
        if len(selected)<2: st.error("Select at least two numeric features.")
        else:
            work=df[selected].replace([np.inf,-np.inf],np.nan).dropna(how="all").fillna(df[selected].median(numeric_only=True)).dropna()
            if len(work)>max_rows: work=work.sample(max_rows,random_state=RANDOM_STATE)
            if len(work)<=k: st.error("Not enough rows for this number of clusters.")
            else:
                scaler=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
                X=scaler.fit_transform(work)
                model=KMeans(n_clusters=k,n_init=10,random_state=RANDOM_STATE) if algo=="K-Means" else AgglomerativeClustering(n_clusters=k)
                labels=model.fit_predict(X)
                result=work.copy(); result["Cluster"]=labels.astype(str)
                if len(np.unique(labels))>1 and len(np.unique(labels))<len(labels):
                    st.metric("Silhouette score",f"{silhouette_score(X,labels,sample_size=min(1500,len(labels)),random_state=RANDOM_STATE):.3f}")
                st.dataframe(result.groupby("Cluster")[selected].mean().round(2),use_container_width=True)
                pca=PCA(n_components=2,random_state=RANDOM_STATE).fit_transform(X)
                viz=pd.DataFrame({"PC1":pca[:,0],"PC2":pca[:,1],"Cluster":labels.astype(str)})
                chart(px.scatter(viz,x="PC1",y="PC2",color="Cluster",title="Clusters projected onto PCA components"))
                st.download_button("Download cluster assignments",result.to_csv(index=False).encode("utf-8"),"organ_cluster_assignments.csv","text/csv")

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
