# OrganScope: Organ Transplantation & Donor Activity Analysis

Foundation of Data Science (FDS) project combining an interactive Streamlit dashboard, Tableau workbook, Jupyter analysis, cleaned datasets, statistical tests, regression model comparison, and clustering.

## Project overview

OrganScope explores organ donation and transplantation activity across countries, organs, continents, and years. It covers donor rates, donor and transplant volumes, waiting-list pressure, survival measures, data quality, and relationships between numerical variables. The data is observational, so associations should not be interpreted as causal effects.

## Repository contents

| Component | File | Purpose |
|---|---|---|
| Streamlit dashboard | [app.py](app.py) | Main interactive OrganScope application |
| Tableau workbook | [Organ_Donation_Dashboard_OR.twbx](Organ_Donation_Dashboard_OR.twbx) | Native Tableau workbook |
| Tableau data source | [organ_donation_cleaned_tableau.csv](organ_donation_cleaned_tableau.csv) | Flat Tableau data source |
| Jupyter analysis | [organ_donation_notebook.ipynb](organ_donation_notebook.ipynb) | EDA, 15 questions/visualizations, hypothesis tests, ML analysis, and conclusions |
| HTML dashboard | [organ_donation_15_visualizations_dashboard.html](organ_donation_15_visualizations_dashboard.html) | Standalone Plotly dashboard |
| Application dataset | [final organ dataset.csv](final%20organ%20dataset.csv) | CSV loaded by the app when present |
| Cleaned dataset | [final_organ_clean.csv](final_organ_clean.csv) | Cleaned organ-level data |
| Source download | [data/GODT_downloads.xlsx](data/GODT_downloads.xlsx) | Source workbook retained in data/ |
| Tableau notes | [TABLEAU_DASHBOARD_README.txt](TABLEAU_DASHBOARD_README.txt) | Field mapping and dashboard setup |

## Streamlit dashboard

Run app.py to open OrganScope | Donation Analytics. The sidebar provides:

- Overview — headline metrics, dataset preview, organ totals, continent donor-rate summaries, missing-value review, and a numeric correlation heatmap.
- Data Explorer — numeric distributions, category counts, descriptive statistics, missing values, and IQR outlier counts.
- Country & Organ Analysis — country, organ, and continent filters; donor-rate and transplant summaries; donor-type comparisons; filtered CSV download.
- Trends & Survival — annual transplant trends grouped overall, by organ, or by continent, plus survival views when fields are available.
- Statistical Tests — Kruskal–Wallis or one-way ANOVA for donor-rate differences between continents, and Mann–Whitney U for eligibility versus five-year survival when required columns exist.
- Regression Models — Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, and Gradient Boosting comparisons with R², RMSE, MAE, and residual diagnostics.
- Clustering — K-Means or Agglomerative Clustering, cluster summaries, silhouette score when calculable, PCA projection, and downloadable assignments.
- About Project — project purpose and interpretation notes.

The app trims text fields and converts numeric fields on load. Plot and model row caps keep interaction responsive; metrics depend on selected fields, available data, and train/test sampling.

## Setup and run locally

### Clone

~~~bash
git clone https://github.com/guru-14-prasad/fds-organ-transplantation-analysis.git
cd fds-organ-transplantation-analysis
~~~

### Create an environment and install dependencies

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

macOS/Linux:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

~~~bash
pip install -r requirements.txt
~~~

The requirements cover Streamlit, Pandas, NumPy, Plotly, SciPy, and scikit-learn.

### Launch the dashboard

~~~bash
streamlit run app.py
~~~

Open the local URL printed by Streamlit, normally http://localhost:8501. Keep the application CSV in the repository root so app.py can find it.

## Jupyter analysis

Open [organ_donation_notebook.ipynb](organ_donation_notebook.ipynb) in Jupyter Notebook, JupyterLab, or VS Code. Run cells from top to bottom after confirming that referenced data files exist at the paths expected by the notebook.

The notebook documents the exploratory workflow, 15 analysis questions and plots, hypothesis tests, an ML model, and conclusions. It complements the interactive app.

## Tableau workbook

1. Open [Organ_Donation_Dashboard_OR.twbx](Organ_Donation_Dashboard_OR.twbx) in Tableau Desktop or Tableau Public.
2. If Tableau requests a source, use [organ_donation_cleaned_tableau.csv](organ_donation_cleaned_tableau.csv).
3. Use the filters and worksheet views in the workbook.
4. Consult [TABLEAU_DASHBOARD_README.txt](TABLEAU_DASHBOARD_README.txt) for worksheet field mapping.

The Tableau documentation covers donor volume/rates, country rankings, donor mix, transplants by organ, waiting lists, donor/transplant relationships, survival, annual trends, waiting-list-to-transplant ratio, and eligibility versus living-donor share.

## Dataset notes

The project includes country-year and organ-year analysis. Existing project documentation describes cleaned files with country, regional, population, donor, utilization, transplant, organ, and rate fields. It records 3,104 country-year rows across 194 countries for 2010–2025 and 21,728 organ-year rows; missingness can vary by field.

The dashboard uses fields such as Country, Organ, Continent, Year, Total_Donors, Transplants, Donor_Rate_PMP, Waiting_List, Survival_Rate_1Y_Pct, and Survival_Rate_5Y_Pct when present. Inspect the data-quality views before comparisons.

Important interpretation notes:

- Counts and rates per million answer different questions.
- Regional comparisons can be affected by country coverage and missing values.
- Avg_Wait_Years_US is explicitly US-labelled and should not be read as every country’s waiting time.
- Statistical tests use available non-missing observations.
- Regression and clustering outputs are exploratory, not fixed project results.

## Visual coverage and screenshots

The project covers dataset dimensions, regional donor-rate comparisons, organ activity, donor/transplant relationships, utilization, rates per million, temporal trends, high-donor-activity patterns, probability and sampling concepts, confidence intervals, sampling variability, correlation, regression, and clustering.

> Screenshot placeholder — add an OrganScope Overview capture when available.

> Screenshot placeholder — add a Tableau workbook/dashboard capture when available.

## Limitations and license

This observational analysis is not clinical or policy guidance. Verify definitions, provenance, coverage, and missing-data assumptions before reuse. Correlation and model performance do not establish causation, and outputs can change with filters, preprocessing, and train/test samples.

No license file is currently included. Add a license before distributing the project for reuse.
