# Organ Transplantation and Donor Activity Analysis

## Foundation of Data Science (FDS)

A data-driven analysis of organ transplantation activity, deceased donor rates, regional differences, and trends over time using Python, statistical methods, and Tableau.

## Project Overview

Organ transplantation is an important component of healthcare systems worldwide. Analyzing donor activity and transplant rates can help us understand regional differences, organ-specific activity, and changes in transplantation over time.

This project applies Foundation of Data Science concepts to cleaned country-year and organ-year datasets. It combines exploratory data analysis, statistical reasoning, sampling concepts, and data visualization to investigate patterns in organ donation and transplantation.

## Objectives

- Examine the dimensions and coverage of the datasets.
- Compare donor rates and utilization across regions.
- Identify organs with the highest total transplant activity and transplantation rates per million.
- Investigate relationships between donor activity and transplant activity.
- Analyze changes in transplantation over time.
- Study regional patterns in high donor activity.
- Apply probability, confidence intervals, sampling variability, and sampling-distribution concepts.
- Explore relationships between numerical variables using correlation analysis.
- Communicate findings through clear visualizations and statistical interpretation.

## Dataset Description

The project uses two cleaned datasets:

| Dataset | Description |
|---|---|
| `country_year_clean.csv` | Country-year observations containing regional information, population, donor activity, utilization, transplant counts, and rates. |
| `organ_year_clean.csv` | Organ-year observations containing organ-specific transplant counts and transplantation rates. |

The datasets are used for different parts of the analysis. The country-year dataset supports regional, temporal, and donor-related analysis, while the organ-year dataset supports comparisons across organ categories.

**Dataset coverage:** The supplied cleaned files contain 3,104 country-year rows representing 194 countries across 2010–2025, and 21,728 organ-year rows. Actual coverage can vary by variable because some observations may be missing.

## Research Questions

The project investigates the following 15 questions:

1. What are the dataset dimensions and coverage?
2. Do donor rates differ across regions?
3. Which organ has the greatest transplant activity?
4. Is donor activity associated with transplant activity?
5. How does donor utilization vary by region?
6. Are actual donors associated with total transplants?
7. Which organs have the highest transplant rate per million?
8. How has transplantation changed over time?
9. Is region associated with high donor activity?
10. How does transplant activity vary across organs?
11. What is the probability of high-transplant records?
12. How precise is the mean donor-rate estimate?
13. How does sample size affect variability?
14. Does the sampling distribution approach normality?
15. Which variables best explain transplant activity?

## Tools and Technologies

- **Python** — Data processing, analysis, and statistical calculations.
- **Pandas and NumPy** — Data manipulation and numerical operations.
- **Matplotlib** — Data visualization.
- **Tableau** — Interactive charts and dashboards.
- **Jupyter Notebook** — Analysis workflow and documentation.
- **Git and GitHub** — Version control and project collaboration.
- **Microsoft PowerPoint** — Presentation of methodology, visualizations, and findings.

## Methodology

The project follows these stages:

1. **Data inspection:** Examine dataset dimensions, columns, data types, missing values, and coverage.
2. **Data preparation:** Use the supplied cleaned datasets and identify relevant fields for each question.
3. **Exploratory analysis:** Summarize donor rates, utilization, transplant counts, organ categories, and temporal patterns.
4. **Visualization:** Build bar charts, box plots, scatterplots, line charts, contingency heatmaps, and correlation heatmaps.
5. **Statistical analysis:** Apply suitable probability and sampling methods, confidence intervals, and association measures where appropriate.
6. **Interpretation:** Explain findings while considering uncertainty, missing data, and limitations.
7. **Presentation:** Communicate the analysis through Tableau visualizations and a final project presentation.

## Visualizations

The planned visualizations include:

- Dataset coverage and regional record counts
- Donor-rate distributions by region
- Total transplants by organ
- Donor-rate versus transplant-rate scatterplots
- Regional utilization comparisons
- Actual deceased donors versus total transplants
- Organ-specific transplantation rates per million
- Transplantation trends over time
- Regional high-donor-activity heatmaps
- Organ-level transplant-rate distributions
- Binomial probability mass function
- Confidence intervals for mean donor rates
- Sampling-size and variability comparisons
- Sampling-distribution histograms for different sample sizes
- Correlation heatmaps for numerical variables

## Key Considerations

- Transplant counts and transplantation rates per million measure different aspects of activity.
- Regional comparisons should account for differences in country coverage and missing values.
- Correlation indicates association, not causation.
- Sampling simulations illustrate
