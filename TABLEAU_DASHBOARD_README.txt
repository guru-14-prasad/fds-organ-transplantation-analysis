# Tableau Dashboard Setup — Organ Donation Case Study

## Included
- `organ_donation_case_study.ipynb`: preprocessing, EDA, 15 professional questions/plots, hypothesis tests, ML model, conclusion.
- `organ_donation_cleaned_tableau.csv`: flat data source for Tableau.
- `organ_donation_15_visualizations_dashboard.html`: interactive Plotly dashboard containing the same 15 visualizations.

## Build the native Tableau workbook
1. Open Tableau Desktop/Public and connect to `organ_donation_cleaned_tableau.csv`.
2. Create 15 worksheets matching the question titles in the notebook/dashboard.
3. Use the chart field mappings shown below; set `Year` as a dimension and numeric measures as measures.
4. Add dashboard filters for `Year`, `Continent`, `Country`, and `Organ`; apply to all worksheets.
5. Save as `organ_donation_dashboard.twbx`.

## Worksheet field mapping
1. Donor volume by continent: `Continent` + SUM(`Total_Donors`) — bar.
2. Top countries: `Country` + SUM(`Total_Donors`), Top 15 — horizontal bar.
3. Donor rate by continent: `Continent` + `Donor_Rate_PMP` — box plot or median bar.
4. Living vs deceased donors: `Deceased_Donors` + `Living_Donors` — scatter.
5. Transplants by organ: `Organ` + SUM(`Transplants`) — bar.
6. Waiting list by organ: `Organ` + MEDIAN(`Waiting_List`) — bar.
7. Donors vs transplants: `Total_Donors` vs `Transplants` — scatter.
8. Donor rate vs waiting list: `Donor_Rate_PMP` vs `Waiting_List` — scatter.
9. One-year survival by organ: `Organ` + `Survival_Rate_1Y_Pct` — box plot.
10. Five-year survival by organ: `Organ` + `Survival_Rate_5Y_Pct` — box plot.
11. One-year vs five-year survival: pivot the two survival columns to long format, then violin/box plot.
12. Living donor share by organ: `Organ` + `Living_Donor_Pct` — box plot.
13. Annual transplant trend: `Year` + SUM(`Transplants`) — line.
14. Highest waiting-list-to-transplant ratio: create calculated field `Waiting_List / MAX(1, Transplants)` and show top 20 records — horizontal bar.
15. Eligibility vs living donor share: `Can_Living_Donate` + `Living_Donor_Pct` — box plot.

## Additional useful dashboard elements
- KPI cards: total records, countries, total donors, total transplants, median waiting list.
- Filters: year, continent, organ, country.
- Tooltip fields: country, year, organ, donor counts, waiting list, transplant count, survival rates.
- Add a data-quality note: this is observational data; verify provenance and definitions before policy/clinical use.
