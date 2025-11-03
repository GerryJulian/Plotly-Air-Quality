# -*- coding: utf-8 -*-
"""
Storyboard Page 2 – Vehicles & Air Quality (5 countries, 3 different chart types)
Data: dataset/storyboard2_airquality_cars_tidy_2019_2024.csv
Countries: USA, China, India, Mexico, Brazil
Years: 2019–2023
Outputs:
  charts/sb2_plot1_scatter_cars_vs_pm25.html      (animated scatter)
  charts/sb2_plot2_bars_no2_by_year.html          (grouped bars)
  charts/sb2_plot3_lines_evshare_trend.html       (line trends)
"""

from pathlib import Path
import pandas as pd
import plotly.express as px

# ---------------------------
# Paths
# ---------------------------
BASE   = Path(__file__).resolve().parents[1]
DATA   = BASE / "dataset"
CHARTS = BASE / "charts"
CHARTS.mkdir(exist_ok=True)

TIDY = DATA / "storyboard2_airquality_cars_tidy_2019_2024.csv"

OUT1 = CHARTS / "sb2_plot1_scatter_cars_vs_pm25.html"
OUT2 = CHARTS / "sb2_plot2_bars_no2_by_year.html"
OUT3 = CHARTS / "sb2_plot3_lines_evshare_trend.html"

# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(TIDY, low_memory=False)

# Dtypes
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
for c in ["Population", "CarSales_total", "EV_share", "PM25_mean", "NO2_mean"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Focus years & countries
df = df[df["Year"].between(2019, 2023, inclusive="both")]

target_iso3 = ["USA", "CHN", "IND", "MEX", "BRA"]
sub = df[df["ISO3"].isin(target_iso3)].copy()

# Safety hint
missing = [iso for iso in target_iso3 if iso not in set(sub["ISO3"])]
if missing:
    print("⚠️ Missing countries in filtered data:", missing)

# ===========================
# Chart 1 — Animated scatter
# Car sales vs PM2.5 (bubble size = population)
# ===========================
need1 = sub.dropna(subset=["CarSales_total", "PM25_mean", "Population", "Year"]).copy()

fig1 = px.scatter(
    need1,
    x="CarSales_total",
    y="PM25_mean",
    color="Country",
    size="Population",
    size_max=46,
    hover_name="Country",
    animation_frame="Year",
    animation_group="ISO3",
    template="plotly_white",
    labels={
        "CarSales_total": "Car sales (units)",
        "PM25_mean": "PM2.5 concentration (μg/m³)",
        "Population": "Population",
        "Country": "Country",
        "Year": "Year",
    },
    title="Car Sales vs PM2.5 (2019–2023) — USA, China, India, Mexico, Brazil"
)
fig1.update_layout(title_x=0.5, legend_title_text="Country", margin=dict(l=40, r=20, t=60, b=40))
fig1.update_xaxes(type = "log")
fig1.update_xaxes(tickformat=",")  # thousands separator
fig1.write_html(OUT1, include_plotlyjs="cdn")
print("✅ Saved:", OUT1)

# ===========================
# Chart 2 — Grouped bars
# NO₂ by country & year (compare years side-by-side)
# ===========================
need2 = sub.dropna(subset=["NO2_mean", "Year"]).copy()

# Ensure fixed order of countries for consistent x-axis
country_order = (
    need2.groupby("Country", as_index=False)["NO2_mean"]
         .mean(numeric_only=True)
         .sort_values("NO2_mean", ascending=False)["Country"]
         .tolist()
)
fig2 = px.bar(
    need2,
    x="Country",
    y="NO2_mean",
    color="Year",
    barmode="group",
    category_orders={"Country": country_order},
    template="plotly_white",
    labels={
        "NO2_mean": "NO₂ concentration (μg/m³)",
        "Country": "Country",
        "Year": "Year"
    },
    title="NO₂ Concentration by Country and Year (2019–2023)"
)
fig2.update_layout(title_x=0.5, legend_title_text="Year", margin=dict(l=40, r=20, t=60, b=40))
fig2.write_html(OUT2, include_plotlyjs="cdn")
print("✅ Saved:", OUT2)

# ===========================
# Chart 3 — Line trends
# EV share trend by country (2019–2023)
# ===========================
need3 = sub.dropna(subset=["EV_share", "Year"]).copy()

fig3 = px.line(
    need3,
    x="Year",
    y="EV_share",
    color="Country",
    markers=True,
    template="plotly_white",
    labels={
        "EV_share": "EV share (EV / total car sales)",
        "Year": "Year",
        "Country": "Country"
    },
    title="EV Share Trend by Country (2019–2023)"
)
fig3.update_layout(title_x=0.5, legend_title_text="Country", margin=dict(l=40, r=20, t=60, b=40))
fig3.write_html(OUT3, include_plotlyjs="cdn")
print("✅ Saved:", OUT3)

# Uncomment during development to preview interactively:
# fig1.show(); fig2.show(); fig3.show()
