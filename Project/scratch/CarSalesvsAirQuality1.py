# -*- coding: utf-8 -*-
"""
Storyboard Page 2 – Chart 1
Animated scatter: Car Sales vs PM2.5 (2019–2023)
Countries: USA, China, India, Mexico, Brazil
"""

from pathlib import Path
import pandas as pd
import plotly.express as px

BASE   = Path(__file__).resolve().parents[1]
DATA   = BASE / "dataset"
CHARTS = BASE / "charts"
CHARTS.mkdir(exist_ok=True)

TIDY = DATA / "storyboard2_airquality_cars_tidy_2019_2024.csv"
OUT  = CHARTS / "CarSalesvsAirQuality1.html"

# Load data
df = pd.read_csv(TIDY, low_memory=False)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

for c in ["Population", "CarSales_total", "PM25_mean"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Filter countries and years
target_iso3 = ["USA", "CHN", "IND", "MEX", "BRA"]
df = df[df["ISO3"].isin(target_iso3)]
df = df[df["Year"].between(2019, 2023)]

# Plot
fig = px.scatter(
    df,
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

fig.update_layout(title_x=0.5, legend_title_text="Country", margin=dict(l=40, r=20, t=60, b=40))
fig.update_xaxes(type="log")
fig.update_xaxes(tickformat=".1e")
fig.write_html(OUT, include_plotlyjs="cdn")
print(f"✅ Saved: {OUT}")
