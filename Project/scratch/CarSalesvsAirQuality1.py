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

# Create the animated scatter plot
fig = px.scatter(
    df,
    x="CarSales_total",
    y="PM25_mean",
    animation_frame="Year",
    animation_group="Country",
    size="CarSales_total",  
    color="Country",
    hover_name="Country",
    size_max=60,
    labels={
        "CarSales_total": "Car Sales (units)",
        "PM25_mean": "PM2.5 Concentration (µg/m³)",
    },
)

# Fix layout and axis formatting
fig.update_layout(
    xaxis=dict(
        title="Car Sales (units)",
        tickformat=",.0f",  # Adds commas (e.g., 1,000,000)
        range=[df["CarSales_total"].min() * 0.8, df["CarSales_total"].max() * 1.1],
    ),
    yaxis=dict(
        title="PM2.5 Concentration (µg/m³)",
        range=[0, df["PM25_mean"].max() * 1.2],
    ),
    title=dict(
        text="Car Sales vs PM2.5 (2019–2023) — USA, China, India, Mexico, Brazil",
        font=dict(size=20, family="Segoe UI Semibold", color="#1C1C1C"),
        x=0.5,
        xanchor="center"
    ),
    plot_bgcolor="white",
)

# Optional: make gridlines lighter and easier to read
fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

fig.write_html(OUT, include_plotlyjs="cdn")
print(f"✅ Saved: {OUT}")
