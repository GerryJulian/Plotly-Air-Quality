# -*- coding: utf-8 -*-
"""
Storyboard Page 2 – Chart 2
Grouped Bars: NO₂ concentration by Country and Year (2019–2023)
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
OUT  = CHARTS / "CarSalesvsAirQuality2.html"

# Load data
df = pd.read_csv(TIDY, low_memory=False)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
df["NO2_mean"] = pd.to_numeric(df["NO2_mean"], errors="coerce")

# Filter countries and years
target_iso3 = ["USA", "CHN", "IND", "MEX", "BRA"]
df = df[df["ISO3"].isin(target_iso3)]
df = df[df["Year"].between(2019, 2023)]

# Ensure consistent order
country_order = (
    df.groupby("Country", as_index=False)["NO2_mean"]
      .mean(numeric_only=True)
      .sort_values("NO2_mean", ascending=False)["Country"]
      .tolist()
)

fig = px.bar(
    df,
    x="Country",
    y="NO2_mean",
    color="Year",
    barmode="group",
    category_orders={"Country": country_order},
    template="plotly_white",
    labels={
        "NO2_mean": "NO₂ concentration (μg/m³)",
        "Country": "Country",
        "Year": "Year",
    },
    title="NO₂ Concentration by Country and Year (2019–2023)"
)

fig.update_layout(title_x=0.5, legend_title_text="Year", margin=dict(l=40, r=20, t=60, b=40))
fig.write_html(OUT, include_plotlyjs="cdn")
print(f"✅ Saved: {OUT}")
