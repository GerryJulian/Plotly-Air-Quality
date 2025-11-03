# -*- coding: utf-8 -*-
"""
Storyboard Page 2 – Chart 3
Line Trends: EV Share by Country (2019–2023)
Countries: USA, China, India, Mexico, Brazil
"""

from pathlib import Path
import pandas as pd
import plotly.express as px

BASE   = Path(__file__).resolve().parents[1]/"Project"
DATA   = BASE / "dataset"
CHARTS = BASE / "charts"
CHARTS.mkdir(exist_ok=True)

TIDY = DATA / "storyboard2_airquality_cars_tidy_2019_2024.csv"
OUT  = CHARTS / "CarSalesvsAirQuality3.html"

# Load data
df = pd.read_csv(TIDY, low_memory=False)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
df["EV_share"] = pd.to_numeric(df["EV_share"], errors="coerce")

# Filter countries and years
target_iso3 = ["USA", "CHN", "IND", "MEX", "BRA"]
df = df[df["ISO3"].isin(target_iso3)]
df = df[df["Year"].between(2019, 2023)]

# Plot
fig = px.line(
    df,
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

fig.update_layout(title_x=0.5, legend_title_text="Country", margin=dict(l=40, r=20, t=60, b=40))
fig.write_html(OUT, include_plotlyjs="cdn")
print(f"✅ Saved: {OUT}")
