import plotly.express as px
import pandas as pd
import os

# CMAQ budget data (in $000)
data = {
    "Year": [2019, 2020, 2021, 2022, 2023, 2024],
    "Budget ($000)": [2448516, 2496403, 2493589, 2536491, 2587221, 2638965]
}

# Convert to billions
df = pd.DataFrame(data)
df["Budget (Billions)"] = df["Budget ($000)"] / 1_000_000

# Create the chart
fig = px.bar(
    df,
    x="Year",
    y="Budget (Billions)",
    text="Budget (Billions)",
    color="Budget (Billions)",
    color_continuous_scale="Bluyl",
)

# Style it
fig.update_traces(
    hovertemplate=(
        "<b>Year:</b> %{x}<br>"
        "<b>Budget:</b> %{y:.2f} Billion USD<br>"
        "<extra></extra>"  # removes the trace name from tooltip
    ),
    texttemplate="%{text:.2f}", 
    textposition="outside"
)
fig.update_layout(
    title=dict(
        text="CMAQ Program Budget (2019–2024)",
        font=dict(size=20, family="Segoe UI Semibold", color="#1C1C1C"),
        x=0.5,
        xanchor="center"
    ),
    yaxis_title="Budget (Billions USD)",
    xaxis_title="Year",
    font=dict(family="Segoe UI", size=13, color="#333333"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    title_x=0.5,
    showlegend=False,
    width=650
)

# --- Determine filename and save ---
filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"
output_dir = os.path.join(os.path.dirname(__file__), "../charts")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

# --- Save interactive chart ---
fig.write_html(output_path, include_plotlyjs="cdn")