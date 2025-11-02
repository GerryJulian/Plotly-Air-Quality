import plotly.express as px
import plotly.graph_objects as go
import os

# Data (already in percentages)
labels = [
    "Road Dust (66%)",
    "Vehicles (14%)",
    "MSW & Biomass Burning (11%)",
    "Capacity Building (4%)",
    "Construction & Demolition (2%)",
    "Domestic Fuel (1%)",
    "Industries (1%)",
    "Public Outreach (1%)"
]
values = [66, 14, 11, 4, 2, 1, 1, 1]

# Create pie chart
fig = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.35,  # donut style
            textinfo="label",  # show category names with % in label
            hovertemplate="%{label}<br><b>Share:</b> %{value}%<extra></extra>",
            textfont=dict(size=14, family="Segoe UI"),
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color="white", width=2)
            ),
            pull=[0.02]*8
        )
    ]
)

# Layout styling
fig.update_layout(
    title=dict(
        text="NCAP Fund Allocation (2024)",
        font=dict(size=20, family="Segoe UI Semibold", color="#1C1C1C"),
        x=0.5,
        xanchor="center"
    ),
    font=dict(family="Segoe UI", size=13, color="#333333"),
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(t=60, b=20, l=20, r=20),
    width=600
)

fig.update_traces(textposition='outside')

# --- Determine filename and save ---
filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"
output_dir = os.path.join(os.path.dirname(__file__), "../charts")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

# --- Save interactive chart ---
fig.write_html(output_path, include_plotlyjs="cdn")