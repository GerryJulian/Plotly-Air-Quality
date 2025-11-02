import plotly.express as px
import plotly.graph_objects as go
import os

# Data
labels = [
    "World Bank Loan (US$208M)",
    "Huaxia Commercial Bank (US$430M)",
    "Sub-borrowers (US$662M)"
]
values = [208, 430, 662]

# Create figure
fig = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.35,  # donut style
            textinfo="label+percent",
            textfont=dict(size=14, family="Segoe UI"),
            hovertemplate="%{label}<br><b>Share:</b> %{percent}<extra></extra>",
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color="white", width=2)
            ),
            pull=[0.03, 0.02, 0.01]  # subtle separation effect
        )
    ]
)

# Layout configuration
fig.update_layout(
    title=dict(
        text="Program Financing Composition (Total: US$1.3 Billion)",
        font=dict(family="Segoe UI Semibold", size=20, color="#1C1C1C"),
        x=0.5,
        xanchor="center"
    ),
    font=dict(family="Segoe UI", size=13, color="#333333"),
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(t=60, b=20, l=20, r=20),
    width=650
)

fig.update_traces(textposition='outside')

# --- Determine filename and save ---
filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"
output_dir = os.path.join(os.path.dirname(__file__), "../charts")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

# --- Save interactive chart ---
fig.write_html(output_path, include_plotlyjs="cdn")