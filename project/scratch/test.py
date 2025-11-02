import plotly.graph_objects as go
import os
import numpy as np

# --- Sample data for 5 countries over 5 years ---
years = [2019, 2020, 2021, 2022, 2023]
countries = ["USA", "China", "India", "Germany", "Brazil"]

# Random example data
np.random.seed(42)
data = {country: np.random.randint(50, 150, size=len(years)) for country in countries}

# --- Create initial traces (first year) ---
fig = go.Figure()

for country in countries:
    fig.add_trace(
        go.Scatter(
            x=[years[0]], 
            y=[data[country][0]],
            mode="lines+markers",
            name=country,
            visible=True  # We'll toggle this with dropdown
        )
    )

# --- Add animation frames (one frame per year) ---
frames = []
for i, year in enumerate(years):
    frame_data = [
        go.Scatter(
            x=years[:i+1], 
            y=[data[country][j] for j in range(i+1)],
            mode="lines+markers",
            name=country
        )
        for country in countries
    ]
    frames.append(go.Frame(data=frame_data, name=str(year)))

fig.frames = frames

# --- Create country filter dropdown ---
dropdown_buttons = [
    {
        "label": "All",
        "method": "update",
        "args": [{"visible": [True]*len(countries)}, {"title": "Country Trends: All"}]
    }
]

for i, country in enumerate(countries):
    visible = [False]*len(countries)
    visible[i] = True
    dropdown_buttons.append(
        {
            "label": country,
            "method": "update",
            "args": [{"visible": visible}, {"title": f"Country Trends: {country}"}]
        }
    )

# --- Layout with animation buttons, slider, and country filter ---
fig.update_layout(
    title="Country Trends Over 5 Years",
    xaxis_title="Year",
    yaxis_title="Value",
    updatemenus=[
        # Animation Play/Pause
        {
            "type": "buttons",
            "showactive": False,
            "y": 1.05,
            "x": 1.15,
            "xanchor": "right",
            "yanchor": "top",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ]
        },
        # Country filter dropdown
        {
            "type": "dropdown",
            "showactive": True,
            "y": 1.05,
            "x": -0.05,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": dropdown_buttons
        }
    ],
    sliders=[{
        "steps": [
            {
                "method": "animate",
                "label": str(year),
                "args": [[str(year)], {"mode": "immediate", "frame": {"duration": 800, "redraw": True}}]
            } for year in years
        ],
        "transition": {"duration": 300},
        "x": 0,
        "y": -0.1,
        "currentvalue": {"prefix": "Year: "}
    }]
)

# --- Determine filename and save ---
filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"
output_dir = os.path.join(os.path.dirname(__file__), "../charts")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

# --- Save interactive chart ---
fig.write_html(output_path, include_plotlyjs="cdn")

print(f"✅ {filename} generated at {output_dir}!")
