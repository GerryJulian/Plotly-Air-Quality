import pandas as pd
import plotly.graph_objects as go
import os

# --- Paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))  # current script directory
charts_dir = os.path.join(base_dir, "../charts")       # target folder
os.makedirs(charts_dir, exist_ok=True)                 # create if not exists
output_path = os.path.join(charts_dir, "populationMap.html")

# --- Load dataset ---
df = pd.read_csv("../dataset/globalPopulation.csv", sep=';')

# --- Prepare data ---
df_melted = df.melt(id_vars=["Country Name"], var_name="Year", value_name="Population")
df_melted["Year"] = df_melted["Year"].astype(int)

# --- Create map frames for each year ---
years = sorted(df_melted["Year"].unique())
frames = []

for year in years:
    df_year = df_melted[df_melted["Year"] == year]
    frames.append(
        go.Choropleth(
            locations=df_year["Country Name"],
            locationmode="country names",
            z=df_year["Population"],
            colorscale="Viridis",
            colorbar=dict(title="Population", x=1.02),  # move colorbar slightly right
            zmin=df_melted["Population"].min(),
            zmax=df_melted["Population"].max(),
            visible=False  # will be controlled by dropdown
        )
    )

# Make the first frame visible
frames[0].visible = True

# --- Create dropdown buttons ---
dropdown_buttons = []
for i, year in enumerate(years):
    visibility = [False] * len(years)
    visibility[i] = True
    dropdown_buttons.append(
        dict(
        label=str(year),
        method="update",
        args=[
            {"visible": visibility},  # update traces visibility
            {
                "title": {
                    "text": f"Global Population in {year}",
                    "x": 0.5,
                    "y": 0.95,
                    "xanchor": "center",
                    "yanchor": "top"
                }
            }
        ],
    )
    )

# --- Create figure ---
fig = go.Figure(data=frames)

fig.update_layout(
    title=dict(
        text=f"Global Population in {years[0]}",  
        x=0.5,       
        y=0.95,       
        xanchor='center',
        yanchor='top',
        font=dict(size=20)  
    ),
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type="equirectangular",
    ),
    updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=1.02,  # position above color legend
            xanchor="left",
            y=1.1,     # top alignment
            yanchor="top",
            pad=dict(t=0, r=0),
        )
    ],
    autosize=True,
    margin=dict(l=0, r=0, t=50, b=0),  # adjust margins for responsiveness
)

# --- Save responsive HTML ---
fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
print(f"✅ Responsive population map saved to: {output_path}")

# --- Show the figure ---
fig.show()
