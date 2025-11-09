import plotly.graph_objects as go
import pandas as pd
import os

os.makedirs("../charts", exist_ok=True)
df_cars = pd.read_csv("../dataset/carSales.csv", sep=';')

countries = sorted(df_cars['Entity'].unique())
years = sorted(df_cars['Year'].unique())

def get_values(country, year):
    row = df_cars[(df_cars['Entity'] == country) & (df_cars['Year'] == year)]
    if not row.empty:
        return [row['Electric cars sold'].values[0], row['Non-electric car sales'].values[0]]
    else:
        return [0, 0]

initial_country = countries[0]
initial_year = years[0]
initial_values = get_values(initial_country, initial_year)

fig = go.Figure(
    go.Pie(
        labels=['Electric Cars', 'Non-Electric Cars'],
        values=initial_values,
        marker=dict(colors=['#1f77b4', '#ff7f0e']),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Sales: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>',
        domain=dict(x=[0.2, 0.8], y=[0.20, 0.88])
    )
)

TITLE_Y = 0.95
DROPDOWN_Y = 1.2  # under the title

# Build a slider factory for any country
def make_slider_for(country):
    steps = []
    for yr in years:
        steps.append(dict(
            method="update",
            args=[
                {"values": [get_values(country, yr)]},
                {"title": {"text": f"Car Sales Distribution - {country} ({yr})",
                           "x": 0.5, "y": TITLE_Y, "xanchor": "center", "yanchor": "top"}}
            ],
            label=str(yr)
        ))
    return dict(
        active=0,
        yanchor="top",
        y=0.10,
        xanchor="center",
        currentvalue=dict(prefix=f"Year ({country}): ", visible=True, xanchor="center"),
        pad=dict(b=10, t=30),
        len=0.9,
        x=0.5,
        steps=steps
    )

# Initial slider
slider = make_slider_for(initial_country)

# Country dropdown buttons: update values, title AND replace the slider with one tied to that country
dropdown_buttons = []
for country in countries:
    dropdown_buttons.append(dict(
        label=country,
        method="update",
        args=[
            {"values": [get_values(country, years[0])]},
            {
                "title": {"text": f"Car Sales Distribution - {country} ({years[0]})",
                          "x": 0.5, "y": TITLE_Y, "xanchor": "center", "yanchor": "top"},
                "sliders": [make_slider_for(country)],  # rebuild slider for this country
            }
        ]
    ))

updatemenus = [
    dict(
        buttons=dropdown_buttons,
        direction="down",
        pad={"r": 4, "t": 2, "b": 0, "l": 4},
        showactive=True,
        x=0.5, xanchor="center",
        y=DROPDOWN_Y, yanchor="top",
        type="dropdown"
    )
]

fig.update_layout(
    title=dict(text=f'Car Sales Distribution - {initial_country} ({initial_year})',
               x=0.5, y=TITLE_Y, xanchor='center', yanchor='top'),
    sliders=[slider],
    updatemenus=updatemenus,
    autosize=True,
    margin=dict(l=60, r=60, t=90, b=70),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5),
)

output_file = "../charts/pieCarSales.html"
fig.write_html(
    output_file,
    include_plotlyjs="cdn",
    full_html=True,
    config={"responsive": True}
)
print(f"Chart exported to: {output_file}")
