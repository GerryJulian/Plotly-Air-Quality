import pandas as pd
import plotly.express as px
import plotly.io as pio
import os


df_population = pd.read_csv("../dataset/globalPopulation.csv", sep=';')
df_car_sales  = pd.read_csv("../dataset/carSales.csv", sep=';')  


df_pop_long = pd.melt(
    df_population,
    id_vars=['Country Name'],
    value_vars=['2019','2020','2021','2022','2023','2024'],
    var_name='Year',
    value_name='Population'
)  
df_pop_long['Year'] = df_pop_long['Year'].astype(int) 
df_pop_long['Population (Millions)'] = df_pop_long['Population'] / 1_000_000  


merged_data = pd.merge(
    df_car_sales,
    df_pop_long,
    left_on=['Entity','Year'],
    right_on=['Country Name','Year'],
    how='inner'
)  
merged_data['Total Sales (Millions)'] = merged_data['Total Sales'] / 1_000_000  # x in millions [web:109]

countries = ['China','United States','India','Brazil','Mexico']
merged_data = merged_data[merged_data['Entity'].isin(countries)]  # subset [web:50]


MANUAL_XMIN, MANUAL_XMAX = 0.0, 30.0
MANUAL_YMIN, MANUAL_YMAX = 0.0, 1200.0


fig = px.scatter(
    merged_data,
    x="Total Sales (Millions)",
    y="Population (Millions)",
    size="Population (Millions)",
    color="Entity",
    hover_name="Entity",
    animation_frame="Year",
    animation_group="Entity",
    size_max=50,
    title="Population vs Car Sales (2019–2024)"
)  
fig.update_layout(
    autosize=True,
    margin=dict(l=50, r=50, t=80, b=80),
    title=dict(text="Population vs Car Sales (2019–2024)", x=0.5, font=dict(size=18)),
    hovermode="closest",
    legend=dict(x=1.02, y=0.92, bgcolor="rgba(255,255,255,0.9)", bordercolor="lightgray", title=dict(text="Countries")),
    xaxis=dict(title="Car Sales (Millions)", range=[MANUAL_XMIN, MANUAL_XMAX], autorange=False),
    yaxis=dict(title="Population (Millions)", range=[MANUAL_YMIN, MANUAL_YMAX], autorange=False),
)  
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>Car Sales: %{x:.2f} M<br>Population: %{y:.2f} M<br><extra></extra>"
)  
for fr in fig.frames:
    fr.layout = dict(
        xaxis=dict(range=[MANUAL_XMIN, MANUAL_XMAX], autorange=False),
        yaxis=dict(range=[MANUAL_YMIN, MANUAL_YMAX], autorange=False)
    )  


base_dir = os.path.dirname(os.path.abspath(__file__))
charts_dir = os.path.join(base_dir, "../charts")
os.makedirs(charts_dir, exist_ok=True)  

script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(charts_dir, f"{script_name}.html")

pio.write_html(
    fig,
    file=output_path,
    include_plotlyjs='cdn',   
    full_html=True,
    config={'responsive': True, 'displayModeBar': True, 'displaylogo': False}
)  

print(f"✅ Exported to: {output_path}")

