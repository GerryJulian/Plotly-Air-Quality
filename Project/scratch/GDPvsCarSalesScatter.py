import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

# --- Load datasets ---
df_gdp = pd.read_csv("../dataset/tbu-global-gdp-per-capita-by-country.csv")  # wide 2019..2024 [web:50]
df_car = pd.read_csv("../dataset/carSales.csv", sep=';')  # columns match your pasted table [web:50]

# --- GDP wide -> long for animation ---
gdp_long = df_gdp.melt(
    id_vars=['Country Name','Country Code'],
    value_vars=['2019','2020','2021','2022','2023','2024'],
    var_name='Year',
    value_name='GDP Per Capita'
)  # tidy format for animation_frame [web:50]
gdp_long['Year'] = gdp_long['Year'].astype(int)  # numeric slider [web:50]
gdp_long['GDP Per Capita ($)'] = gdp_long['GDP Per Capita']  # display label [web:50]

# --- Merge on country+year ---
df = (
    df_car.rename(columns={'Entity':'Country Name','Code':'Country Code'})
          .merge(gdp_long, on=['Country Name','Country Code','Year'], how='inner')
)  # aligned keys for join [web:50]

# --- Prepare plotting fields ---
df['Total Sales (M)'] = df['Total Sales'] / 1_000_000  # x axis in millions [web:109]

# Optional: limit to focus countries
focus = ['China','United States','India','Brazil','Mexico']
df = df[df['Country Name'].isin(focus)]  # subset for clarity [web:50]

# --- Build animated scatter (Total Sales vs GDP) ---
fig = px.scatter(
    df,
    x='Total Sales (M)',                 # Total car sales
    y='GDP Per Capita ($)',           # GDP per capita
    size='GDP Per Capita ($)',        # bubble size by GDP
    color='Country Name',
    hover_name='Country Name',
    animation_frame='Year',
    animation_group='Country Name',
    size_max=35,
    title='GDP Per Capita vs Total Car Sales (2019–2024)'
)  # animated scatter pattern [web:22][web:109]

# Styling
fig.update_layout(
    autosize=True,
    margin=dict(l=50, r=50, t=80, b=80),
    title=dict(text='GDP Per Capita vs Total Car Sales (2019–2024)', x=0.5, font=dict(size=18)),
    hovermode='closest',
    legend=dict(x=1.02, y=0.92, bgcolor='rgba(255,255,255,0.9)', bordercolor='lightgray', title=dict(text='Countries')),
    xaxis=dict(title='Car Sales (Millions)', range=[-3.0, 38.0], autorange=False, tickformat='.2f'),
    yaxis=dict(title='GDP Per Capita ($)', range=[-8000.0, 98000.0], autorange=False, tickformat=",.2f"),
)  # consistent axes and style [web:50]
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>Car Sales: %{x:.2f} M<br>GDP Per Capita: $%{y:,.2f}<br><extra></extra>"
)  # clean hover text [web:50]

# Keep axis ranges stable across frames
for fr in fig.frames:
    fr.layout = dict(
        xaxis=dict(range=[-3.0, 38.0], autorange=False, tickformat='.2f'),
        yaxis=dict(range=[-8000.0, 98000.0], autorange=False, tickformat=",.2f")
    )  # stable animation ranges [web:24]

# --- Export HTML to ../charts using script name ---
base_dir = os.path.dirname(os.path.abspath(__file__))
charts_dir = os.path.join(base_dir, "../charts")
os.makedirs(charts_dir, exist_ok=True)  # ensure output dir exists [web:54]

script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(charts_dir, f"{script_name}.html")

pio.write_html(
    fig,
    file=output_path,
    include_plotlyjs='cdn',   # smaller file, interactive via CDN [web:55]
    full_html=True,
    config={'responsive': True, 'displayModeBar': True, 'displaylogo': False}
)  # interactive standalone HTML export [web:54]

print(f"✅ Exported to: {output_path}")
# fig.show(config={'responsive': True, 'displayModeBar': True, 'displaylogo': False})
