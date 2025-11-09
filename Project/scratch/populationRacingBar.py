import pandas as pd
import plotly.express as px
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
charts_dir = os.path.join(base_dir, "../charts")
os.makedirs(charts_dir, exist_ok=True)
output_path = os.path.join(charts_dir, "populationRacingBar.html")


df = pd.read_csv("../dataset/globalPopulation.csv", sep=';')


df_melted = df.melt(id_vars=["Country Name"], var_name="Year", value_name="Population")
df_melted["Year"] = df_melted["Year"].astype(int)


df_melted = df_melted[df_melted["Country Name"].str.lower() != "world"]


highlight_countries = ["China", "India", "United States", "Brazil", "Mexico"]
df_melted["Highlight"] = df_melted["Country Name"].apply(lambda x: "Selected Countries" if x in highlight_countries else "Other")


df_melted['Rank'] = df_melted.groupby('Year')['Population'].rank(method='first', ascending=False)
df_top = df_melted[df_melted['Rank'] <= 20]  


fig = px.bar(
    df_top,
    x='Population',
    y='Country Name',
    color='Highlight',
    text='Rank',
    orientation='h',
    animation_frame='Year',
    range_x=[0, df_top['Population'].max() * 1.1],  
    color_discrete_map={"Selected Countries": "blue", "Other": "lightgray"},
)


fig.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    title={
        'text': "Top Populated Countries Over Time",
        'x': 0.5,        
        'y': 0.95,        
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=20, color='#1e3a8a', family='Arial')
    },
    xaxis_title="Population",
    yaxis_title="Country",
    autosize=True,
    margin=dict(l=100, r=50, t=80, b=50),
)



fig.update_traces(textposition='outside')


fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
print(f"✅ Racing bar chart saved to: {output_path}")


fig.show()
