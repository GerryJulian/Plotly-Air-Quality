import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

def CollectData():
    df_Esales = pd.read_csv("../dataset/tbu-electric-car-sales-by-country.csv")
    df_gdp = pd.read_csv("../dataset/tbu-global-gdp-per-capita-by-country.csv")

    countries = df_Esales['Entity'].unique()
    years = df_Esales['Year'].unique()

    country_order = [1, 2, 5, 4, 3]
    ordered_countries = [countries[i - 1] for i in country_order]

    y_total_sales = np.zeros((len(years), len(countries)))
    y_Esales_data = np.zeros((len(years), len(countries)))
    y_NonEsales = np.zeros((len(years), len(countries)))
    y_GDP = np.zeros((len(years), len(countries)))

    for i, year in enumerate(years):
        for j, country in enumerate(countries):
            esales_data = df_Esales[(df_Esales['Year'] == year) & (df_Esales['Entity'] == country)]
            if esales_data.empty:
                esales_data = df_Esales[(df_Esales['Year'] == year) &
                                        (df_Esales['Entity'].str.contains(
                                            country.split()[0] if ' ' in country else country))]

            if not esales_data.empty:
                y_Esales_data[i][j] = esales_data.iloc[0, 3]
                y_NonEsales[i][j] = esales_data.iloc[0, 4]

            y_total_sales[i][j] = y_NonEsales[i][j] + y_Esales_data[i][j]

            gdp_data = df_gdp[df_gdp['Country Name'] == country]
            if not gdp_data.empty:
                year_col = str(year)
                if year_col in gdp_data.columns:
                    y_GDP[i][j] = gdp_data[year_col].values[0]

    def reorder_data(data, original_countries, ordered_countries):
        reordered_data = np.zeros_like(data)
        for i, country in enumerate(ordered_countries):
            original_index = np.where(original_countries == country)[0][0]
            reordered_data[:, i] = data[:, original_index]
        return reordered_data

    ordered_y_total_sales = reorder_data(y_total_sales, countries, ordered_countries)
    ordered_y_Esales_data = reorder_data(y_Esales_data, countries, ordered_countries)
    ordered_y_NonEsales = reorder_data(y_NonEsales, countries, ordered_countries)
    ordered_y_GDP = reorder_data(y_GDP, countries, ordered_countries)

    return ordered_countries, years, ordered_y_total_sales, ordered_y_Esales_data, ordered_y_NonEsales, ordered_y_GDP

def CreateSplitBarChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.02,
        shared_yaxes=True
    )

    initial_country_idx = 0
    n_years = len(years)

    gdp_colors = ['#1e90ff', '#1c86ee', '#1874cd', '#104e8b', '#0d2f5e', '#0a1f3d'][:n_years]
    non_electric_colors = ['#9370db', '#8968cd', '#7a5fbe', '#6a5acd', '#5d478b', '#4b0082'][:n_years]
    electric_colors = ['#20b2aa', '#1eaea7', '#1ca9a0', '#189f95', '#138f85', '#0e7c6b'][:n_years]

    for i, year in enumerate(years):
        fig.add_trace(go.Bar(
            name=f'GDP {year}',
            y=[year],
            x=[-y_GDP[i, initial_country_idx]],
            orientation='h',
            marker_color=gdp_colors[i],
            hovertemplate=f'Year: {year}<br>GDP: ${y_GDP[i, initial_country_idx]:,.0f}<extra></extra>',
            showlegend=False,
            width=0.6
        ), 1, 1)

    for i, year in enumerate(years):
        fig.add_trace(go.Bar(
            name=f'Non-Electric {year}',
            y=[year],
            x=[y_NonEsales[i, initial_country_idx]],
            orientation='h',
            marker_color=non_electric_colors[i],
            hovertemplate=f'Year: {year}<br>Non-Electric Sales: {y_NonEsales[i, initial_country_idx]:,.0f} units<extra></extra>',
            showlegend=False,
            width=0.6
        ), 1, 2)

    for i, year in enumerate(years):
        fig.add_trace(go.Bar(
            name=f'Electric {year}',
            y=[year],
            x=[y_Esales_data[i, initial_country_idx]],
            orientation='h',
            marker_color=electric_colors[i],
            hovertemplate=f'Year: {year}<br>Electric Sales: {y_Esales_data[i, initial_country_idx]:,.0f} units<extra></extra>',
            showlegend=False,
            width=0.6
        ), 1, 2)

    buttons = []
    for country_idx, country in enumerate(countries):
        # Dynamic axis scaling and tick labeling for GDP for each country selection:
        gdp_vals = y_GDP[:, country_idx]
        max_gdp = max(gdp_vals)
        gdp_range = [-max_gdp * 1.1, 0]
        gdp_ticks = [-v for v in np.linspace(0, max_gdp, 5)]
        gdp_ticktext = [f"{int(v):,}" for v in np.linspace(0, max_gdp, 5)]
        sales_range = [0, max(y_total_sales[:, country_idx]) * 1.1]

        gdp_data = [[-val] for val in gdp_vals]
        non_electric_data = [[val] for val in y_NonEsales[:, country_idx]]
        electric_data = [[val] for val in y_Esales_data[:, country_idx]]
        all_data = gdp_data + non_electric_data + electric_data

        button = dict(
            args=[
                {'x': all_data},
                {
                    'title.text': f'GDP and Car Sales Analysis - {country}',
                    'xaxis.range': gdp_range,
                    'xaxis2.range': sales_range,
                    'xaxis.tickvals': gdp_ticks,
                    'xaxis.ticktext': gdp_ticktext
                }
            ],
            label=country,
            method='update'
        )
        buttons.append(button)

    # --- Layout setup ---
    fig.update_layout(
        title={
            'text': f'GDP Growth and Car Sales Analysis - {countries[initial_country_idx]}',
            'x': 0.5, 'xanchor': 'center', 'y': 0.95, 'yanchor': 'top',
            'font': {'size': 18, 'color': '#1e3a8a', 'family': 'Arial'}
        },
        font=dict(color='#1e3a8a', size=14, family='Arial'),
        showlegend=False,
        barmode='stack',
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'pad': {'r': 10, 't': 5, 'b': 5},
            'showactive': True,
            'x': 0.5,
            'xanchor': 'center',
            'y': 1.2,
            'yanchor': 'top',
            'borderwidth': 1,
            'active': initial_country_idx
        }],
        autosize=True,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Initial layout
    init_gdp_vals = y_GDP[:, initial_country_idx]
    init_max_gdp = max(init_gdp_vals)
    init_gdp_ticks = [-v for v in np.linspace(0, init_max_gdp, 5)]
    init_gdp_ticktext = [f"{int(v):,}" for v in np.linspace(0, init_max_gdp, 5)]

    fig.update_xaxes(
        title_text="GDP Per Capita ($)",
        row=1, col=1,
        title_font=dict(size=12),
        tickfont=dict(size=12),
        tickformat=',.0f',
        showgrid=True,
        range=[-init_max_gdp * 1.1, 0],
        tickvals=init_gdp_ticks,
        ticktext=init_gdp_ticktext
    )

    fig.update_xaxes(
        title_text="Car Sales (units)",
        row=1, col=2,
        title_font=dict(size=12),
        tickfont=dict(size=12),
        tickformat=',.0f',
        showgrid=True,
        range=[0, max(y_total_sales[:, initial_country_idx]) * 1.1]
    )

    fig.update_yaxes(
        title_text="Year",
        row=1, col=1,
        title_font=dict(size=16),
        tickfont=dict(size=12),
        autorange="reversed"
    )

    fig.update_yaxes(row=1, col=2, showticklabels=False)

    return fig

def main():
    countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP = CollectData()

    fig = CreateSplitBarChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP)

    filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"
    output_dir = os.path.join(os.path.dirname(__file__), "../charts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True, config={"responsive": True})
    print(f"Responsive mirrored split bar chart saved to: {output_path}")

if __name__ == "__main__":
    main()
