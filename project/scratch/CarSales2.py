import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

def CollectData():
    # --- Load dataset ---
    df = pd.read_csv("../dataset/carSales.csv", sep=';')

    # Extract unique countries and years
    countries = df['Entity'].unique()
    years = df['Year'].unique()

    # --- Optional fixed order ---
    country_order = [1, 2, 5, 4, 3]  # you can edit this order or remove
    ordered_countries = [countries[i - 1] for i in country_order]

    # --- Prepare arrays ---
    y_total_sales = np.zeros((len(years), len(countries)))
    y_Esales_data = np.zeros((len(years), len(countries)))
    y_NonEsales = np.zeros((len(years), len(countries)))

    for i, year in enumerate(years):
        for j, country in enumerate(countries):
            data = df[(df['Year'] == year) & (df['Entity'] == country)]
            if not data.empty:
                y_Esales_data[i][j] = data.iloc[0]['Electric cars sold']
                y_NonEsales[i][j] = data.iloc[0]['Non-electric car sales']
                y_total_sales[i][j] = data.iloc[0]['Total Sales']

    # reorder data based on custom order
    def reorder_data(data, original_countries, ordered_countries):
        reordered = np.zeros_like(data)
        for i, country in enumerate(ordered_countries):
            original_index = np.where(original_countries == country)[0][0]
            reordered[:, i] = data[:, original_index]
        return reordered

    ordered_y_total_sales = reorder_data(y_total_sales, countries, ordered_countries)
    ordered_y_Esales_data = reorder_data(y_Esales_data, countries, ordered_countries)
    ordered_y_NonEsales = reorder_data(y_NonEsales, countries, ordered_countries)

    return ordered_countries, years, ordered_y_total_sales, ordered_y_Esales_data, ordered_y_NonEsales

def CreateLineChartWithButtons(countries, years, y_total_sales, y_Esales_data, y_NonEsales):
    fig = go.Figure()
    blue_colors = ['#1e3a8a', '#1e40af', '#2563eb', '#3b82f6', '#60a5fa']

    initial_data = y_total_sales
    data_type = 'Total Sales'

    # --- Add line traces (color per country) ---
    for i, country in enumerate(countries):
        fig.add_trace(go.Scatter(
            name=country,
            x=years,
            y=initial_data[:, i],
            mode='lines+markers',
            line=dict(color=blue_colors[i % len(blue_colors)], width=4),
            marker=dict(color=blue_colors[i % len(blue_colors)], size=10, line=dict(color='white', width=2)),
            hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>{data_type}: %{{y:,.0f}}<extra></extra>'
        ))

    # --- Define function to get dynamic y-axis range with padding ---
    def get_range(data):
        return [0, data.max() * 1.25] if data.max() > 0 else [0, 1]

    total_range = get_range(y_total_sales)
    electric_range = get_range(y_Esales_data)
    nonelec_range = get_range(y_NonEsales)

    # --- Dropdown buttons with y-axis range set dynamically ---
    buttons = [
        dict(
            args=[{'y': [y_total_sales[:, i] for i in range(len(countries))]},
                  {'title.text': 'Total Sales Trends by Country',
                   'yaxis.title.text': 'Total Sales Volume',
                   'yaxis.range': total_range}],
            label='Total Sales', method='update'
        ),
        dict(
            args=[{'y': [y_Esales_data[:, i] for i in range(len(countries))]},
                  {'title.text': 'Electric Car Sales Trends by Country',
                   'yaxis.title.text': 'Electric Car Sales Volume',
                   'yaxis.range': electric_range}],
            label='Electric Sales', method='update'
        ),
        dict(
            args=[{'y': [y_NonEsales[:, i] for i in range(len(countries))]},
                  {'title.text': 'Non-Electric Car Sales Trends by Country',
                   'yaxis.title.text': 'Non-Electric Car Sales Volume',
                   'yaxis.range': nonelec_range}],
            label='Non-Electric Sales', method='update'
        )
    ]

    # --- Layout configuration (NO yaxis range here) ---
    fig.update_layout(
        title=dict(
            text='Total Sales Trends by Country',
            x=0.5, y=0.95, xanchor='center',
            font=dict(size=20, color='#1e3a8a', family='Arial')
        ),
        xaxis_title='Year',
        yaxis_title='Total Sales Volume',
        plot_bgcolor='rgba(248,250,252,0.9)',
        paper_bgcolor='white',
        font=dict(color='#1e3a8a', size=16, family='Arial'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.03,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(128,128,128,0.4)',
            borderwidth=2,
            font=dict(size=16, family='Arial'),
            itemwidth=40
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickmode='array',
            tickvals=years,
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            title_font=dict(size=18),
            tickfont=dict(size=16)
            # Don't set range here!
        ),
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            pad={'r': 10, 't': 10},
            showactive=True,
            x=0.62,
            xanchor='center',
            y=1.32,
            yanchor='top',
            active=0
        )]
    )

    fig.update_layout(
        width=650,
        height=350,
        autosize=True,
        margin=dict(l=20, r=180, t=100, b=60)
    )

    return fig

def main():
    countries, years, y_total_sales, y_Esales_data, y_NonEsales = CollectData()
    fig = CreateLineChartWithButtons(countries, years, y_total_sales, y_Esales_data, y_NonEsales)

    output_dir = os.path.join(os.path.dirname(__file__), "../charts")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"
    output_path = os.path.join(output_dir, filename)

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True, config={"responsive": True})
    print(f"✅ Responsive chart saved to: {output_path}")

if __name__ == "__main__":
    main()
