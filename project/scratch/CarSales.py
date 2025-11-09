import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

def CollectData():
    # --- Load dataset ---
    df = pd.read_csv("../dataset/carSales.csv", sep=';')

    # --- Explicit country order ---
    ordered_countries = ['China', 'United States', 'India', 'Brazil', 'Mexico']

    # --- Extract unique years ---
    years = sorted(df['Year'].unique())

    # --- Initialize matrices ---
    y_Esales_data = np.zeros((len(years), len(ordered_countries)))
    y_NonEsales = np.zeros((len(years), len(ordered_countries)))
    y_total_sales = np.zeros((len(years), len(ordered_countries)))

    # --- Fill data ---
    for i, year in enumerate(years):
        for j, country in enumerate(ordered_countries):
            row = df[(df['Entity'] == country) & (df['Year'] == year)]
            if not row.empty:
                y_Esales_data[i][j] = row['Electric cars sold'].values[0]
                y_NonEsales[i][j] = row['Non-electric car sales'].values[0]
                y_total_sales[i][j] = row['Total Sales'].values[0]

    return ordered_countries, years, y_total_sales, y_Esales_data, y_NonEsales

def CreateCombinedChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales):
    fig = go.Figure()
    initial_year_idx = 0

    colors = {
        'non_electric': '#1e40af',
        'electric': '#3b82f6'
    }

    # --- Non-electric bars (stacked) ---
    fig.add_trace(go.Bar(
        name='Non-Electric Sales',
        x=countries,
        y=y_NonEsales[initial_year_idx],
        marker_color=colors['non_electric'],
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Non-Electric Sales: %{y:,.0f}<extra></extra>'
    ))

    # --- Electric bars (stacked) ---
    fig.add_trace(go.Bar(
        name='Electric Sales',
        x=countries,
        y=y_Esales_data[initial_year_idx],
        marker_color=colors['electric'],
        opacity=0.9,
        hovertemplate='<b>%{x}</b><br>Electric Sales: %{y:,.0f}<extra></extra>'
    ))

    # --- Scatter (black diamond) for total sales ---
    fig.add_trace(go.Scatter(
        name='Total Sales',
        x=countries,
        y=y_total_sales[initial_year_idx],
        mode='markers',
        marker=dict(color='black', size=16, symbol='diamond'),
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Total Sales: %{y:,.0f}<br>" +
            "Non-Electric Sales: %{customdata[0]:,.0f}<br>" +
            "Electric Sales: %{customdata[1]:,.0f}<extra></extra>"
        ),
        customdata=np.stack([y_NonEsales[initial_year_idx], y_Esales_data[initial_year_idx]], axis=-1)
    ))

    # --- Dropdown buttons for year selection ---
    buttons = []
    for i, year in enumerate(years):
        # Button updates all three traces: bars and scatter
        button = dict(
            args=[
                {
                    'y': [y_NonEsales[i], y_Esales_data[i], y_total_sales[i]],
                    'customdata': [None, None, np.stack([y_NonEsales[i], y_Esales_data[i]], axis=-1)]
                },
                {'title.text': f'Automobile Sales by Country - {year}'}
            ],
            label=str(year),
            method='update'
        )
        buttons.append(button)

    # --- Layout ---
    fig.update_layout(
        title={
            'text': f'Automobile Sales by Country - {years[initial_year_idx]}',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e3a8a'}
        },
        xaxis_title='Countries',
        yaxis_title='Sales Volume',
        barmode='stack',
        plot_bgcolor='rgba(248,250,252,0.9)',
        paper_bgcolor='white',
        font=dict(color='#1e3a8a', size=16),
        xaxis=dict(categoryorder='array', categoryarray=countries),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.92),
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'x': 0.95,
            'xanchor': 'right',
            'y': 0.95,
            'yanchor': 'top',
            'showactive': True,
            'active': initial_year_idx
        }],
        margin=dict(l=60, r=60, t=80, b=60),
        autosize=True
    )

    # --- Axis title font sizes ---
    fig.update_xaxes(title_font=dict(size=15))
    fig.update_yaxes(title_font=dict(size=15))

    # Make figure responsive
    fig.update_layout(height=None, width=None)

    return fig

def main():
    countries, years, y_total_sales, y_Esales_data, y_NonEsales = CollectData()
    fig = CreateCombinedChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales)

    filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"
    output_dir = os.path.join(os.path.dirname(__file__), "../charts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={'responsive': True}
    )

    print(f"✅ Responsive stacked bar chart saved to: {output_path}")

if __name__ == "__main__":
    main()
