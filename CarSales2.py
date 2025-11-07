import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os


def CollectData():
    # 读取数据
    df_Esales = pd.read_csv("../dataset/tbu-electric-car-sales-by-country.csv")

    # 获取国家和年份的唯一值
    countries = df_Esales['Entity'].unique()
    years = df_Esales['Year'].unique()

    # 按照1、2、5、4、3的顺序重新排列国家
    country_order = [1, 2, 5, 4, 3]  # 指定的顺序
    # 假设countries数组的索引从0开始，对应国家1、2、3、4、5
    ordered_countries = [countries[i - 1] for i in country_order]  # 因为索引从0开始，所以要减1

    # 创建销售数据矩阵
    y_total_sales = np.zeros((len(years), len(countries)))
    y_Esales_data = np.zeros((len(years), len(countries)))
    y_NonEsales = np.zeros((len(years), len(countries)))

    # 填充销售数据
    for i, year in enumerate(years):
        for j, country in enumerate(countries):

            # 获取电动汽车销量
            esales_data = df_Esales[(df_Esales['Year'] == year) & (df_Esales['Entity'] == country)]
            if esales_data.empty:
                # 如果找不到对应国家的电动汽车数据，尝试其他匹配方式
                esales_data = df_Esales[(df_Esales['Year'] == year) &
                                        (df_Esales['Entity'].str.contains(
                                            country.split()[0] if ' ' in country else country))]

            if not esales_data.empty:
                y_Esales_data[i][j] = esales_data.iloc[0, 3]  # 电动汽车销量在第4列
                y_NonEsales[i][j] = esales_data.iloc[0, 4]

            y_total_sales[i][j] = y_NonEsales[i][j] + y_Esales_data[i][j]

    # 重新排列数据以匹配国家顺序
    def reorder_data(data, original_countries, ordered_countries):
        reordered_data = np.zeros_like(data)
        for i, country in enumerate(ordered_countries):
            original_index = np.where(original_countries == country)[0][0]
            reordered_data[:, i] = data[:, original_index]
        return reordered_data

    ordered_y_total_sales = reorder_data(y_total_sales, countries, ordered_countries)
    ordered_y_Esales_data = reorder_data(y_Esales_data, countries, ordered_countries)
    ordered_y_NonEsales = reorder_data(y_NonEsales, countries, ordered_countries)

    return ordered_countries, years, ordered_y_total_sales, ordered_y_Esales_data, ordered_y_NonEsales


def CreateLineChartWithButtons(countries, years, y_total_sales, y_Esales_data, y_NonEsales):
    # 创建图表
    fig = go.Figure()

    # 蓝色系渐变颜色
    blue_colors = [
        '#1e3a8a', '#1e40af', '#2563eb', '#3b82f6', '#60a5fa'
    ]

    # 不同的线型
    line_dashes = ['solid', 'dash', 'dot', 'dashdot', 'longdash']

    # 不同的标记符号
    line_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'star']

    # 初始显示总销量
    data_type = 'Total Sales'
    initial_data = y_total_sales

    # 为每个国家添加折线
    for i, country in enumerate(countries):
        fig.add_trace(go.Scatter(
            name=country,
            x=years,
            y=initial_data[:, i],
            mode='lines+markers',
            line=dict(
                color=blue_colors[i % len(blue_colors)],
                width=4,
                dash=line_dashes[i % len(line_dashes)]
            ),
            marker=dict(
                color=blue_colors[i % len(blue_colors)],
                size=12,
                symbol=line_symbols[i % len(line_symbols)],
                line=dict(color='white', width=2)
            ),
            hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>{data_type}: %{{y:,.0f}}<extra></extra>'
        ))

    # 创建数据选择按钮
    buttons = [
        dict(
            args=[
                # 更新所有轨迹的数据和标题
                {'y': [y_total_sales[:, i] for i in range(len(countries))]},
                {
                    'title': 'Total Sales Trends by Country',
                    'yaxis.title': 'Total Sales Volume'
                }
            ],
            label='Total Sales',
            method='update'
        ),
        dict(
            args=[
                {'y': [y_Esales_data[:, i] for i in range(len(countries))]},
                {
                    'title': 'Electric Car Sales Trends by Country',
                    'yaxis.title': 'Electric Car Sales Volume'
                }
            ],
            label='Electric Sales',
            method='update'
        ),
        dict(
            args=[
                {'y': [y_NonEsales[:, i] for i in range(len(countries))]},
                {
                    'title': 'Non-Electric Car Sales Trends by Country',
                    'yaxis.title': 'Non-Electric Car Sales Volume'
                }
            ],
            label='Non-Electric Sales',
            method='update'
        )
    ]

    # 更新布局
    fig.update_layout(
        title={
            'text': 'Total Sales Trends by Country',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1e3a8a', 'family': 'Arial'}
        },
        xaxis_title='Year',
        yaxis_title='Total Sales Volume',
        plot_bgcolor='rgba(248,250,252,0.9)',
        paper_bgcolor='white',
        height=800,
        width=1200,
        font=dict(color='#1e3a8a', size=16, family='Arial'),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickmode='array',
            tickvals=years,
            title_font=dict(size=20),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            title_font=dict(size=20),
            tickfont=dict(size=16)
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(128,128,128,0.4)',
            borderwidth=2,
            font=dict(size=18, family='Arial'),
            itemwidth=40
        ),
        updatemenus=[
            {
                'buttons': buttons,
                'direction': 'down',
                'pad': {'r': 15, 't': 15, 'b': 15},
                'showactive': True,
                'x': 0.95,
                'xanchor': 'right',
                'y': 0.95,
                'yanchor': 'top',
                'bgcolor': 'rgba(30, 58, 138, 0.15)',
                'bordercolor': 'rgba(30, 58, 138, 0.4)',
                'borderwidth': 2,
                'font': {'color': '#1e3a8a', 'size': 20, 'family': 'Arial'},
                'active': 0  # 默认选中第一个按钮
            }
        ],
        # 添加数据说明
        annotations=[
            dict(
                text=" Data Type",
                x=0.95,
                y=1.03,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color='#1e3a8a', size=20, family="Arial"),
                xanchor='right',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(30, 58, 138, 0.4)',
                borderwidth=2,
                borderpad=8
            )
        ]
    )

    return fig


def main():
    # 收集数据（已经按照指定顺序排列）
    countries, years, y_total_sales, y_Esales_data, y_NonEsales = CollectData()

    #print("国家顺序:", countries)  # 调试输出，确认顺序
    #print("年份范围:", years)

    # 创建折线图
    fig = CreateLineChartWithButtons(countries, years, y_total_sales, y_Esales_data, y_NonEsales)

    # 获取当前文件名并更改扩展名为.html
    filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"

    # 保存路径: /project/charts
    output_dir = os.path.join(os.path.dirname(__file__), "../charts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # 保存HTML
    fig.write_html(output_path, include_plotlyjs="cdn")
    #print(f"折线图已保存到: {output_path}")


if __name__ == "__main__":
    main()