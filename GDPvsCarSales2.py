import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import math


def CollectData():
    # 读取数据
    df_Esales = pd.read_csv("../dataset/tbu-electric-car-sales-by-country.csv")
    df_gdp = pd.read_csv("../dataset/tbu-global-gdp-per-capita-by-country.csv")

    # 获取国家和年份的唯一值
    countries = df_Esales['Entity'].unique()
    years = df_Esales['Year'].unique()

    # 按照1、2、5、4、3的顺序重新排列国家
    country_order = [1, 2, 5, 4, 3]  # 指定的顺序
    ordered_countries = [countries[i - 1] for i in country_order]

    # 创建销售数据矩阵
    y_total_sales = np.zeros((len(years), len(countries)))
    y_Esales_data = np.zeros((len(years), len(countries)))
    y_NonEsales = np.zeros((len(years), len(countries)))
    y_GDP = np.zeros((len(years), len(countries)))

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

            # 获取GDP数据
            gdp_data = df_gdp[df_gdp['Country Name'] == country]
            if not gdp_data.empty:
                year_col = str(year)
                if year_col in gdp_data.columns:
                    y_GDP[i][j] = gdp_data[year_col].values[0]

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
    ordered_y_GDP = reorder_data(y_GDP, countries, ordered_countries)

    return ordered_countries, years, ordered_y_total_sales, ordered_y_Esales_data, ordered_y_NonEsales, ordered_y_GDP


def CreateArcChartsWithCountries(countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP):
    # 创建子图：左侧GDP环形图，右侧三个弧环图（包含国家数据）
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "scatterpolar"}]],
        subplot_titles=("GDP by Country", "Car Sales by Country and Type"),
        horizontal_spacing=0.15
    )

    # 按照12543顺序的蓝色系渐变颜色 - 用于左右两侧
    country_colors = [
        '#0f1f4d',  # 国家1 - 最深蓝
        '#1e3a8a',  # 国家2 - 深蓝
        '#2563eb',  # 国家5 - 中蓝
        '#3b82f6',  # 国家4 - 中浅蓝
        '#93c5fd'  # 国家3 - 浅蓝
    ]

    # 初始年份索引
    initial_year_idx = 0

    # 左侧：GDP环形图
    fig.add_trace(go.Pie(
        name="GDP",
        labels=countries,
        values=y_GDP[initial_year_idx],
        hole=0.5,
        marker=dict(colors=country_colors),
        textinfo='percent+label',
        insidetextorientation='radial',
        hovertemplate='<b>%{label}</b><br>GDP: $%{value:,.0f}<br>Contribution: %{percent}<extra></extra>',
        domain=dict(x=[0, 0.45]),
        sort=False
    ), 1, 1)

    # 定义环的半径范围
    ring_radii = {
        'electric': [2.5, 4.0],  # 电动销量环 2.5-4
        'non_electric': [4.5, 6.0],  # 非电动销量环 4.5-6
        'total': [6.5, 8.0]  # 总销量环 6.5-8
    }

    # 计算每个销售类型的总和
    total_sales_sum = np.sum(y_total_sales[initial_year_idx])
    non_esales_sum = np.sum(y_NonEsales[initial_year_idx])
    esales_sum = np.sum(y_Esales_data[initial_year_idx])

    # 找到最大值用于标准化弧度
    max_sales = max(total_sales_sum, non_esales_sum, esales_sum)

    # 计算每个销售类型的弧度（最大为2π，即360度）
    total_arc = (total_sales_sum / max_sales) * 2 * math.pi
    non_esales_arc = (non_esales_sum / max_sales) * 2 * math.pi
    esales_arc = (esales_sum / max_sales) * 2 * math.pi

    # 为每个销售类型和国家创建弧环
    sales_types = [
        ('Electric Sales', y_Esales_data[initial_year_idx], ring_radii['electric'], esales_arc),
        ('Non-Electric Sales', y_NonEsales[initial_year_idx], ring_radii['non_electric'], non_esales_arc),
        ('Total Sales', y_total_sales[initial_year_idx], ring_radii['total'], total_arc)
    ]

    # 创建弧环的数据点
    def create_country_arc_points(start_angle, arc_angle, inner_r, outer_r, country_data, country_name, country_color,
                                  sales_type, total_sales_type):
        """创建国家弧环的数据点"""
        angles = np.linspace(start_angle, start_angle + arc_angle, 50)
        inner_x = inner_r * np.cos(angles)
        inner_y = inner_r * np.sin(angles)
        outer_x = outer_r * np.cos(angles)
        outer_y = outer_r * np.sin(angles)

        # 创建闭合路径
        x = np.concatenate([inner_x, outer_x[::-1]])
        y = np.concatenate([inner_y, outer_y[::-1]])

        # 计算半径和角度
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x) * 180 / math.pi

        return {
            'r': r,
            'theta': theta,
            'country_data': country_data,
            'country_name': country_name,
            'country_color': country_color,
            'sales_type': sales_type,
            'total_sales_type': total_sales_type
        }

    # 为每个销售类型和国家添加弧环
    arc_start = 0  # 从0度开始

    # 存储所有弧环数据用于更新
    all_arc_data = []

    for sales_type, data, radii, arc_length in sales_types:
        # 计算每个国家在弧环中的比例
        total_sales_type = np.sum(data)
        if total_sales_type == 0:
            continue

        # 每个国家的弧度比例
        country_arcs = [(data[i] / total_sales_type) * arc_length for i in range(len(countries))]

        # 当前弧环的起始角度
        current_arc_start = arc_start

        # 为每个国家创建弧环
        for i, country in enumerate(countries):
            country_arc = country_arcs[i]
            if country_arc == 0:
                continue

            arc_data = create_country_arc_points(
                current_arc_start, country_arc,
                radii[0], radii[1],
                data[i], country, country_colors[i], sales_type, total_sales_type
            )

            # 添加国家弧环
            fig.add_trace(go.Scatterpolar(
                r=arc_data['r'],
                theta=arc_data['theta'],
                fill='toself',
                fillcolor=arc_data['country_color'],
                line=dict(color=arc_data['country_color'], width=2),
                name=f"{country}",
                hovertemplate=(
                    f'<b>{arc_data["country_name"]}</b><br>'
                    f'{arc_data["sales_type"]}: {arc_data["country_data"]:,.0f} units<br>'
                    f'Contribution: {arc_data["country_data"] / arc_data["total_sales_type"] * 100:.1f}%<extra></extra>'
                ),
                showlegend=False
            ), 1, 2)

            # 保存弧环数据用于更新
            all_arc_data.append(arc_data)

            # 更新当前弧环的起始角度
            current_arc_start += country_arc

    # 创建年份选择按钮
    buttons = []
    for i, year in enumerate(years):
        # 计算当前年份的数据
        total_sales_sum_i = np.sum(y_total_sales[i])
        non_esales_sum_i = np.sum(y_NonEsales[i])
        esales_sum_i = np.sum(y_Esales_data[i])

        max_sales_i = max(total_sales_sum_i, non_esales_sum_i, esales_sum_i)

        total_arc_i = (total_sales_sum_i / max_sales_i) * 2 * math.pi
        non_esales_arc_i = (non_esales_sum_i / max_sales_i) * 2 * math.pi
        esales_arc_i = (esales_sum_i / max_sales_i) * 2 * math.pi

        # 准备更新数据
        update_data = []

        # 为每个销售类型和国家准备数据
        sales_types_i = [
            ('Electric Sales', y_Esales_data[i], ring_radii['electric'], esales_arc_i),
            ('Non-Electric Sales', y_NonEsales[i], ring_radii['non_electric'], non_esales_arc_i),
            ('Total Sales', y_total_sales[i], ring_radii['total'], total_arc_i)
        ]

        arc_start_i = 0

        for sales_type, data, radii, arc_length in sales_types_i:
            # 计算每个国家在弧环中的比例
            total_sales_type = np.sum(data)
            if total_sales_type == 0:
                continue

            # 每个国家的弧度比例
            country_arcs = [(data[j] / total_sales_type) * arc_length for j in range(len(countries))]

            # 当前弧环的起始角度
            current_arc_start = arc_start_i

            # 为每个国家创建弧环数据
            for j, country in enumerate(countries):
                country_arc = country_arcs[j]
                if country_arc == 0:
                    continue

                arc_data = create_country_arc_points(
                    current_arc_start, country_arc,
                    radii[0], radii[1],
                    data[j], country, country_colors[j], sales_type, total_sales_type
                )

                update_data.append({
                    'r': arc_data['r'],
                    'theta': arc_data['theta'],
                    'hovertemplate': (
                        f'<b>{arc_data["country_name"]}</b><br>'
                        f'{arc_data["sales_type"]}: {arc_data["country_data"]:,.0f} units<br>'
                        f'Contribution: {arc_data["country_data"] / arc_data["total_sales_type"] * 100:.1f}%<extra></extra>'
                    )
                })

                # 更新当前弧环的起始角度
                current_arc_start += country_arc

        button = dict(
            args=[
                # 更新GDP数据
                {'values': [y_GDP[i]]},
                # 更新弧环数据
                update_data,
                {
                    'title': f'Economic & Automotive Market Analysis - {year}'
                }
            ],
            label=str(year),
            method='update'
        )
        buttons.append(button)

    # 更新布局
    fig.update_layout(
        title={
            'text': f'Economic & Automotive Market Analysis - {years[initial_year_idx]}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1e3a8a', 'family': 'Arial'}
        },
        height=800,
        width=1200,
        font=dict(color='#1e3a8a', size=16, family='Arial'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(128,128,128,0.3)',
            borderwidth=1,
            font=dict(size=14),
            title=dict(
                text="Countries",
                side="top"
            )
        ),
        polar=dict(
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 90, 180, 270],
                ticktext=['0°', '90°', '180°', '270°'],
                direction='clockwise'
            ),
            radialaxis=dict(
                visible=True,
                range=[0, 9],
                showticklabels=False,  # 隐藏径向轴标签
                showline=False,
                ticks=''
            ),
            bgcolor='rgba(248,250,252,0.5)'
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
                'active': initial_year_idx
            }
        ],
        annotations=[
            dict(
                text="Year Selector",
                x=0.95,
                y=1.02,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color='#1e3a8a', size=20, family="Arial"),
                xanchor='right',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(30, 58, 138, 0.4)',
                borderwidth=2,
                borderpad=8
            ),
            # 在右侧图表上直接标识环的类型
            dict(
                text="Electric Sales",
                x=0.75,
                y=0.35,
                xref="paper",
                yref="paper",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=-30,
                ay=0,
                font=dict(color='#1e3a8a', size=16, family="Arial"),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#1e3a8a',
                borderwidth=1,
                borderpad=4
            ),
            dict(
                text="Non-Electric Sales",
                x=0.75,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=-30,
                ay=0,
                font=dict(color='#1e3a8a', size=16, family="Arial"),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#1e3a8a',
                borderwidth=1,
                borderpad=4
            ),
            dict(
                text="Total Sales",
                x=0.75,
                y=0.65,
                xref="paper",
                yref="paper",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=-30,
                ay=0,
                font=dict(color='#1e3a8a', size=16, family="Arial"),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#1e3a8a',
                borderwidth=1,
                borderpad=4
            ),
            # 添加说明文字
            dict(
                text="Inner to Outer Rings:",
                x=0.75,
                y=0.85,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color='#1e3a8a', size=14, family="Arial"),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#1e3a8a',
                borderwidth=1,
                borderpad=4
            )
        ]
    )

    # 更新子图标题
    fig.update_annotations(font=dict(size=18, color='#1e3a8a'))

    return fig


def main():
    # 收集数据（包含GDP）
    countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP = CollectData()

    """
    print("国家顺序:", countries)
    print("GDP数据:", y_GDP[0])
    print("总销量数据:", y_total_sales[0])
    print("电动销量数据:", y_Esales_data[0])
    print("非电动销量数据:", y_NonEsales[0])
    """

    # 创建图表
    fig = CreateArcChartsWithCountries(countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP)

    # 获取当前文件名并更改扩展名为.html
    filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"

    # 保存路径: /project/charts
    output_dir = os.path.join(os.path.dirname(__file__), "../charts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # 保存HTML
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"带国家数据的弧环图已保存到: {output_path}")


if __name__ == "__main__":
    main()