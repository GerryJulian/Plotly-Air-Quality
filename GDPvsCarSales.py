import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os


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
                y_NonEsales[i][j] = esales_data.iloc[0, 4]  # 非电动汽车销量在第5列

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


def CreateSplitBarChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP):
    # 创建子图：中间纵轴，左右两侧横向条形图
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.02,  # 减少子图间距
        shared_yaxes=True
    )

    # 初始国家索引
    initial_country_idx = 0

    # 生成年份渐变颜色
    n_years = len(years)

    # GDP - 蓝色系渐变（从浅蓝到深蓝）
    gdp_colors = ['#1e90ff', '#1c86ee', '#1874cd', '#104e8b', '#0d2f5e', '#0a1f3d']

    # 非电动汽车销售 - 紫色系渐变（从浅紫到深紫）
    non_electric_colors = ['#9370db', '#8968cd', '#7a5fbe', '#6a5acd', '#5d478b', '#4b0082']

    # 电动汽车销售 - 青色系渐变（从浅青到深青）
    electric_colors = ['#20b2aa', '#1eaea7', '#1ca9a0', '#189f95', '#138f85', '#0e7c6b']

    # 确保颜色数量与年份数量匹配
    gdp_colors = gdp_colors[:n_years]
    non_electric_colors = non_electric_colors[:n_years]
    electric_colors = electric_colors[:n_years]

    # 左侧：GDP横向条形图 - 使用负值向左延展
    for i, year in enumerate(years):
        fig.add_trace(go.Bar(
            name=f'GDP {year}',
            y=[year],
            x=[-y_GDP[i, initial_country_idx]],  # 使用负值向左延展
            orientation='h',
            marker_color=gdp_colors[i],
            hovertemplate=(
                f'Year: {year}<br>'
                f'GDP: ${y_GDP[i, initial_country_idx]:,.0f}<extra></extra>'
            ),
            showlegend=False,
            width=0.6  # 调整条形的宽度
        ), 1, 1)

    # 右侧：堆叠横向条形图 - 非电动汽车销量
    for i, year in enumerate(years):
        fig.add_trace(go.Bar(
            name=f'Non-Electric {year}',
            y=[year],
            x=[y_NonEsales[i, initial_country_idx]],
            orientation='h',
            marker_color=non_electric_colors[i],
            hovertemplate=(
                f'Year: {year}<br>'
                f'Non-Electric Sales: {y_NonEsales[i, initial_country_idx]:,.0f} units<extra></extra>'
            ),
            showlegend=False,
            width=0.6
        ), 1, 2)

    # 右侧：堆叠横向条形图 - 电动汽车销量
    for i, year in enumerate(years):
        fig.add_trace(go.Bar(
            name=f'Electric {year}',
            y=[year],
            x=[y_Esales_data[i, initial_country_idx]],
            orientation='h',
            marker_color=electric_colors[i],
            hovertemplate=(
                f'Year: {year}<br>'
                f'Electric Sales: {y_Esales_data[i, initial_country_idx]:,.0f} units<extra></extra>'
            ),
            showlegend=False,
            width=0.6
        ), 1, 2)

    # 创建所有注释（annotations）列表
    all_annotations = [
        dict(
            text="🏛️ Country Selector",
            x=0.5,
            y=1.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color='#1e3a8a', size=14, family="Arial"),
            xanchor='center'
        ),
        dict(
            text="Color Gradient: Light → Dark (Earlier → Later Years)",
            x=0.5,
            y=-0.05,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color='#666666', size=12, family="Arial"),
            xanchor='center'
        ),
        dict(
            x=0.25,  # 左侧子图标题位置
            y=1.0,
            xref="paper",
            yref="paper",
            text="GDP by Year",
            showarrow=False,
            font=dict(size=16, color='#1e3a8a'),
            xanchor='center'
        ),
        dict(
            x=0.75,  # 右侧子图标题位置
            y=1.0,
            xref="paper",
            yref="paper",
            text="Car Sales by Year",
            showarrow=False,
            font=dict(size=16, color='#1e3a8a'),
            xanchor='center'
        )
    ]

    # 创建国家选择按钮
    buttons = []
    for country_idx, country in enumerate(countries):
        # 准备GDP数据 (使用负值向左延展)
        gdp_data = [[-val] for val in y_GDP[:, country_idx]]
        # 准备非电动汽车数据
        non_electric_data = [[val] for val in y_NonEsales[:, country_idx]]
        # 准备电动汽车数据
        electric_data = [[val] for val in y_Esales_data[:, country_idx]]

        # 合并所有数据 (GDP + 非电动 + 电动)
        all_data = gdp_data + non_electric_data + electric_data

        button = dict(
            args=[
                # 第一个元素：更新所有轨迹的数据
                {'x': all_data},
                # 第二个元素：更新布局
                {
                    'title.text': f'GDP and Car Sales Analysis - {country}',
                    'xaxis.range': [-max(y_GDP[:, country_idx]) * 1.1, 0],
                    'xaxis2.range': [0, max(y_total_sales[:, country_idx]) * 1.1]
                }
            ],
            label=country,
            method='update'
        )
        buttons.append(button)

    # 更新布局
    fig.update_layout(
        title={
            'text': f'GDP and Car Sales Analysis - {countries[initial_country_idx]}',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#1e3a8a', 'family': 'Arial'}
        },
        height=700,
        width=1400,  # 增加宽度以适应新的布局
        font=dict(color='#1e3a8a', size=14, family='Arial'),
        showlegend=False,  # 关闭自动图例，因为我们使用颜色编码年份
        barmode='stack',  # 右侧堆叠条形图
        updatemenus=[
            {
                'buttons': buttons,
                'direction': 'down',
                'pad': {'r': 10, 't': 10, 'b': 10},
                'showactive': True,
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.98,
                'yanchor': 'top',
                'bgcolor': 'rgba(30, 58, 138, 0.1)',
                'bordercolor': 'rgba(30, 58, 138, 0.3)',
                'borderwidth': 1,
                'font': {'color': '#1e3a8a', 'size': 14, 'family': 'Arial'},
                'active': initial_country_idx
            }
        ],
        annotations=all_annotations
    )

    # 更新坐标轴
    fig.update_xaxes(
        title_text="GDP ($)",
        row=1, col=1,
        title_font=dict(size=16),
        tickfont=dict(size=12),
        tickformat=',.0f',
        showgrid=True,
        range=[-max(y_GDP[:, initial_country_idx]) * 1.1, 0]  # 左侧x轴范围，从负值到0
    )

    fig.update_xaxes(
        title_text="Car Sales (units)",
        row=1, col=2,
        title_font=dict(size=16),
        tickfont=dict(size=12),
        tickformat=',.0f',
        showgrid=True,
        range=[0, max(y_total_sales[:, initial_country_idx]) * 1.1]  # 右侧x轴范围，从0到正值
    )

    fig.update_yaxes(
        title_text="Year",
        row=1, col=1,
        title_font=dict(size=16),
        tickfont=dict(size=12),
        autorange="reversed"  # 确保年份从上到下递增
    )

    fig.update_yaxes(
        row=1, col=2,
        showticklabels=False  # 右侧隐藏Y轴标签，因为已经共享
    )

    return fig


def main():
    # 收集数据
    countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP = CollectData()

    """
    print("国家顺序:", countries)
    print("年份范围:", years)
    print("GDP数据示例:", y_GDP[:, 0])
    print("电动销量数据示例:", y_Esales_data[:, 0])
    print("非电动销量数据示例:", y_NonEsales[:, 0])
    """

    # 创建图表
    fig = CreateSplitBarChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales, y_GDP)

    # 获取当前文件名并更改扩展名为.html
    filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"

    # 保存路径: /project/charts
    output_dir = os.path.join(os.path.dirname(__file__), "../charts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # 保存HTML
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"改进版分屏条形图已保存到: {output_path}")


if __name__ == "__main__":
    main()