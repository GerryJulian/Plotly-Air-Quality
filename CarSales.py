import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os


def CollectData():
    # 读取数据
    #df_sales = pd.read_excel('../dataset/CarSales2019-2024.xlsx')
    df_Esales = pd.read_csv("../dataset/tbu-electric-car-sales-by-country.csv")
    #df_gdp = pd.read_csv("../dataset/tbu-global-gdp-per-capita-by-country.csv")

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
    #y_GDP=np.zeros((len(years), len(countries)))

    """
        for i, year in enumerate(years):
        for j, country in enumerate(countries):
            y_GDP[i][j]=df_gdp.iloc[j,i+2].values
    """

    # 填充销售数据
    for i, year in enumerate(years):
        for j, country in enumerate(countries):
            """
            # 获取总汽车销量
            sales_data = df_sales[(df_sales['Years'] == year) & (df_sales['Countries'] == country)]
            if not sales_data.empty:
                y_total_sales[i][j] = sales_data.iloc[0, 2]  # 总销量在第3列
            """

            # 获取电动汽车销量
            esales_data = df_Esales[(df_Esales['Year'] == year) & (df_Esales['Entity'] == country)]
            if esales_data.empty:
                # 如果找不到对应国家的电动汽车数据，尝试其他匹配方式
                esales_data = df_Esales[(df_Esales['Year'] == year) &
                                        (df_Esales['Entity'].str.contains(
                                            country.split()[0] if ' ' in country else country))]

            if not esales_data.empty:
                y_Esales_data[i][j] = esales_data.iloc[0, 3]  # 电动汽车销量在第4列
                y_NonEsales[i][j]=esales_data.iloc[0, 4]

            y_total_sales[i][j]=y_NonEsales[i][j]+y_Esales_data[i][j]
            # 计算非电动汽车销量
            #y_NonEsales[i][j] = y_total_sales[i][j] - y_Esales_data[i][j]
            # 确保非负
            #y_NonEsales[i][j] = max(0, y_NonEsales[i][j])

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


def CreateCombinedChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales):
    # 创建图表
    fig = go.Figure()

    # 初始年份索引
    initial_year_idx = 0

    # 蓝色系配色方案
    colors = {
        'non_electric': '#1e40af',  # 深蓝色 - 非电动汽车
        'electric': '#3b82f6',  # 中等蓝色 - 电动汽车
        'total_line': '#60a5fa',  # 浅蓝色 - 总销量折线
        'total_marker': '#1e3a8a'  # 深蓝色 - 总销量标记点
    }

    # 添加非电动汽车销量条形图
    fig.add_trace(go.Bar(
        name='Non-Electric Sales',
        x=countries,
        y=y_NonEsales[initial_year_idx],
        marker_color=colors['non_electric'],
        marker_line_color='rgba(255,255,255,0.8)',
        marker_line_width=1,
        opacity=0.8,
        width=0.35,
        hovertemplate='<b>%{x}</b><br>Non-Electric Sales: %{y:,.0f}<extra></extra>'
    ))

    # 添加电动汽车销量条形图
    fig.add_trace(go.Bar(
        name='Electric Sales',
        x=countries,
        y=y_Esales_data[initial_year_idx],
        marker_color=colors['electric'],
        marker_line_color='rgba(255,255,255,0.8)',
        marker_line_width=1,
        opacity=0.9,
        width=0.35,
        hovertemplate='<b>%{x}</b><br>Electric Sales: %{y:,.0f}<extra></extra>'
    ))

    # 添加总销量折线图（在条形图上显示点）
    fig.add_trace(go.Scatter(
        name='Total Sales',
        x=countries,
        y=y_total_sales[initial_year_idx],
        mode='markers+lines',
        line=dict(color=colors['total_line'], width=3, dash='dot'),
        marker=dict(
            color=colors['total_marker'],
            size=10,
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Total Sales: %{y:,.0f}<extra></extra>'
    ))

    # 创建年份选择按钮
    buttons = []
    for i, year in enumerate(years):
        button = dict(
            args=[
                # 更新所有轨迹的数据
                {
                    'y': [
                        y_NonEsales[i],  # 非电动条形图
                        y_Esales_data[i],  # 电动条形图
                        y_total_sales[i]  # 总销量折线图
                    ],
                    'title': f'Automobile Sales by Country - {year}'
                }
            ],
            label=str(year),
            method='update'
        )
        buttons.append(button)

    # 更新布局
    fig.update_layout(
        title={
            'text': f'Automobile Sales by Country - {years[initial_year_idx]}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a'}
        },
        xaxis_title='Countries',
        yaxis_title='Sales Volume',
        barmode='group',  # 分组条形图
        plot_bgcolor='rgba(248,250,252,0.9)',
        paper_bgcolor='white',
        height=700,
        width=1000,
        font=dict(color='#1e3a8a', size=20),
        xaxis=dict(
            tickangle=-0,
            gridcolor='rgba(128,128,128,0.2)',
            # 确保国家按指定顺序显示
            categoryorder='array',
            categoryarray=countries
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(128,128,128,0.3)',
            borderwidth=1
        ),
        updatemenus=[
            {
                'buttons': buttons,
                'direction': 'down',
                'pad': {'r': 10, 't': 10, 'b': 10},
                'showactive': True,
                'x': 0.95,
                'xanchor': 'right',
                'y': 0.95,
                'yanchor': 'top',
                'bgcolor': 'rgba(30, 58, 138, 0.1)',
                'bordercolor': 'rgba(30, 58, 138, 0.3)',
                'borderwidth': 1,
                'font': {'color': '#1e3a8a', 'size': 20},
                'active': initial_year_idx
            }
        ]
    )

    return fig


def main():
    # 收集数据（已经按照指定顺序排列）
    countries, years, y_total_sales, y_Esales_data, y_NonEsales = CollectData()

    #print("国家顺序:", countries)  # 调试输出，确认顺序

    # 创建组合图表
    fig = CreateCombinedChart(countries, years, y_total_sales, y_Esales_data, y_NonEsales)

    # 获取当前文件名并更改扩展名为.html
    filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"

    # 保存路径: /project/charts
    output_dir = os.path.join(os.path.dirname(__file__), "../charts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # 保存HTML
    fig.write_html(output_path, include_plotlyjs="cdn")
    #print(f"组合图表已保存到: {output_path}")


if __name__ == "__main__":
    main()