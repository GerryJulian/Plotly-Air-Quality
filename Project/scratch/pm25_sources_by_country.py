# -*- coding: utf-8 -*-
import os
import pandas as pd
import plotly.graph_objects as go

# 路径（脚本在 scratch/ 下；数据在 ../Background；输出在 ../charts）
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "../Background")
OUT_DIR  = os.path.join(HERE, "../charts")

# 国家与对应 CSV 文件
FILES = {
    "China": "china_pm25_sector_shares_2017.csv",
    "United States": "united_states_pm25_sector_shares_2017.csv",
    "India": "india_pm25_sector_shares_2017.csv",
    "Brazil": "brazil_pm25_sector_shares_2017.csv",
    "Mexico": "mexico_pm25_sector_shares_2017.csv",
}

# 样式设置
EMPH_SECTOR = "Transport"   # 扇区高亮关键词
MINOR_THRESHOLD = 2.0       # <2% 合并为 Other

def load_country_table(csv_path: str, country_name: str) -> pd.DataFrame:
    """读取单国 CSV，合并小于阈值的项，按占比降序；返回列：Sector, Share, Label, Pull"""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # 找占比列
    share_col = next(c for c in df.columns if ("share" in c.lower()) or ("%" in c))
    df[share_col] = pd.to_numeric(df[share_col], errors="coerce").fillna(0).clip(lower=0)

    major = df[df[share_col] >= MINOR_THRESHOLD].copy()
    other_sum = df.loc[df[share_col] < MINOR_THRESHOLD, share_col].sum()
    if other_sum > 0:
        major = pd.concat(
            [major, pd.DataFrame([{"Sector": f"Other (<{MINOR_THRESHOLD}%)", share_col: other_sum}])],
            ignore_index=True
        )

    major = major.sort_values(share_col, ascending=False, kind="stable").reset_index(drop=True)
    major.rename(columns={share_col: "Share"}, inplace=True)
    major["Label"] = major.apply(lambda r: f"{r['Sector']} — {r['Share']:.1f}%", axis=1)
    major["Pull"]  = major["Sector"].str.contains(EMPH_SECTOR, case=False, na=False).map(lambda x: 0.15 if x else 0.0)
    return major[["Sector", "Share", "Label", "Pull"]]

def build_dropdown_figure(country_tables: dict) -> go.Figure:
    countries = list(country_tables.keys())
    init = countries[0]
    t0 = country_tables[init]

    fig = go.Figure(
        data=[go.Pie(
            labels=t0["Label"],
            values=t0["Share"],
            pull=t0["Pull"],
            hole=0.25,
            textinfo="percent",           # 饼图内仅显示百分比
            hoverinfo="label+percent",
            marker=dict(line=dict(color="white", width=2))
        )]
    )

    # 下拉菜单按钮：切换国家时同时更新 labels / values / pull / 标题
    buttons = []
    for c in countries:
        tb = country_tables[c]
        buttons.append(dict(
            method="update",
            label=c,
            args=[
                {"labels": [tb["Label"]], "values": [tb["Share"]], "pull": [tb["Pull"]]},
                {"title": {
                    "text": f"{c}: Sources of Air Pollution",
                    "x": 0.5, "xanchor": "center"
                }}
            ]
        ))

    fig.update_layout(
        title={
            "text": f"{init}: Sources of Air Pollution",
            "x": 0.5, "xanchor": "center", "font": {"size": 20}
        },
        margin=dict(l=50, r=50, t=100, b=150),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.25,
            xanchor="center", x=0.5,
            font=dict(size=12)
        ),
        updatemenus=[dict(
            type="dropdown",
            showactive=True,
            x=0.98, xanchor="right",
            y=0.98, yanchor="top",
            buttons=buttons,
            bgcolor="rgba(30,58,138,0.08)",
            bordercolor="rgba(30,58,138,0.25)",
            borderwidth=1
        )]
    )
    return fig

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 逐国读取
    tables = {}
    for country, fname in FILES.items():
        csv = os.path.join(DATA_DIR, fname)
        if not os.path.exists(csv):
            print(f"⚠️ Missing: {csv}")
            continue
        tables[country] = load_country_table(csv, country)

    if not tables:
        raise FileNotFoundError("No CSVs loaded. Please check DATA_DIR/FILES paths.")

    fig = build_dropdown_figure(tables)

    # 导出单个 HTML
    out_html = os.path.join(OUT_DIR, "pm25_sources_5countries_dropdown.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"✅ Exported: {out_html}")

if __name__ == "__main__":
    main()
