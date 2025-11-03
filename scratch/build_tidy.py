# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "dataset"


# ---------- Helpers ----------
def strip_weird(s):
    """去掉表头里的 =\"Country\" 之类包裹"""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r'^="?', '', s)  # 开头的 ="
    s = re.sub(r'"?$', '', s)  # 结尾的 "
    return s

def clean_excel_formula_strings(df):
    """
    清理 Excel 导出中常见的公式字符串形式，如 =\"China\"、=12.5 等。
    对整个 DataFrame 扫描字符串列。
    """
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r'^="(.*)"$', r'\1', regex=True)
                .str.replace(r'^=(.*)$', r'\1', regex=True)
                .str.replace('"', '', regex=False)
                .replace({'null': np.nan, 'NULL': np.nan, 'None': np.nan, 'NaN': np.nan, '': np.nan})
                .str.strip()
            )
    return df



def load_population_any(path: Path):
    """
    尝试自动识别人口CSV的3种结构：
    A) 长表：['Entity' or 'Country', 'Year', 'Population']
    B) 宽表（WDI）：包含 'Country Name' 且年份列为 '1960'...'2024'（可能表头不在第一行）
    C) 分号一列：只有1列，内容含有 ';'
    返回：长表 DataFrame，列为 [Country, Year, Population]
    """
    import csv, io, re
    # 先读原始文本的前100行，猜测结构
    with open(path, "r", encoding="utf-8") as f:
        head_txt = "".join([next(f) for _ in range(100)])  # 读前100行
    # 情况 C：分号一列
    if ";" in head_txt and head_txt.count(",") < 3:
        df0 = pd.read_csv(path, header=None)
        if df0.shape[1] == 1:
            split = df0.iloc[:, 0].astype(str).str.split(";", expand=True)
            # 尝试命名（至少有 'Country' + 若干年份）
            cols = ["Country"] + [str(y) for y in range(1960, 2101)]
            split.columns = cols[:split.shape[1]]
            year_cols = [c for c in split.columns if re.fullmatch(r"\d{4}", str(c))]
            pop_long = split.melt(id_vars="Country", var_name="Year", value_name="Population")
            pop_long["Year"] = pd.to_numeric(pop_long["Year"], errors="coerce").astype("Int64")
            pop_long["Population"] = pd.to_numeric(pop_long["Population"].str.replace(",", "", regex=False),
                                                   errors="coerce")
            return pop_long.dropna(subset=["Year"])

    # 尝试直接读一遍看看是不是已经是长表
    df_try = pd.read_csv(path, low_memory=False)
    cols = [c.strip() for c in df_try.columns.map(str)]
    lower = [c.lower() for c in cols]

    # 情况 A：长表
    if (("entity" in lower or "country" in lower) and "year" in lower):
        # 找人口列
        candidates = [c for c in cols if c.lower() in ("population", "pop", "value", "pop_total")]
        if not candidates:
            # 退而求其次：找数值列中非 year 的最后一列
            num_cols = [c for c in cols if c.lower() not in ("year", "entity", "country", "code", "iso3")]
            # 按非空数值最多来挑
            best, best_nonnull = None, -1
            for c in num_cols:
                s = pd.to_numeric(df_try[c], errors="coerce")
                nn = s.notna().sum()
                if nn > best_nonnull:
                    best, best_nonnull = c, nn
            candidates = [best] if best else []
        if not candidates:
            raise ValueError("无法在长表中识别人口列。请确认存在 Population/Value 等。")
        pop_col = candidates[0]
        # 统一列名
        if "entity" in lower:
            country_col = cols[lower.index("entity")]
        else:
            country_col = cols[lower.index("country")]
        pop_long = df_try.rename(columns={
            country_col: "Country",
            "Year": "Year",
            pop_col: "Population"
        })
        pop_long["Year"] = pd.to_numeric(pop_long["Year"], errors="coerce").astype("Int64")
        pop_long["Population"] = pd.to_numeric(pop_long["Population"], errors="coerce")
        return pop_long.dropna(subset=["Year"])

    # 情况 B：WDI 宽表，可能表头不在第一行——定位 'Country Name' 所在行再读
    # 扫描文件找出包含 'Country Name' 的那一行作为 header
    header_row = None
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "Country Name" in line or "Country,Name" in line or "Country, Name" in line:
                header_row = i
                break
    if header_row is not None:
        dfw = pd.read_csv(path, skiprows=header_row, low_memory=False)
    else:
        # 如果上面没找到，但第一行就有大量年份，可能本来就是宽表
        dfw = df_try.copy()

    dfw.columns = [c.strip().strip('"') for c in dfw.columns.map(str)]
    if "Country Name" in dfw.columns:
        dfw = dfw.rename(columns={"Country Name": "Country"})
    elif "Country" not in dfw.columns:
        # 尝试找相近的列
        for alt in ["CountryName", "Name"]:
            if alt in dfw.columns:
                dfw = dfw.rename(columns={alt: "Country"})
                break

    year_cols = [c for c in dfw.columns if re.fullmatch(r"\d{4}", str(c))]
    if year_cols:
        keep = ["Country"] + year_cols
        dfw = dfw[keep]
        # 一些 WDI 文件会有多余的尾部空列，先丢掉全空列
        dfw = dfw.dropna(axis=1, how="all")
        pop_long = dfw.melt(id_vars="Country", var_name="Year", value_name="Population")
        pop_long["Year"] = pd.to_numeric(pop_long["Year"], errors="coerce").astype("Int64")
        pop_long["Population"] = pd.to_numeric(pop_long["Population"].astype(str).str.replace(",", "", regex=False),
                                               errors="coerce")
        return pop_long.dropna(subset=["Year"])

    # 三种策略都失败了：
    raise ValueError(
        "人口文件无法识别年份列或国家列。请确认文件为：长表(Entity/Year/Population) 或 WDI 宽表(Country Name + 年份列)。")


def load_car_any(path_csv: Path, path_xlsx: Path = None):
    """车销量，既支持 csv 也支持 xlsx；返回列: Country, Year, CarSales_total"""
    if path_csv is not None and path_csv.exists():
        car = pd.read_csv(path_csv)
    elif path_xlsx is not None and path_xlsx.exists():
        car = pd.read_excel(path_xlsx, sheet_name=0)
    else:
        raise FileNotFoundError("未找到 CarSales2019-2024.csv 或 .xlsx")

    # 兼容不同列名
    rename_map = {
        "Countries": "Country", "Country": "Country",
        "Years": "Year", "Year": "Year",
        "CarSold": "CarSales_total", "CarSales_total": "CarSales_total", "Car_sold": "CarSales_total"
    }
    car = car.rename(columns={k: v for k, v in rename_map.items() if k in car.columns})
    required = ["Country", "Year", "CarSales_total"]
    missing = [c for c in required if c not in car.columns]
    if missing:
        raise ValueError(f"车销量文件缺少列: {missing}；请检查列名。")
    car["Year"] = pd.to_numeric(car["Year"], errors="coerce").astype("Int64")
    return car


# ---------- 专门处理 PM2.5 和 NO2 文件的函数 ----------
def load_air_quality_file(path: Path, pollutant: str) -> pd.DataFrame:
    """
    读取 PM2.5/NO2 CSV，清理 ="..."，并把列名标准化为：
    Country, ISO3(如有), Year, {pollutant}_mean, {pollutant}_units(如有), Region Name(如有)
    """
    # 先按字符串读，避免早期类型误判
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # 清理 ="..." / "null"
    for c in df.columns:
        df[c] = (
            df[c].astype(str).str.strip()
                .str.replace(r'^="(.*)"$', r'\1', regex=True)
                .str.replace(r'^=(.*)$', r'\1', regex=True)
                .str.replace('"', '', regex=False)
                .replace({'null': np.nan, 'NULL': np.nan, 'None': np.nan, 'NaN': np.nan, '': np.nan})
                .str.strip()
        )

    # 列名清理（去 BOM/空白 + 去掉 ="..." 套壳）  ← 新增 strip_weird
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
    df.columns = [strip_weird(c) for c in df.columns]  # ★ 关键补刀：把 ="Country" → Country，="Year" → Year

    # 可能的列名映射
    colmap = {}
    # Country / Entity / Country Name
    for cand in ["Country","Entity","Country Name","Name"]:
        if cand in df.columns:
            colmap[cand] = "Country"; break
    # Year
    for cand in ["Year","year"," YEARS "]:
        if cand in df.columns:
            colmap[cand] = "Year"; break
    # ISO3 / Code
    for cand in ["ISO3","Code","code","ISO 3"]:
        if cand in df.columns:
            colmap[cand] = "ISO3"; break
    # Region
    for cand in ["Region Name","Region name","Region"]:
        if cand in df.columns:
            colmap[cand] = "Region Name"; break
    # Exposure Mean（允许有附加词）
    exp_col = next((c for c in df.columns if 'exposure' in c.lower() and 'mean' in c.lower()), None)
    if exp_col: colmap[exp_col] = f"{pollutant}_mean"
    # Units
    unit_col = next((c for c in df.columns if 'unit' in c.lower()), None)
    if unit_col: colmap[unit_col] = f"{pollutant}_units"

    df = df.rename(columns=colmap)

    # 兜底校验
    missing_core = [c for c in ["Country","Year",f"{pollutant}_mean"] if c not in df.columns]
    if missing_core:
        raise ValueError(
            f"{path.name} 缺少关键列：{missing_core}\n实际列：{df.columns.tolist()}"
        )

    # 类型转换
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df[f"{pollutant}_mean"] = pd.to_numeric(df[f"{pollutant}_mean"], errors="coerce")
    if "ISO3" in df.columns:
        df["ISO3"] = df["ISO3"].astype(str).str.strip().str.upper().replace({'': np.nan}).astype("string")

    return df





# ---------- Load files ----------
print("正在加载数据文件...")

# 使用新的函数加载 PM2.5 和 NO2 数据
pm25 = load_air_quality_file(DATA / "tbu-global-PM2.5-concentration-by-country.csv", "PM25")
no2 = load_air_quality_file(DATA / "tbu-global-NO2-concentration-by-country.csv", "NO2")


# 立即清理掉 ="China" 这类单元格内容
pm25 = clean_excel_formula_strings(pm25)
no2  = clean_excel_formula_strings(no2)

# 加载其他数据
ev = pd.read_csv(DATA / "tbu-electric-car-sales-by-country.csv")
pop_long = load_population_any(DATA / "tbu-global-population-by-country.csv")
car = load_car_any(DATA / "CarSales2019-2024.csv", DATA / "CarSales2019-2024.xlsx")

# 清洗 EV 数据的列名
ev.columns = [strip_weird(c) for c in ev.columns]
ev = ev.rename(columns={
    "Entity": "Country",
    "Code": "ISO3",
    "Electric cars sold": "EV_sales",
    "Non-electric car sales": "NonEV_sales"
})

import re

def normalize_country(s):
    if not isinstance(s, str): return s
    x = s.strip()
    lo = re.sub(r"[.\,']", "", x.lower()).replace("  ", " ")
    # 常见别名 → 统一到完整英文
    mapping = {
        r"^(usa|u s a|u\.s\.a\.|us)$": "United States",
        r"^united states of america$": "United States",
        r"^uk$|^u\.k\.$": "United Kingdom",
        r"^czechia$": "Czech Republic",
        r"^south korea$|^republic of korea$|^korea rep$|^korea, rep$": "Korea, Rep.",
        r"^russia$": "Russian Federation",
        r"^uae$": "United Arab Emirates",
        r"^laos$": "Lao PDR",
        r"^ivory coast$|^cote d ivoire$": "Cote d'Ivoire",
        r"^macao$": "Macao SAR, China",
        r"^hong kong$": "Hong Kong SAR, China",
    }
    for pat, rep in mapping.items():
        if re.match(pat, lo): return rep
    # 首字母保持原样，不强改
    return x

for d in [pm25, no2, ev, car, pop_long]:
    if "Country" in d.columns:
        d["Country"] = d["Country"].astype(str).map(normalize_country)


# 限年 & 数值化
years = list(range(2019, 2025))


def keep_years(df):
    return df[df["Year"].isin(years)]

def keys(df):
    have = [c for c in ['ISO3','Country','Year'] if c in df.columns]
    return df[have].dropna(subset=['Year']).drop_duplicates()

# 先看 PM/NO2 均值列在自身文件里的非空
print("PM25 原表非空：", pd.to_numeric(pm25.get('PM25_mean'), errors='coerce').notna().sum())
print("NO2  原表非空：", pd.to_numeric(no2.get('NO2_mean'),  errors='coerce').notna().sum())

# 键重叠（以 car 为参照，若你用 ISO3+Year 合并，用 ISO3）
if 'ISO3' in car.columns and 'ISO3' in pm25.columns:
    k_car   = car[['ISO3','Year']].dropna().drop_duplicates()
    k_pm25  = pm25[['ISO3','Year']].dropna().drop_duplicates()
    k_no2   = no2 [['ISO3','Year']].dropna().drop_duplicates()
    print("与 car 的键重叠（ISO3,Year）- PM25:", len(k_car.merge(k_pm25, on=['ISO3','Year'])))
    print("与 car 的键重叠（ISO3,Year）- NO2 :", len(k_car.merge(k_no2,  on=['ISO3','Year'])))
else:
    # 回退到 Country+Year 检查（只是诊断）
    k_car   = car[['Country','Year']].dropna().drop_duplicates()
    k_pm25  = pm25[['Country','Year']].dropna().drop_duplicates()
    k_no2   = no2 [['Country','Year']].dropna().drop_duplicates()
    print("与 car 的键重叠（Country,Year）- PM25:", len(k_car.merge(k_pm25, on=['Country','Year'])))
    print("与 car 的键重叠（Country,Year）- NO2 :", len(k_car.merge(k_no2,  on=['Country','Year'])))



pm25, no2, ev, car, pop_long = map(keep_years, [pm25, no2, ev, car, pop_long])

# ---------- 调试：检查各表的结构 ----------
print("\n=== 各表结构检查 ===")
print(f"PM2.5 表列名: {pm25.columns.tolist()}")
print(f"NO2 表列名: {no2.columns.tolist()}")
print(f"EV 表列名: {ev.columns.tolist()}")
print(f"Car 表列名: {car.columns.tolist()}")
print(f"Population 表列名: {pop_long.columns.tolist()}")

print(f"\nPM2.5 表行数: {len(pm25)}")
print(f"NO2 表行数: {len(no2)}")
print(f"EV 表行数: {len(ev)}")
print(f"Car 表行数: {len(car)}")
print(f"Population 表行数: {len(pop_long)}")

# ---------- 数据诊断段（合并前执行） ----------
print("\n=== 数据诊断 ===")

def keys_any(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "iso3" in cols and "year" in cols:
        return df[[cols["iso3"], cols["year"]]].dropna().drop_duplicates().rename(columns={cols["iso3"]:"ISO3", cols["year"]:"Year"})
    elif "country" in cols and "year" in cols:
        return df[[cols["country"], cols["year"]]].dropna().drop_duplicates().rename(columns={cols["country"]:"Country", cols["year"]:"Year"})
    else:
        print("[WARN] 无可用键：", df.columns.tolist())
        return pd.DataFrame(columns=["Year"])

# ① 检查原表的非空行数
print("PM25 原表均值非空：", pd.to_numeric(pm25.get("PM25_mean"), errors="coerce").notna().sum())
print("NO2  原表均值非空：", pd.to_numeric(no2.get("NO2_mean"),  errors="coerce").notna().sum())

# ② 计算键重叠（看是否能跟 car 对齐）
k_car  = keys_any(car)
k_pm25 = keys_any(pm25)
k_no2  = keys_any(no2)

# 选取共同的列名
common_pm = list(set(k_car.columns) & set(k_pm25.columns))
common_no = list(set(k_car.columns) & set(k_no2.columns))

overlap_pm = k_car.merge(k_pm25, on=common_pm).shape[0] if common_pm else 0
overlap_no = k_car.merge(k_no2,  on=common_no).shape[0] if common_no else 0

print(f"与 car 键重叠 ({common_pm}) - PM25：", overlap_pm)
print(f"与 car 键重叠 ({common_no}) - NO2 ：", overlap_no)
print("======================================\n")


# 反向检查
def keys(df):
    return df[["Country","Year"]].dropna().drop_duplicates()

def anti(left, right):
    return left.merge(keys(right), on=["Country","Year"], how="left", indicator=True)\
               .query('_merge=="left_only"')[["Country","Year"]]

print("仍未能与 car 对上的 PM2.5 键：", len(anti(keys(pm25), keys(car))))
print("仍未能与 car 对上的 NO2   键：", len(anti(keys(no2),  keys(car))))


# —— 1) 构造 ISO3 映射（用空气质量 & EV 三张表，通常它们自带 ISO3）——
def norm_country_simple(s):
    if not isinstance(s, str): return s
    x = s.strip()
    x = re.sub(r"[.\,']", "", x.lower())
    x = re.sub(r"\s+", " ", x)
    # 常见别名（可按需扩展）
    if x in ("usa","u s a","us","united states of america"): return "United States"
    if x in ("uk","u k"): return "United Kingdom"
    if x in ("south korea","republic of korea","korea rep","korea, rep"): return "Korea, Rep."
    if x == "russia": return "Russian Federation"
    return s.strip()

for d in [pm25, no2, ev, car, pop_long]:
    if "Country" in d.columns:
        d["country_norm"] = d["Country"].map(norm_country_simple)

iso_map = pd.concat([
    ev[["country_norm","ISO3"]],
    pm25[["country_norm","ISO3"]] if "ISO3" in pm25.columns else pd.DataFrame(columns=["country_norm","ISO3"]),
    no2[["country_norm","ISO3"]]  if "ISO3" in no2.columns  else pd.DataFrame(columns=["country_norm","ISO3"]),
], ignore_index=True).dropna().drop_duplicates()

# 手动兜底（如仍有个别国缺 ISO3）
manual_iso = {"United States":"USA","China":"CHN","India":"IND","Japan":"JPN","Germany":"DEU",
              "United Kingdom":"GBR","Korea, Rep.":"KOR","France":"FRA","Canada":"CAN","Mexico":"MEX"}

# —— 2) 给 car / pop_long 回填 ISO3 ——
def attach_iso3(df):
    if "ISO3" not in df.columns:
        df["ISO3"] = pd.NA
    df = df.merge(iso_map, on="country_norm", how="left", suffixes=("","_map"))
    df["ISO3"] = df["ISO3"].fillna(df["ISO3_map"])
    df.drop(columns=[c for c in df.columns if c.endswith("_map")], inplace=True)
    # 手动兜底
    df.loc[df["ISO3"].isna(), "ISO3"] = df.loc[df["ISO3"].isna(), "country_norm"].map(manual_iso)
    return df

car = attach_iso3(car)
pop_long = attach_iso3(pop_long)
pm25 = attach_iso3(pm25)
no2  = attach_iso3(no2)
ev   = attach_iso3(ev)

# —— 3) 统一键 dtype ——
for d in [pm25,no2,ev,car,pop_long]:
    d["ISO3"] = d["ISO3"].astype("string")
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")

# —— 4) 用 ISO3+Year 合并（把你现在的 5 次 merge 改成下面这段）——
merged = (car
    .merge(ev  [["ISO3","Year","EV_sales","NonEV_sales"]],                on=["ISO3","Year"], how="left")
    .merge(pm25[["ISO3","Year","PM25_mean","PM25_units","Region Name"]],  on=["ISO3","Year"], how="left")
    .merge(no2 [["ISO3","Year","NO2_mean","NO2_units"]],                  on=["ISO3","Year"], how="left")
    .merge(pop_long[["ISO3","Year","Population","Country"]],              on=["ISO3","Year"], how="left",
           suffixes=("","_pop"))
)

# 展示用国家名：优先 car，其次人口表
if "Country_pop" in merged.columns:
    merged["Country"] = merged["Country"].fillna(merged["Country_pop"])
    merged.drop(columns=["Country_pop"], inplace=True)



# ---------- 合并 ----------
# print("\n=== 开始合并数据 ===")
#
# # 逐步合并以便调试
# merged = car.copy()
# print(f"初始 car 表行数: {len(merged)}")
#
# merged = merged.merge(ev[["Country", "Year", "EV_sales", "NonEV_sales", "ISO3"]], on=["Country", "Year"], how="left")
# print(f"合并 EV 后行数: {len(merged)}")
#
# merged = merged.merge(pm25[["Country", "Year", "PM25_mean", "PM25_units", "Region Name"]], on=["Country", "Year"],
#                       how="left")
# print(f"合并 PM2.5 后行数: {len(merged)}")
#
# merged = merged.merge(no2[["Country", "Year", "NO2_mean", "NO2_units"]], on=["Country", "Year"], how="left")
# print(f"合并 NO2 后行数: {len(merged)}")
#
# merged = merged.merge(pop_long[["Country", "Year", "Population"]], on=["Country", "Year"], how="left")
# print(f"合并 Population 后行数: {len(merged)}")

# 检查合并后是否有 PM2.5 和 NO2 数据
print(f"\n=== 合并后数据质量检查 ===")
print(f"PM2.5 非空值数量: {merged['PM25_mean'].notna().sum()}")
print(f"NO2 非空值数量: {merged['NO2_mean'].notna().sum()}")
print(f"EV_sales 非空值数量: {merged['EV_sales'].notna().sum()}")
print(f"Population 非空值数量: {merged['Population'].notna().sum()}")

# ---------- 派生前统一数值格式 ----------
print("\n正在统一数值格式...")
for col in ["EV_sales", "CarSales_total", "Population"]:
    if col in merged.columns:
        merged[col] = (
            merged[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

# ---------- 派生 ----------
print("正在计算派生指标...")
merged["EV_share"] = merged["EV_sales"] / merged["CarSales_total"]
merged["CarSales_per_1k_people"] = merged["CarSales_total"] / merged["Population"] * 1000
merged["EV_sales_per_1k"] = merged["EV_sales"] / merged["Population"] * 1000

# ---------- 输出 ----------
out_path = DATA / "storyboard2_airquality_cars_tidy_2019_2024.csv"
merged.to_csv(out_path, index=False, encoding="utf-8")
print(f"\n✅ 整洁数据已生成: {out_path}")
print(f"最终数据形状: {merged.shape}")
print(f"最终数据列名: {merged.columns.tolist()}")

print("\n前10行数据预览:")
print(merged.head(10).to_string(index=False))

print(f"\n数据已成功保存至: {out_path}")

# 快速自检
print("PM25非空:", merged["PM25_mean"].notna().sum(), "NO2非空:", merged["NO2_mean"].notna().sum())
print(merged[merged["ISO3"]=="USA"][["Year","PM25_mean","NO2_mean"]].sort_values("Year").head(10))