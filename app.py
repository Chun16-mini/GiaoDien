# -*- coding: utf-8 -*-
"""
Dự báo doanh số bán hàng bằng ARIMA / SARIMA  (Giao diện Streamlit – Người 7)

Chạy:   streamlit run app.py
Cần để cùng thư mục: sales_monthly_clean.csv (dữ liệu mẫu của nhóm, phân tách bằng ';')
"""
import io
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Hằng số
# ----------------------------------------------------------------------------
SAMPLE_FILE = Path(__file__).parent / "sales_monthly_clean.csv"

INK = "#14213D"        # dữ liệu thực tế
TEAL = "#0F766E"       # xu hướng / mô hình chính
ROSE = "#D1495B"       # dự báo
GREY = "#8A94A6"
MODEL_COLORS = [ROSE, TEAL, "#C77D0A", "#5B5BD6"]

PRESET = {
    "ARIMA": ((1, 1, 1), None),
    "SARIMA": ((0, 1, 1), (0, 1, 1, 12)),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp li,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p {
    font-family: 'Be Vietnam Pro', sans-serif;
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.stApp h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
.sub { color: #5B6578; font-size: 1.02rem; margin-bottom: 1.4rem; }
[data-testid="stMetricValue"] { font-weight: 600; }
[data-testid="stMetricLabel"] p { color: #5B6578; font-size: 0.86rem; }
.stTabs [data-baseweb="tab"] { font-family: 'Be Vietnam Pro', sans-serif; font-weight: 500; }
</style>
"""


# ----------------------------------------------------------------------------
# Đọc và chuẩn hoá dữ liệu
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_table(raw: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="latin-1")
    df.columns = df.columns.astype(str).str.strip()
    return df


def _to_number(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.replace(",", "", regex=False), errors="coerce")


def guess_columns(df: pd.DataFrame):
    """Đoán cột thời gian và cột doanh số để điền sẵn."""
    cols = list(df.columns)
    date_keys = ("date", "month", "time", "ngày", "ngay", "tháng", "thang")
    date_col = next((c for c in cols if any(k in c.lower() for k in date_keys)), cols[0])
    numeric = [c for c in cols if c != date_col and _to_number(df[c]).notna().mean() > 0.8]
    sales_keys = ("sale", "revenue", "doanh", "amount")
    sales_col = next((c for c in numeric if any(k in c.lower() for k in sales_keys)),
                     numeric[0] if numeric else cols[-1])
    return date_col, sales_col


@st.cache_data(show_spinner=False)
def build_series(df: pd.DataFrame, date_col: str, sales_col: str):
    """Trả về (chuỗi theo tháng, số dòng bị bỏ, số tháng thiếu đã nội suy)."""
    d = df[[date_col, sales_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d[sales_col] = _to_number(d[sales_col])
    dropped = int(d.isna().any(axis=1).sum())
    d = d.dropna()
    if d.empty:
        return None, dropped, 0
    d[date_col] = d[date_col].dt.to_period("M").dt.to_timestamp()
    s = d.groupby(date_col)[sales_col].sum().sort_index().asfreq("MS")
    missing = int(s.isna().sum())
    if missing:
        s = s.interpolate().ffill().bfill()
    s.name = "sales"
    return s, dropped, missing


# ----------------------------------------------------------------------------
# Mô hình và chỉ số đánh giá
# ----------------------------------------------------------------------------
def fmt_spec(kind, order, seas):
    if kind == "ARIMA":
        return f"ARIMA({order[0]},{order[1]},{order[2]})"
    return f"SARIMA({order[0]},{order[1]},{order[2]})({seas[0]},{seas[1]},{seas[2]})[12]"


def metrics(y, f):
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    e = y - f
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAPE (%)": float((np.abs(e) / np.abs(y)).mean() * 100),
        "sMAPE (%)": float((2 * np.abs(e) / (np.abs(y) + np.abs(f))).mean() * 100),
    }


def baselines(train: pd.Series, test: pd.Series):
    return {
        "Naive (giá trị cuối)": np.full(len(test), train.iloc[-1], dtype=float),
        "Seasonal naive (cùng tháng năm trước)": np.resize(train.iloc[-12:].to_numpy(dtype=float), len(test)),
    }


@st.cache_data(show_spinner=False)
def run_forecast(series: pd.Series, kind: str, order: tuple, seas, steps: int):
    if kind == "ARIMA":
        res = ARIMA(series, order=order).fit()
    else:
        res = SARIMAX(series, order=order, seasonal_order=seas,
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=300)
    fc = res.get_forecast(steps)
    ci = fc.conf_int(alpha=0.05)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    out = pd.DataFrame({
        "forecast": np.asarray(fc.predicted_mean, dtype=float),
        "lower": np.asarray(ci.iloc[:, 0], dtype=float),
        "upper": np.asarray(ci.iloc[:, 1], dtype=float),
    }, index=idx)
    return out, float(res.aic), float(res.bic)


@st.cache_data(show_spinner=False)
def auto_search(train: pd.Series, kind: str):
    """Tìm tham số có AIC nhỏ nhất (lưới nhỏ, giống cách làm ở Chương 4)."""
    best = None
    if kind == "ARIMA":
        for p, q in itertools.product(range(3), range(3)):
            try:
                aic = ARIMA(train, order=(p, 1, q)).fit().aic
            except Exception:
                continue
            if np.isfinite(aic) and (best is None or aic < best[0]):
                best = (aic, (p, 1, q), None)
    else:
        for p, q, P, Q in itertools.product(range(2), range(2), range(2), range(2)):
            try:
                aic = SARIMAX(train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                              enforce_stationarity=False, enforce_invertibility=False
                              ).fit(disp=False, maxiter=200).aic
            except Exception:
                continue
            if np.isfinite(aic) and (best is None or aic < best[0]):
                best = (aic, (p, 1, q), (P, 1, Q, 12))
    if best is None:
        return PRESET[kind]
    return best[1], best[2]


@st.cache_data(show_spinner=False)
def adf_table(series: pd.Series) -> pd.DataFrame:
    variants = {
        "Chuỗi gốc": series,
        "Sai phân bậc 1 (d=1)": series.diff(),
        "Sai phân mùa vụ (D=1, s=12)": series.diff(12),
        "Sai phân d=1 và D=1": series.diff().diff(12),
    }
    rows = []
    for name, x in variants.items():
        x = x.dropna()
        try:
            stat, p = adfuller(x, autolag="AIC")[:2]
            verdict = "Dừng" if p < 0.05 else "Chưa dừng"
            rows.append({"Chuỗi": name, "Thống kê ADF": stat, "p-value": p, "Kết luận (mức 5%)": verdict})
        except Exception:
            rows.append({"Chuỗi": name, "Thống kê ADF": np.nan, "p-value": np.nan,
                         "Kết luận (mức 5%)": "Không đủ dữ liệu"})
    return pd.DataFrame(rows)


def plausible(fc: pd.DataFrame, series: pd.Series) -> bool:
    v = fc["forecast"].to_numpy()
    return bool(np.isfinite(v).all() and v.min() > 0 and v.max() < 3 * series.max())


# ----------------------------------------------------------------------------
# Vẽ biểu đồ
# ----------------------------------------------------------------------------
def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def fig_history(series: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(series.index, series.values, color=INK, lw=1.8, label="Doanh số")
    if len(series) >= 12:
        ax.plot(series.index, series.rolling(12).mean(), color=TEAL, lw=2.2,
                label="Trung bình trượt 12 tháng (xu hướng)")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_seasonal(series: pd.Series):
    ratio = series.groupby(series.index.month).mean() / series.mean()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar([f"T{m}" for m in ratio.index], ratio.values,
           color=[TEAL if v >= 1 else GREY for v in ratio.values])
    ax.axhline(1, color=INK, lw=1)
    style_ax(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_ylabel("so với trung bình cả kỳ", fontsize=9)
    fig.tight_layout()
    return fig


def fig_decompose(series: pd.Series, mode: str):
    dec = seasonal_decompose(series, model=mode, period=12)
    parts = [("Dữ liệu", dec.observed), ("Xu hướng", dec.trend),
             ("Mùa vụ", dec.seasonal), ("Phần dư", dec.resid)]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.4), sharex=True)
    for ax, (title, s) in zip(axes, parts):
        ax.plot(s.index, s.values, color=INK if title == "Dữ liệu" else TEAL, lw=1.5)
        ax.set_title(title, fontsize=10, loc="left")
        style_ax(ax)
    fig.tight_layout()
    return fig


def fig_eval(train, test, preds):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(train.index[-24:], train.values[-24:], color=GREY, lw=1.6, label="Huấn luyện")
    ax.plot(test.index, test.values, color=INK, lw=2.4, label="Thực tế (kiểm tra)")
    for color, (name, p) in zip(MODEL_COLORS + [GREY, GREY], preds.items()):
        ax.plot(test.index, p, ls="--", lw=1.6, color=color, label=name)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def fig_future(series: pd.Series, fc: pd.DataFrame, label: str):
    hist = series.iloc[-36:]
    xs = [hist.index[-1]] + list(fc.index)
    ys = [hist.iloc[-1]] + list(fc["forecast"])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(hist.index, hist.values, color=INK, lw=1.9, label="Doanh số thực tế")
    ax.plot(xs, ys, color=ROSE, ls="--", marker="o", ms=4, lw=1.9, label=f"Dự báo {label}")
    ax.fill_between(fc.index, np.maximum(fc["lower"], 0), fc["upper"],
                    color=ROSE, alpha=0.14, label="Khoảng tin cậy 95%")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Giao diện
# ----------------------------------------------------------------------------
def sidebar_data():
    """Chọn nguồn dữ liệu và cột. Trả về (df, tên nguồn)."""
    st.sidebar.header("Dữ liệu")
    options = (["Dữ liệu mẫu của nhóm"] if SAMPLE_FILE.exists() else []) + ["Tải file lên"]
    source = st.sidebar.radio("Nguồn dữ liệu", options) if len(options) > 1 else options[0]

    if source == "Dữ liệu mẫu của nhóm":
        return read_table(SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name), SAMPLE_FILE.name

    up = st.sidebar.file_uploader("Chọn file CSV hoặc Excel", type=["csv", "xlsx", "xls"])
    if up is None:
        st.info("Chọn file CSV hoặc Excel ở thanh bên trái. File cần có một cột ngày/tháng "
                "và một cột doanh số.")
        st.stop()
    return read_table(up.getvalue(), up.name), up.name


def sidebar_columns(df, source_name):
    date_guess, sales_guess = guess_columns(df)
    cols = list(df.columns)
    with st.sidebar.expander("Cột dữ liệu", expanded=False):
        date_col = st.selectbox("Cột thời gian", cols, index=cols.index(date_guess),
                                key=f"date_{source_name}")
        sales_col = st.selectbox("Cột doanh số", cols, index=cols.index(sales_guess),
                                 key=f"sales_{source_name}")
    return date_col, sales_col


def sidebar_model(n_obs):
    st.sidebar.header("Mô hình")
    mode_label = st.sidebar.radio(
        "Cách chọn tham số",
        ["Mặc định", "Tự động (AIC thấp nhất)", "Nhập tay"],
        help="Mặc định: ARIMA(1,1,1) và SARIMA(0,1,1)(0,1,1)[12]. "
             "Tự động: thử một lưới tham số nhỏ trên tập huấn luyện, chọn AIC thấp nhất (chạy lâu hơn).",
    )
    mode = {"Mặc định": "preset", "Tự động (AIC thấp nhất)": "auto", "Nhập tay": "manual"}[mode_label]

    manual = dict(PRESET)
    if mode == "manual":
        st.sidebar.caption("ARIMA (p, d, q)")
        a = st.sidebar.columns(3)
        p = a[0].number_input("p", 0, 5, 1, key="a_p")
        d = a[1].number_input("d", 0, 2, 1, key="a_d")
        q = a[2].number_input("q", 0, 5, 1, key="a_q")
        st.sidebar.caption("SARIMA (p, d, q)(P, D, Q), chu kỳ 12")
        b = st.sidebar.columns(3)
        sp = b[0].number_input("p", 0, 5, 0, key="s_p")
        sd = b[1].number_input("d", 0, 2, 1, key="s_d")
        sq = b[2].number_input("q", 0, 5, 1, key="s_q")
        c = st.sidebar.columns(3)
        sP = c[0].number_input("P", 0, 2, 0, key="s_P")
        sD = c[1].number_input("D", 0, 1, 1, key="s_D")
        sQ = c[2].number_input("Q", 0, 2, 1, key="s_Q")
        manual = {"ARIMA": ((int(p), int(d), int(q)), None),
                  "SARIMA": ((int(sp), int(sd), int(sq)), (int(sP), int(sD), int(sQ), 12))}

    st.sidebar.header("Thiết lập")
    test_n = st.sidebar.slider("Số tháng kiểm tra (đánh giá)", 3, min(18, n_obs - 12), min(12, n_obs - 12))
    steps = st.sidebar.slider("Số tháng dự báo tương lai", 1, 24, 12)
    return mode, manual, test_n, steps


def resolve_specs(mode, train, kinds, manual):
    specs = {}
    for kind in kinds:
        if mode == "auto":
            with st.spinner(f"Đang tìm tham số {kind} theo AIC..."):
                specs[kind] = auto_search(train, kind)
        elif mode == "manual":
            specs[kind] = manual[kind]
        else:
            specs[kind] = PRESET[kind]
    return specs


def tab_data(series, df, date_col, sales_col, dropped, missing):
    c = st.columns(4)
    c[0].metric("Số tháng", len(series))
    c[1].metric("Giai đoạn", f"{series.index[0]:%m/%Y} – {series.index[-1]:%m/%Y}")
    c[2].metric("Doanh số trung bình", f"{series.mean():,.0f}")
    c[3].metric("Tháng cao nhất", f"{series.idxmax():%m/%Y}", f"{series.max():,.0f}", delta_color="off")

    if len(series) >= 24:
        last, prev = series.iloc[-12:].sum(), series.iloc[-24:-12].sum()
        st.write(f"12 tháng gần nhất đạt **{last:,.0f}**, "
                 f"{'tăng' if last >= prev else 'giảm'} **{abs(last / prev - 1) * 100:.1f}%** "
                 f"so với 12 tháng trước đó.")
    if dropped:
        st.warning(f"Đã bỏ {dropped} dòng không đọc được ngày hoặc doanh số.")
    if missing:
        st.warning(f"Thiếu {missing} tháng; app nội suy tuyến tính để mô hình chạy được.")

    show(fig_history(series))
    with st.expander("Xem bảng dữ liệu"):
        view = series.reset_index()
        view.columns = ["Tháng", "Doanh số"]
        view["Tháng"] = view["Tháng"].dt.strftime("%m/%Y")
        st.dataframe(view, hide_index=True)


def tab_analysis(series):
    st.markdown("#### Tính mùa vụ")
    st.caption("Doanh số trung bình của từng tháng chia cho trung bình cả kỳ. "
               "Cột trên 1,00 là tháng thường bán tốt hơn mức trung bình.")
    show(fig_seasonal(series))

    st.markdown("#### Phân rã chuỗi")
    mode = st.radio("Kiểu phân rã", ["additive", "multiplicative"], horizontal=True,
                    format_func=lambda m: "Cộng (additive)" if m == "additive" else "Nhân (multiplicative)")
    show(fig_decompose(series, mode))

    st.markdown("#### Kiểm định tính dừng (ADF)")
    table = adf_table(series)
    st.dataframe(table.style.format({"Thống kê ADF": "{:.3f}", "p-value": "{:.4f}"}, na_rep="–"),
                 hide_index=True)
    p0 = table.loc[0, "p-value"]
    if pd.notna(p0):
        if p0 < 0.05:
            st.success("Chuỗi gốc đã dừng ở mức 5%.")
        else:
            st.info("Chuỗi gốc chưa dừng (p-value ≥ 0,05), nên mô hình dùng sai phân (d=1; "
                    "và D=1 với SARIMA) để làm chuỗi dừng.")


def tab_eval(series, test_n, mode, manual):
    train, test = series.iloc[:-test_n], series.iloc[-test_n:]
    st.write(f"Huấn luyện trên **{len(train)} tháng** ({train.index[0]:%m/%Y} – {train.index[-1]:%m/%Y}), "
             f"kiểm tra trên **{test_n} tháng** cuối ({test.index[0]:%m/%Y} – {test.index[-1]:%m/%Y}).")

    kinds = ["ARIMA"] + (["SARIMA"] if len(train) >= 36 else [])
    if "SARIMA" not in kinds:
        st.info("SARIMA cần ít nhất 36 tháng huấn luyện (3 chu kỳ mùa vụ), nên tạm thời chỉ chạy ARIMA.")
    specs = resolve_specs(mode, train, kinds, manual)

    rows, preds, spec_of = {}, {}, {}
    for kind in kinds:
        order, seas = specs[kind]
        label = fmt_spec(kind, order, seas)
        try:
            fc, aic, bic = run_forecast(train, kind, order, seas, test_n)
        except Exception as e:
            st.error(f"{label} không chạy được: {e}")
            continue
        preds[label] = fc["forecast"].to_numpy()
        spec_of[label] = (kind, order, seas)
        rows[label] = {"Loại": "Mô hình", **metrics(test.values, preds[label]), "AIC": aic, "BIC": bic}

    for name, p in baselines(train, test).items():
        rows[name] = {"Loại": "Mốc so sánh", **metrics(test.values, p), "AIC": np.nan, "BIC": np.nan}

    if not preds:
        st.stop()

    table = pd.DataFrame(rows).T
    num_cols = ["MAE", "RMSE", "MAPE (%)", "sMAPE (%)", "AIC", "BIC"]
    table[num_cols] = table[num_cols].astype(float)
    table = table.sort_values("RMSE")

    best = min(preds, key=lambda k: rows[k]["RMSE"])
    st.markdown("#### So sánh sai số")
    st.dataframe(
        table.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}",
                            "sMAPE (%)": "{:.2f}", "AIC": "{:.1f}", "BIC": "{:.1f}"}, na_rep="–")
        .highlight_min(subset=["MAE", "RMSE", "MAPE (%)", "sMAPE (%)"], color="#CFE8E4"))
    st.caption("Ô tô màu là giá trị nhỏ nhất của từng cột (càng nhỏ càng tốt).")
    st.download_button("Tải bảng đánh giá (CSV)",
                       table.round(4).to_csv().encode("utf-8-sig"),
                       file_name="danh_gia_mo_hinh.csv", mime="text/csv")

    st.success(f"Mô hình có RMSE thấp nhất là **{best}**, sai số trung bình khoảng "
               f"**{rows[best]['MAPE (%)']:.1f}%** (MAPE) trên {test_n} tháng kiểm tra.")
    snaive = "Seasonal naive (cùng tháng năm trước)"
    if rows[snaive]["RMSE"] < rows[best]["RMSE"]:
        st.warning("Mô hình chưa tốt hơn cách dự báo đơn giản “lặp lại cùng tháng năm trước”. "
                   "Nên thử tham số khác hoặc xem lại dữ liệu trước khi tin vào dự báo.")

    show(fig_eval(train, test, {**preds, **baselines(train, test)}))

    with st.expander("Cách đọc các chỉ số"):
        st.markdown(
            "- **MAE**: sai số tuyệt đối trung bình, cùng đơn vị với doanh số.\n"
            "- **RMSE**: giống MAE nhưng phạt nặng các tháng lệch lớn.\n"
            "- **MAPE / sMAPE**: sai số tính theo phần trăm, dễ so sánh giữa các bộ dữ liệu.\n"
            "- **AIC / BIC**: đo độ phù hợp của mô hình trên tập huấn luyện, có phạt số tham số; "
            "dùng để chọn tham số, không thay cho sai số trên tập kiểm tra.\n"
            "- **Mốc so sánh**: nếu mô hình không thắng được các mốc này thì chưa đáng dùng."
        )
    return specs, best, spec_of


def tab_forecast(series, steps, specs, best, spec_of):
    labels = list(spec_of)
    choice = st.selectbox("Mô hình dự báo", labels, index=labels.index(best) if best in labels else 0,
                          help="Mặc định là mô hình có RMSE thấp nhất ở tab Đánh giá.")
    kind, order, seas = spec_of[choice]
    with st.spinner("Đang tính dự báo..."):
        try:
            fc, aic, bic = run_forecast(series, kind, order, seas, steps)
        except Exception as e:
            st.error(f"Mô hình không chạy được trên dữ liệu này: {e}")
            return

    if not plausible(fc, series):
        st.warning("Dự báo có giá trị bất thường (âm hoặc lớn gấp 3 lần mức cao nhất trong quá khứ). "
                   "Hãy thử mô hình hoặc tham số khác trước khi dùng kết quả này.")

    recent = series.iloc[-steps:].mean()
    change = (fc["forecast"].mean() / recent - 1) * 100
    peak = fc["forecast"].idxmax()
    c = st.columns(3)
    c[0].metric("Dự báo trung bình mỗi tháng", f"{fc['forecast'].mean():,.0f}")
    c[1].metric(f"So với {steps} tháng gần nhất", f"{change:+.1f}%")
    c[2].metric("Tháng dự báo cao nhất", f"{peak:%m/%Y}", f"{fc['forecast'].max():,.0f}", delta_color="off")

    show(fig_future(series, fc, choice))
    st.caption("Khoảng tin cậy càng xa càng rộng vì sai số tích lũy theo thời gian.")
    if kind == "ARIMA":
        st.caption("ARIMA không có thành phần mùa vụ nên đường dự báo thường phẳng hơn SARIMA.")

    out = pd.DataFrame({
        "Tháng": fc.index.strftime("%m/%Y"),
        "Doanh số dự báo": fc["forecast"].round(0),
        "Cận dưới 95%": fc["lower"].round(0),
        "Cận trên 95%": fc["upper"].round(0),
    })
    st.dataframe(out.style.format({"Doanh số dự báo": "{:,.0f}", "Cận dưới 95%": "{:,.0f}",
                                   "Cận trên 95%": "{:,.0f}"}), hide_index=True)
    st.download_button("Tải kết quả dự báo (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="du_bao_doanh_so.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="Dự báo doanh số", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Dự báo doanh số bán hàng")
    st.markdown('<div class="sub">Phân tích chuỗi thời gian và dự báo theo tháng bằng ARIMA và SARIMA.</div>',
                unsafe_allow_html=True)

    df, source_name = sidebar_data()
    if df.shape[1] < 2:
        st.error("File cần có ít nhất hai cột: thời gian và doanh số.")
        st.stop()
    date_col, sales_col = sidebar_columns(df, source_name)
    if date_col == sales_col:
        st.error("Cột thời gian và cột doanh số đang trùng nhau. Hãy chọn lại ở mục “Cột dữ liệu”.")
        st.stop()

    series, dropped, missing = build_series(df, date_col, sales_col)
    if series is None:
        st.error("Không đọc được dữ liệu. Kiểm tra lại cột thời gian (ví dụ 01/03/2017) và cột doanh số (số).")
        st.stop()
    if len(series) < 24:
        st.error(f"File có {len(series)} tháng. Cần ít nhất 24 tháng để phân tích và đánh giá mô hình.")
        st.stop()

    mode, manual, test_n, steps = sidebar_model(len(series))

    t1, t2, t3, t4 = st.tabs(["Dữ liệu", "Phân tích", "Đánh giá mô hình", "Dự báo"])
    with t1:
        tab_data(series, df, date_col, sales_col, dropped, missing)
    with t2:
        tab_analysis(series)
    with t3:
        specs, best, spec_of = tab_eval(series, test_n, mode, manual)
    with t4:
        tab_forecast(series, steps, specs, best, spec_of)


if __name__ == "__main__":
    main()
