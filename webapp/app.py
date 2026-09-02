"""DITP-Analysis Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_service import AnalysisResult, analyze_bytes, recalculate_boundary_outputs
from webapp.plots import (
    alltrace_figure,
    alltrace_heatmap_figure,
    alltrace_traces_figure,
    cluster_conductance_figure,
    cluster_heatmap_figure,
    cluster_traces_figure,
    length_histogram_figure,
    representative_figure,
    representative_heatmap_figure,
    cluster_color,
)


st.set_page_config(page_title="DITP-Analysis", page_icon=":material/analytics:", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ditp-ink: #172321;
        --ditp-muted: #60706D;
        --ditp-line: #D7E0DE;
        --ditp-primary: #116B66;
        --ditp-accent: #A33A4A;
        --ditp-bg: #F3F6F5;
    }
    html, body, .stApp {
        font-family: Inter, "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
    }
    [data-testid="stIconMaterial"],
    .material-symbols-rounded {
        font-family: "Material Symbols Rounded" !important;
    }
    .stApp {
        background: var(--ditp-bg);
        color: var(--ditp-ink);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, h4 {
        color: var(--ditp-ink);
        letter-spacing: 0;
    }
    h1 { font-size: 2.35rem !important; line-height: 1.12 !important; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.18rem !important; }
    .ditp-hero {
        padding: 1.1rem 0 1.3rem 0;
        border-bottom: 1px solid var(--ditp-line);
        margin-bottom: 1.4rem;
    }
    .ditp-eyebrow {
        color: var(--ditp-primary);
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .ditp-hero-title {
        color: var(--ditp-ink);
        font-size: 2.45rem;
        font-weight: 780;
        line-height: 1.1;
        letter-spacing: 0;
        margin: 0;
    }
    .ditp-hero-copy {
        color: var(--ditp-muted);
        font-size: 1rem;
        line-height: 1.7;
        margin: 0.7rem 0 0 0;
        max-width: 680px;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF;
        border: 1px dashed #8CA6A1;
        border-radius: 6px;
        min-height: 132px;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.2rem;
        text-align: center;
    }
    [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploaderDropzoneInstructions"] {
        align-items: center;
        text-align: center;
    }
    [data-testid="stFileUploaderDropzone"] button {
        margin-left: auto;
        margin-right: auto;
        transform: translateY(0.7rem);
    }
    [data-testid="stFileUploader"] button[aria-label="Add files"] {
        display: none;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.62);
        border: 1px solid var(--ditp-line);
        border-radius: 6px;
    }
    [data-testid="stSelectbox"] label p {
        color: var(--ditp-primary);
        font-size: 1rem;
        font-weight: 750;
    }
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button {
        background: var(--ditp-primary);
        border-color: var(--ditp-primary);
        border-radius: 5px;
        min-height: 2.7rem;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #0A5753;
        border-color: #0A5753;
    }
    .ditp-section-divider {
        border-top: 1px solid var(--ditp-line);
        margin: 2.1rem 0 1.25rem 0;
    }
    .ditp-cluster-heading {
        border-left: 5px solid;
        border-radius: 4px;
        padding: 0.65rem 0.85rem;
        margin: 1.25rem 0 0.35rem 0;
        font-weight: 700;
        color: var(--ditp-ink);
    }
    .ditp-sidebar-brand {
        color: var(--ditp-ink);
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0.15rem 0 0.8rem 0;
    }
    [data-testid="stSidebar"] {
        background: #EAF0EE;
        border-right: 1px solid var(--ditp-line);
    }
    [data-testid="stSidebar"] h3 {
        color: var(--ditp-ink);
        margin-bottom: 0.25rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 0.45rem 0.55rem;
        transition: background 120ms ease, border-color 120ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #DDE8E5;
        border-color: #B9CBC7;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #D3E7E3;
        border-color: #6FA6A0;
        color: #0B5854;
        font-weight: 650;
    }
    [data-testid="stMetricValue"] {
        color: var(--ditp-ink);
        font-size: 1.65rem;
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1.2rem; }
        .ditp-hero-title { font-size: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _length_rows(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for cluster_id in range(result.n_clusters):
        values = np.asarray([item["length_nm"] for item in result.lengths[cluster_id]], dtype=float)
        rejected = sum(result.rejection_counts[cluster_id].values())
        rows.append(
            {
                "Cluster": f"Cluster {cluster_id + 1}",
                "平均长度 (nm)": float(np.mean(values)) if values.size else np.nan,
                "中位数 (nm)": float(np.median(values)) if values.size else np.nan,
                "标准差 (nm)": float(np.std(values)) if values.size else np.nan,
                "有效轨迹": int(values.size),
                "拒绝轨迹": int(rejected),
                "接受率": float(values.size / (values.size + rejected)) if values.size + rejected else 0.0,
                "拒绝原因": "；".join(f"{key}: {value}" for key, value in result.rejection_counts[cluster_id].items()) or "无",
            }
        )
    return pd.DataFrame(rows)


def _cluster_overview_rows(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for stats in result.cluster_stats:
        rows.append(
            {
                "Cluster": f"Cluster {int(stats['cluster_id'])}",
                "轨迹数量": int(stats["count"]),
                "轨迹占比": float(stats["fraction"]),
                "峰均值": float(stats["peak_mean"]),
                "峰标准差": float(stats["peak_std"]),
            }
        )
    return pd.DataFrame(rows)


def _trace_length_csv(result: AnalysisResult) -> bytes:
    rows = []
    for cluster_id, items in result.lengths.items():
        for item in items:
            rows.append(
                {
                    "trace_id": item["trace_id"],
                    "cluster": cluster_id + 1,
                    "length_nm": item["length_nm"],
                    "slope": item["slope"],
                }
            )
    frame = pd.DataFrame(rows, columns=["trace_id", "cluster", "length_nm", "slope"])
    return frame.to_csv(index=False).encode("utf-8")


def _cluster_csv(result: AnalysisResult) -> bytes:
    frame = pd.DataFrame({"trace_id": result.trace_ids, "cluster_id": result.labels + 1})
    return frame.to_csv(index=False).encode("utf-8")


# Plotly figures are rebuilt on every Streamlit rerun unless their inputs are
# cached. The result object is session-scoped, so its identity is a stable and
# safe cache key while the user switches navigation items or selectors.
@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_alltrace_figure(result: AnalysisResult):
    return alltrace_figure(result)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_alltrace_heatmap_figure(result: AnalysisResult):
    return alltrace_heatmap_figure(result)


@st.cache_data(show_spinner=False, max_entries=32, hash_funcs={AnalysisResult: id})
def _cached_alltrace_traces_figure(result: AnalysisResult, max_traces: int):
    return alltrace_traces_figure(result, max_traces=max_traces)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_cluster_conductance_figure(result: AnalysisResult, cluster_id: int):
    return cluster_conductance_figure(result, cluster_id)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_cluster_heatmap_figure(result: AnalysisResult, cluster_id: int):
    return cluster_heatmap_figure(result, cluster_id)


@st.cache_data(show_spinner=False, max_entries=32, hash_funcs={AnalysisResult: id})
def _cached_cluster_traces_figure(result: AnalysisResult, cluster_id: int, max_traces: int):
    return cluster_traces_figure(result, cluster_id, max_traces=max_traces)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_length_histogram_figure(result: AnalysisResult, cluster_id: int):
    return length_histogram_figure(result, cluster_id)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_representative_figure(result: AnalysisResult, cluster_id: int):
    return representative_figure(result, cluster_id)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_representative_heatmap_figure(result: AnalysisResult, cluster_id: int):
    return representative_heatmap_figure(result, cluster_id)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_cluster_csv(result: AnalysisResult) -> bytes:
    return _cluster_csv(result)


@st.cache_data(show_spinner=False, hash_funcs={AnalysisResult: id})
def _cached_trace_length_csv(result: AnalysisResult) -> bytes:
    return _trace_length_csv(result)


def _show_overview(result: AnalysisResult, max_display: int) -> None:
    boundary = result.boundary
    st.subheader("AllTrace 总体概览")
    overall_traces, overall_heatmap = st.columns(2)
    with overall_traces:
        st.plotly_chart(_cached_alltrace_traces_figure(result, max_display), width="stretch")
    with overall_heatmap:
        st.plotly_chart(_cached_alltrace_heatmap_figure(result), width="stretch")

    st.markdown("#### AllTrace 一维电导与平台-噪音边界")
    conductance_column, boundary_column = st.columns([1.7, 1.0])
    with conductance_column:
        st.plotly_chart(_cached_alltrace_figure(result), width="stretch")
    with boundary_column:
        st.markdown("#### 总体电导统计")
        boundary_cols = st.columns(2)
        boundary_cols[0].metric("平台峰", f"{boundary['platform_peak_log_g_g0']:.3f}")
        boundary_cols[1].metric("背景峰", f"{boundary['background_peak_log_g_g0']:.3f}")
        boundary_cols[0].metric("平台-噪音边界", f"{boundary['boundary_log_g_g0']:.3f}")
        boundary_cols[1].metric("轨迹数量", f"{len(result.traces):,}")
        st.caption("自动边界来自背景峰与平台峰之间的平台侧低谷。修改后重算电导峰统计和平台长度。")
        selected_boundary = st.number_input(
            "边界 log10(G/G0)",
            min_value=-6.9,
            max_value=-1.01,
            value=float(boundary["boundary_log_g_g0"]),
            step=0.01,
            format="%.3f",
            key="boundary_editor",
        )
        if st.button("按新边界重算平台长度", type="secondary"):
            st.session_state.analysis = recalculate_boundary_outputs(
                result,
                float(selected_boundary),
            )
            st.rerun()

    st.markdown('<div class="ditp-section-divider"></div>', unsafe_allow_html=True)
    st.subheader("总体聚类情况")
    st.dataframe(
        _cluster_overview_rows(result).style.format(
            {
                "轨迹占比": "{:.1%}",
                "峰均值": "{:.3f}",
                "峰标准差": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
        height=min(420, 55 + 35 * result.n_clusters),
    )

def _show_cluster_stats(result: AnalysisResult, max_display: int) -> None:
    st.subheader("Cluster 统计")
    cluster_id = st.selectbox(
        "选择要查看的 Cluster",
        options=list(range(result.n_clusters)),
        format_func=lambda value: f"Cluster {value + 1}",
        key="cluster_stats_selector",
        width=340,
    )
    stats = result.cluster_stats[cluster_id]
    color = cluster_color(cluster_id)
    st.markdown(
        f'<div class="ditp-cluster-heading" style="border-left-color:{color};'
        f'background:{_hex_to_rgba(color, 0.10)}">'
        f'Cluster {cluster_id + 1} · {stats["count"]:,} 条轨迹</div>',
        unsafe_allow_html=True,
    )
    conductance_column, stats_column = st.columns([1.65, 1.0])
    with conductance_column:
        st.plotly_chart(_cached_cluster_conductance_figure(result, cluster_id), width="stretch")
    with stats_column:
        st.markdown("#### 电导峰统计")
        stat_cols = st.columns(2)
        stat_cols[0].metric("峰均值", f"{stats['peak_mean']:.3f}")
        stat_cols[1].metric("峰标准差", f"{stats['peak_std']:.3f}")
        stat_cols[0].metric("轨迹占比", f"{stats['fraction']:.1%}")
        stat_cols[1].metric("轨迹数量", f"{stats['count']:,}")
    left, right = st.columns(2)
    with left:
        plotted_count = min(int(max_display), int(stats["count"]))
        st.caption(f"轨迹曲线：实际绘制 {plotted_count:,} / {int(stats['count']):,} 条")
        st.plotly_chart(_cached_cluster_traces_figure(result, cluster_id, max_display), width="stretch")
    with right:
        st.plotly_chart(_cached_cluster_heatmap_figure(result, cluster_id), width="stretch")
    representative = _cached_representative_figure(result, cluster_id)
    representative_heatmap = _cached_representative_heatmap_figure(result, cluster_id)
    if representative is not None and representative_heatmap is not None:
        demo_left, demo_right = st.columns(2)
        with demo_left:
            st.plotly_chart(representative, width="stretch")
        with demo_right:
            st.plotly_chart(representative_heatmap, width="stretch")


def _show_length_stats(result: AnalysisResult) -> None:
    st.subheader("平台长度统计")
    st.dataframe(
        _length_rows(result).style.format({"接受率": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### 各 Cluster 平台长度分布")
    for row_start in range(0, result.n_clusters, 2):
        length_columns = st.columns(2)
        for column_offset, column in enumerate(length_columns):
            cluster_id = row_start + column_offset
            if cluster_id >= result.n_clusters:
                continue
            with column:
                st.plotly_chart(
                    _cached_length_histogram_figure(result, cluster_id),
                    width="stretch",
                )

    st.markdown('<div class="ditp-section-divider"></div>', unsafe_allow_html=True)
    st.subheader("下载结果")
    download_cols = st.columns(2)
    download_cols[0].download_button("下载聚类结果 CSV", _cached_cluster_csv(result), "cluster_assignments.csv", "text/csv")
    download_cols[1].download_button("下载平台长度 CSV", _cached_trace_length_csv(result), "trace_lengths.csv", "text/csv")


def _show_results(result: AnalysisResult) -> None:
    st.sidebar.markdown('<div class="ditp-sidebar-brand">DITP-Analysis</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"当前文件：{st.session_state.get('filename', '未命名')}")
    st.sidebar.caption(
        f"K = {result.n_clusters} · x ROI [{result.roi_x_range[0]:g}, {result.roi_x_range[1]:g}] · "
        f"y ROI [{result.roi_y_range[0]:g}, {result.roi_y_range[1]:g}]"
    )
    st.sidebar.markdown("#### 分析导航")
    section = st.sidebar.radio(
        "选择要查看的内容",
        ["总体聚类概览", "Cluster 统计", "平台长度统计"],
        key="analysis_section",
        label_visibility="collapsed",
    )
    max_display = st.sidebar.slider(
        f"每个 Cluster 最多展示的轨迹数（文件共 {len(result.traces):,} 条）",
        min_value=1,
        max_value=len(result.traces),
        value=min(200, len(result.traces)),
        step=1,
        key="max_display",
    )
    if st.sidebar.button("重新上传数据", width="stretch"):
        st.session_state.pop("analysis", None)
        st.session_state.pop("filename", None)
        st.session_state.pop("analysis_section", None)
        st.session_state.pop("max_display", None)
        st.session_state.pop("boundary_editor", None)
        st.session_state.pop("cluster_stats_selector", None)
        st.rerun()
    if section == "总体聚类概览":
        _show_overview(result, max_display)
    elif section == "Cluster 统计":
        _show_cluster_stats(result, max_display)
    else:
        _show_length_stats(result)


def _show_upload_page() -> None:
    _, center, _ = st.columns([1, 5, 1])
    with center:
        st.markdown(
            """
            <div class="ditp-hero">
                <div class="ditp-eyebrow">SPM TRACE ANALYSIS</div>
                <div class="ditp-hero-title">DITP-Analysis</div>
                <p class="ditp-hero-copy">一次完成平台-噪音边界识别、轨迹聚类和平台长度分析。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("上传数据只在当前会话中处理，不长期保存。")
        with st.form("analysis_form"):
            upload = st.file_uploader(
                "上传无表头 CSV、XLSX、CSV.GZ 或 ZIP 压缩文件（相邻两列为一条轨迹的 x、y），推荐上传 ZIP 压缩文件。",
                type=["csv", "xlsx", "gz", "zip"],
            )
            with st.expander("分析参数", expanded=True):
                k_column, x_column, y_column = st.columns(
                    [0.8, 1.35, 1.55],
                    gap="medium",
                    border=True,
                )
                with k_column:
                    st.markdown("**聚类设置**")
                    n_clusters = st.number_input(
                        "聚类数量 K",
                        min_value=2,
                        max_value=8,
                        value=3,
                        step=1,
                    )
                with x_column:
                    st.markdown("**位移 ROI (nm)**")
                    x_min_column, x_max_column = st.columns(2)
                    x_min = x_min_column.number_input("下界", value=0.0, step=0.1, key="roi_x_min")
                    x_max = x_max_column.number_input("上界", value=2.0, step=0.1, key="roi_x_max")
                with y_column:
                    st.markdown("**电导 ROI log10(G/G0)**")
                    y_min_column, y_max_column = st.columns(2)
                    y_min = y_min_column.number_input("下界", value=-5.0, step=0.1, key="roi_y_min")
                    y_max = y_max_column.number_input("上界", value=-2.0, step=0.1, key="roi_y_max")
            submitted = st.form_submit_button("开始分析", type="primary", width="stretch")

    if submitted:
        if upload is None:
            st.error("请先选择需要分析的数据文件。")
            return
        progress = st.progress(0, text="正在读取上传文件…")
        try:
            data = upload.getvalue()
            progress.progress(15, text="文件读取完成，正在解析轨迹…")
            result = analyze_bytes(
                data,
                upload.name,
                n_clusters=int(n_clusters),
                roi_x_range=(float(x_min), float(x_max)),
                roi_y_range=(float(y_min), float(y_max)),
            )
            progress.progress(85, text="轨迹解析完成，正在整理分析结果…")
            st.session_state.analysis = result
            st.session_state.filename = upload.name
            st.session_state.pop("analysis_section", None)
            st.session_state.pop("max_display", None)
            progress.progress(100, text="分析完成")
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"分析失败：{exc}")
            st.session_state.pop("analysis", None)



def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (
        int(value[offset : offset + 2], 16) for offset in (0, 2, 4)
    )
    return f"rgba({red},{green},{blue},{alpha})"


def main() -> None:
    result = st.session_state.get("analysis")
    if result is None:
        _show_upload_page()
        return

    _show_results(result)


if __name__ == "__main__":
    main()
