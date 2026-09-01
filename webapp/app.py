"""DITP-Analysis Streamlit application."""

from __future__ import annotations

import io
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_service import AnalysisResult, analyze_bytes, calculate_lengths
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
)


st.set_page_config(page_title="DITP-Analysis", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .ditp-section-divider {
        border-top: 1px solid #CBD5E1;
        margin: 2.1rem 0 1.25rem 0;
    }
    .ditp-cluster-heading {
        border-left: 5px solid;
        border-radius: 4px;
        padding: 0.65rem 0.85rem;
        margin: 1.25rem 0 0.35rem 0;
        font-weight: 700;
        color: #1E293B;
    }
    .ditp-cluster-1 { background: #FFF4E5; border-left-color: #B45309; }
    .ditp-cluster-2 { background: #E8F5F2; border-left-color: #0F766E; }
    .ditp-cluster-3 { background: #EAF1FF; border-left-color: #2563EB; }
    [data-testid="stSidebar"] {
        background: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] h3 {
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 0.45rem 0.55rem;
        transition: background 120ms ease, border-color 120ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #E2E8F0;
        border-color: #CBD5E1;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #E0F2F1;
        border-color: #5EEAD4;
        color: #115E59;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _length_rows(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for cluster_id in range(3):
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
            rows.append({"trace_id": item["trace_id"], "cluster_id": cluster_id + 1, **item})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


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
    st.markdown('<div class="ditp-section-divider"></div>', unsafe_allow_html=True)
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
        st.caption("自动边界来自背景峰与平台峰之间的平台侧低谷。修改后只重算平台长度。")
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
            lengths, rejected = calculate_lengths(result.traces, result.labels, float(selected_boundary))
            updated_boundary = {**boundary, "boundary_log_g_g0": float(selected_boundary), "manual_override": 1.0}
            st.session_state.analysis = replace(result, boundary=updated_boundary, lengths=lengths, rejection_counts=rejected)
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
        height=145,
    )

def _show_cluster_stats(result: AnalysisResult, max_display: int) -> None:
    st.markdown('<div class="ditp-section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Cluster 统计")
    cluster_id = st.selectbox(
        "选择 Cluster",
        options=list(range(3)),
        format_func=lambda value: f"Cluster {value + 1}",
        key="cluster_stats_selector",
    )
    stats = result.cluster_stats[cluster_id]
    st.markdown(
        f'<div class="ditp-cluster-heading ditp-cluster-{cluster_id + 1}">'
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
    st.markdown('<div class="ditp-section-divider"></div>', unsafe_allow_html=True)
    st.subheader("平台长度统计")
    st.dataframe(
        _length_rows(result).style.format({"接受率": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### 各 Cluster 平台长度分布")
    for cluster_id in range(3):
        st.plotly_chart(_cached_length_histogram_figure(result, cluster_id), width="stretch")

    st.markdown('<div class="ditp-section-divider"></div>', unsafe_allow_html=True)
    st.subheader("下载结果")
    download_cols = st.columns(2)
    download_cols[0].download_button("下载聚类结果 CSV", _cached_cluster_csv(result), "cluster_assignments.csv", "text/csv")
    download_cols[1].download_button("下载平台长度 CSV", _cached_trace_length_csv(result), "trace_lengths.csv", "text/csv")


def _show_results(result: AnalysisResult) -> None:
    st.sidebar.markdown("### DITP-Analysis")
    st.sidebar.caption(f"当前文件：{st.session_state.get('filename', '未命名')}")
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
    if st.sidebar.button("重新上传数据", use_container_width=True):
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
    st.title("DITP-Analysis")
    st.write("SPM 轨迹聚类与平台长度分析")
    st.info("数据仅在本次应用会话中处理，请勿上传不具备公开处理权限的敏感数据。")
    upload = st.file_uploader("上传无表头 CSV、Excel 或压缩文件（相邻两列为一条轨迹的 x、y）", type=["csv", "xlsx", "xls", "gz", "zip"])
    if upload is not None and st.button("开始分析", type="primary"):
        with st.spinner("正在解析、识别 AllTrace 边界并完成聚类和平台长度计算…"):
            try:
                st.session_state.analysis = analyze_bytes(upload.getvalue(), upload.name)
                st.session_state.filename = upload.name
                st.session_state.pop("analysis_section", None)
                st.session_state.pop("max_display", None)
                st.rerun()
            except Exception as exc:
                st.error(f"分析失败：{exc}")
                st.session_state.pop("analysis", None)

    st.markdown("上传数据后点击“开始分析”。")


def main() -> None:
    result = st.session_state.get("analysis")
    if result is None:
        _show_upload_page()
        return

    _show_results(result)


if __name__ == "__main__":
    main()
