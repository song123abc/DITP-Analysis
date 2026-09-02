"""Plotly figures for the DITP-Analysis web UI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.analysis_service import AnalysisResult, Trace


COLORS = [
    "#B45309",
    "#0F766E",
    "#2563EB",
    "#9F3A5D",
    "#6D5AA8",
    "#2D7D46",
    "#B0413E",
    "#476582",
]
HEATMAP_COLORS = [
    [0.00, "#FFFFFF"],
    [0.06, "#E8F2F1"],
    [0.22, "#9CCBC5"],
    [0.48, "#287F78"],
    [0.72, "#F0B84D"],
    [1.00, "#B4233C"],
]


def cluster_color(cluster_id: int) -> str:
    return COLORS[cluster_id % len(COLORS)]


def alltrace_figure(result: AnalysisResult) -> go.Figure:
    edges = np.linspace(-7.0, 1.0, 301)
    _, projection = _projection(result.traces)
    centers = (edges[:-1] + edges[1:]) / 2
    boundary = result.boundary
    smoothed = _smooth_counts(projection)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centers, y=smoothed, mode="lines", name="AllTrace", line={"color": "#334155", "width": 2}, fill="tozeroy", fillcolor="rgba(51,65,85,0.12)"))
    fig.add_vline(x=boundary["platform_peak_log_g_g0"], line_dash="dot", line_color="#2563EB", annotation_text="平台峰")
    fig.add_vline(x=boundary["background_peak_log_g_g0"], line_dash="dot", line_color="#64748B", annotation_text="背景峰")
    fig.add_vline(x=boundary["boundary_log_g_g0"], line_dash="dash", line_color="#DC2626", annotation_text=f"边界 {boundary['boundary_log_g_g0']:.3f}")
    fig.update_layout(title="AllTrace 一维电导分布", xaxis_title="log10(G/G0)", yaxis_title="计数", height=500, template="plotly_white")
    return fig


def alltrace_traces_figure(result: AnalysisResult, max_traces: int = 500) -> go.Figure:
    """Plot a bounded, evenly sampled overview of all uploaded traces."""
    indices = np.arange(len(result.traces))
    if indices.size > max_traces:
        indices = indices[np.linspace(0, indices.size - 1, max_traces).astype(int)]
    fig = go.Figure()
    for index in indices:
        x, y = result.traces[int(index)]
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line={"color": "#64748B", "width": 0.65}, opacity=0.10, showlegend=False))
    fig.add_hline(y=np.log10(0.1), line_dash="dot", line_color="#64748B", annotation_text="0.1 G0")
    fig.add_hline(y=result.boundary["boundary_log_g_g0"], line_dash="dash", line_color="#DC2626", annotation_text="平台-噪音边界")
    fig.update_layout(title=f"AllTrace 总体轨迹（展示 {len(indices):,}/{len(result.traces):,}）", xaxis_title="位移 (nm)", yaxis_title="log10(G/G0)", height=470, template="plotly_white")
    return fig


def alltrace_heatmap_figure(result: AnalysisResult) -> go.Figure:
    """Aggregate all traces on the full display range using log-count color."""
    x_edges = np.linspace(-0.3, 2.0, 251)
    y_edges = np.linspace(-7.0, 1.0, 301)
    hist = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=float)
    for x, y in result.traces:
        mask = np.isfinite(x) & np.isfinite(y) & (x >= -0.3) & (x <= 2) & (y >= -7) & (y <= 1)
        hist += np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])[0]
    fig = _log_count_heatmap(hist, x_edges, y_edges, "AllTrace 位移-电导二维直方图")
    fig.update_xaxes(range=[-0.3, 2.0])
    fig.update_yaxes(range=[-7.0, 1.0])
    return fig


def cluster_overview_figure(result: AnalysisResult) -> go.Figure:
    """Summarize cluster sizes before the detailed per-cluster sections."""
    labels = [f"Cluster {cluster_id + 1}" for cluster_id in range(result.n_clusters)]
    counts = [int(result.cluster_stats[cluster_id]["count"]) for cluster_id in range(result.n_clusters)]
    fractions = [float(result.cluster_stats[cluster_id]["fraction"]) for cluster_id in range(result.n_clusters)]
    text = [f"{count:,}<br>{fraction:.1%}" for count, fraction in zip(counts, fractions)]
    fig = go.Figure(go.Bar(x=labels, y=counts, text=text, textposition="outside", marker_color=[cluster_color(i) for i in range(result.n_clusters)], width=0.56, hovertemplate="%{x}<br>轨迹数量 %{y:,}<extra></extra>"))
    fig.update_layout(title="总体聚类数量与占比", xaxis_title=None, yaxis_title="轨迹数量", height=360, template="plotly_white", showlegend=False, margin={"t": 55, "r": 20, "b": 45, "l": 55})
    return fig


def cluster_traces_figure(result: AnalysisResult, cluster_id: int, max_traces: int = 100) -> go.Figure:
    fig = go.Figure()
    indices = np.flatnonzero(result.labels == cluster_id)
    if len(indices) > max_traces:
        indices = indices[np.linspace(0, len(indices) - 1, max_traces).astype(int)]
    for index in indices:
        x, y = result.traces[int(index)]
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line={"color": cluster_color(cluster_id), "width": 0.7}, opacity=0.16, showlegend=False))
    fig.add_hline(y=np.log10(0.1), line_dash="dot", line_color="#64748B", annotation_text="0.1 G0")
    fig.update_layout(title=f"Cluster {cluster_id + 1} 轨迹", xaxis_title="位移 (nm)", yaxis_title="log10(G/G0)", height=470, template="plotly_white")
    return fig


def cluster_conductance_figure(result: AnalysisResult, cluster_id: int) -> go.Figure:
    """Plot one-dimensional conductance counts and the fitted cluster peak."""
    indices = np.flatnonzero(result.labels == cluster_id)
    traces = [result.traces[int(index)] for index in indices]
    edges, projection = _projection(traces)
    centers = (edges[:-1] + edges[1:]) / 2.0
    stats = result.cluster_stats[cluster_id]
    mu = float(stats["peak_mean"])
    sigma = float(stats["peak_std"])
    amplitude = float(stats["peak_amplitude"])

    # Use the same seven-bin triangular smoothing as the validated AllTrace
    # display, while keeping the Gaussian fit on the aggregate distribution.
    smoothed = _smooth_counts(projection)
    fig = go.Figure()
    color = cluster_color(cluster_id)
    fig.add_trace(go.Scatter(x=centers, y=smoothed, mode="lines", name="电导计数", line={"color": color, "width": 1.8}, fill="tozeroy", fillcolor=_rgba(color, 0.16)))
    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0 and np.isfinite(amplitude):
        fit_x = np.linspace(
            float(result.boundary["boundary_log_g_g0"]),
            float(np.log10(0.1)),
            300,
        )
        fit_y = amplitude * np.exp(-0.5 * ((fit_x - mu) / sigma) ** 2)
        fig.add_trace(go.Scatter(x=fit_x, y=fit_y, mode="lines", name="高斯拟合", line={"color": "#111827", "width": 2.0}))
    if np.isfinite(mu):
        # Keep the peak marker unobtrusive; the exact fitted value is shown in
        # the statistics panel beside the chart.
        fig.add_vline(x=mu, line_dash="dot", line_color=color)
    fig.update_layout(title=f"Cluster {cluster_id + 1} 一维电导分布与高斯拟合", xaxis_title="log10(G/G0)", yaxis_title="计数", height=350, template="plotly_white", legend={"orientation": "h", "y": 1.08, "x": 0.01})
    return fig


def cluster_heatmap_figure(result: AnalysisResult, cluster_id: int) -> go.Figure:
    # Cluster overview uses the full plotting range. The representative
    # trace heatmap below intentionally remains restricted to the clustering
    # ROI, so the two views answer different questions.
    x_edges = np.linspace(-0.3, 2.0, 251)
    y_edges = np.linspace(-7.0, 1.0, 301)
    hist = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=float)
    for index in np.flatnonzero(result.labels == cluster_id):
        x, y = result.traces[int(index)]
        mask = np.isfinite(x) & np.isfinite(y) & (x >= -0.3) & (x <= 2) & (y >= -7) & (y <= 1)
        hist += np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])[0]
    fig = _log_count_heatmap(hist, x_edges, y_edges, f"Cluster {cluster_id + 1} 位移-电导二维直方图")
    fig.update_xaxes(range=[-0.3, 2.0])
    fig.update_yaxes(range=[-7.0, 1.0])
    return fig


def length_histogram_figure(result: AnalysisResult, cluster_id: int) -> go.Figure:
    values = [item["length_nm"] for item in result.lengths[cluster_id]]
    fig = go.Figure()
    if values:
        values_array = np.asarray(values, dtype=float)
        bin_width = 0.02
        left = np.floor(np.min(values_array) / bin_width) * bin_width
        right = np.ceil(np.max(values_array) / bin_width) * bin_width + bin_width
        edges = np.arange(left, right + bin_width * 0.5, bin_width)
        if edges.size < 2:
            edges = np.array([left, left + bin_width])
        counts, edges = np.histogram(values_array, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        fig.add_trace(go.Bar(x=centers, y=counts, width=bin_width * 0.96, marker_color=cluster_color(cluster_id), opacity=0.84, name="长度计数"))

        # Fit the raw accepted lengths, matching the command-line workflow.
        # The fine histogram is only the display layer and must not determine
        # the fitted parameters.
        mu, sigma = norm.fit(values_array)
        amplitude = float(values_array.size * bin_width / (sigma * np.sqrt(2.0 * np.pi))) if sigma > 0 else np.nan
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            curve_x = np.linspace(float(edges[0]), float(edges[-1]), 300)
            curve_y = amplitude * np.exp(-0.5 * ((curve_x - mu) / sigma) ** 2)
            fig.add_trace(go.Scatter(x=curve_x, y=curve_y, mode="lines", name="高斯拟合", line={"color": "#111827", "width": 2.2}))
            fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper", xanchor="right", yanchor="top", text=f"μ = {mu:.3f} nm<br>σ = {sigma:.3f} nm", showarrow=False, font={"size": 12, "color": "#111827"}, bgcolor="rgba(255,255,255,0.76)")
    else:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="没有有效平台长度", showarrow=False)
    fig.update_layout(title=f"Cluster {cluster_id + 1} 平台长度分布（bin = 0.02 nm）", xaxis_title="平台长度 (nm)", yaxis_title="轨迹数量", height=390, template="plotly_white", bargap=0.04)
    return fig


def representative_figure(result: AnalysisResult, cluster_id: int) -> Optional[go.Figure]:
    candidates = result.lengths[cluster_id]
    if not candidates:
        return None
    item = candidates[len(candidates) // 2]
    x, y = result.traces[item["trace_index"]]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="原始轨迹", line={"color": "#334155", "width": 1.5}))
    fig.add_vline(x=item["x_start"], line_dash="dot", line_color="#059669", annotation_text="起点")
    fig.add_vline(x=item["x_end"], line_dash="dash", line_color="#DC2626", annotation_text="终点")
    fit_x = np.linspace(item["x_start"], item["x_end"], 80)
    fit_y = item["slope"] * fit_x + item["intercept"]
    fig.add_trace(go.Scatter(x=fit_x, y=fit_y, mode="lines", name="OLS 拟合", line={"color": "#7C3AED", "width": 2}))
    fig.add_hline(y=np.log10(0.1), line_dash="dot", line_color="#64748B", annotation_text="0.1 G0")
    fig.add_hline(y=item["boundary_log_g_g0"], line_dash="dash", line_color="#DC2626", annotation_text="平台-噪音边界")
    fig.update_layout(title=f"Cluster {cluster_id + 1} 代表轨迹（ID {item['trace_id']}，L={item['length_nm']:.3f} nm）", xaxis_title="位移 (nm)", yaxis_title="log10(G/G0)", height=470, template="plotly_white")
    return fig


def representative_heatmap_figure(result: AnalysisResult, cluster_id: int) -> Optional[go.Figure]:
    """Show the selected representative trace inside the clustering ROI."""
    candidates = result.lengths[cluster_id]
    if not candidates:
        return None
    item = candidates[len(candidates) // 2]
    x, y = result.traces[item["trace_index"]]
    x_edges = np.linspace(result.roi_x_range[0], result.roi_x_range[1], 29)
    y_edges = np.linspace(result.roi_y_range[0], result.roi_y_range[1], 29)
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= result.roi_x_range[0])
        & (x <= result.roi_x_range[1])
        & (y >= result.roi_y_range[0])
        & (y <= result.roi_y_range[1])
    )
    hist = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])[0]
    total = float(np.sum(hist))
    feature = np.sqrt(hist / total) if total > 0 else hist
    fig = go.Figure(go.Heatmap(
        x=(x_edges[:-1] + x_edges[1:]) / 2,
        y=(y_edges[:-1] + y_edges[1:]) / 2,
        z=feature.T,
        colorscale=HEATMAP_COLORS,
        zmin=0.0,
        zmax=float(np.max(feature)) if np.any(feature) else 1.0,
        colorbar={},
    ))
    fig.update_layout(
        title=f"代表轨迹 ROI 特征图（ID {item['trace_id']}）",
        xaxis_title="位移 (nm)",
        yaxis_title="log10(G/G0)",
        height=470,
        template="plotly_white",
    )
    return fig


def _projection(traces: Sequence[Trace]) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-7.0, 1.0, 301)
    projection = np.zeros(300, dtype=float)
    for x, y in traces:
        mask = np.isfinite(x) & np.isfinite(y) & (x >= -0.3) & (x <= 2.0)
        projection += np.histogram(y[mask], bins=edges)[0]
    return edges, projection


def _log_count_heatmap(hist: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, title: str) -> go.Figure:
    """Create a high-contrast log-count heatmap with robust color clipping."""
    transformed = np.log1p(hist)
    positive = transformed[hist > 0]
    robust_max = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    fig = go.Figure(go.Heatmap(
        x=(x_edges[:-1] + x_edges[1:]) / 2,
        y=(y_edges[:-1] + y_edges[1:]) / 2,
        z=transformed.T,
        customdata=hist.T,
        colorscale=HEATMAP_COLORS,
        zmin=0.0,
        zmax=robust_max,
        colorbar={"title": "log(1+计数)"},
        hovertemplate="位移 %{x:.3f} nm<br>电导 %{y:.3f}<br>计数 %{customdata:.0f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="位移 (nm)", yaxis_title="log10(G/G0)", height=470, template="plotly_white")
    return fig


def _smooth_counts(values: np.ndarray) -> np.ndarray:
    """Smooth count curves with the project's seven-bin triangular kernel."""
    kernel = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    kernel /= np.sum(kernel)
    return np.convolve(np.pad(np.asarray(values, dtype=float), 3, mode="edge"), kernel, mode="valid")


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[offset : offset + 2], 16) for offset in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
