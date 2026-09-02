"""Small, stateless analysis service used by the Streamlit application.

The module intentionally keeps only the production conductance workflow:
Hellinger-style histogram features, KMeans++, an AllTrace plateau/noise
boundary, and the fixed-boundary platform length rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

Trace = Tuple[np.ndarray, np.ndarray]


@dataclass
class AnalysisResult:
    """Container for the values needed by the web UI."""

    traces: List[Trace]
    trace_ids: List[int]
    labels: np.ndarray
    features: np.ndarray
    n_clusters: int
    roi_x_range: Tuple[float, float]
    roi_y_range: Tuple[float, float]
    boundary: Dict[str, float]
    cluster_stats: List[Dict[str, Any]]
    lengths: Dict[int, List[Dict[str, Any]]]
    rejection_counts: Dict[int, Dict[str, int]]


def parse_traces(data: bytes, filename: str) -> Tuple[List[Trace], List[int]]:
    """Parse a headerless alternating x/y CSV or XLSX workbook.

    Rows are filtered pairwise so an invalid x value cannot become misaligned
    with a valid y value. The input bytes are not written to disk.
    """
    frame = _read_upload_frame(data, filename)

    traces: List[Trace] = []
    trace_ids: List[int] = []
    for col in range(0, frame.shape[1] - 1, 2):
        pair = frame.iloc[:, [col, col + 1]].apply(pd.to_numeric, errors="coerce")
        values = pair.to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1)
        if not np.any(finite):
            continue
        x = values[finite, 0]
        y = values[finite, 1]
        if x.size >= 2:
            traces.append((x, y))
            trace_ids.append(len(trace_ids) + 1)

    if not traces:
        raise ValueError("文件中没有包含有效 x/y 数据的轨迹。")
    return traces, trace_ids


def _read_upload_frame(data: bytes, filename: str) -> pd.DataFrame:
    """Read a supported upload, unpacking compression in memory only."""
    normalized_name = filename.lower()
    if normalized_name.endswith(".gz"):
        inner_name = normalized_name[:-3]
        if not inner_name.endswith(".csv"):
            raise ValueError("GZIP 压缩文件必须包含 CSV 数据。")
        with gzip.GzipFile(fileobj=BytesIO(data), mode="rb") as stream:
            return pd.read_csv(stream, header=None)

    if normalized_name.endswith(".zip"):
        with ZipFile(BytesIO(data)) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and not name.startswith("__MACOSX/")
                and name.lower().endswith((".csv", ".xlsx"))
            ]
            if len(candidates) != 1:
                raise ValueError("ZIP 压缩包必须且只能包含一个 CSV 或 XLSX 数据文件。")
            member_name = candidates[0]
            return _read_upload_frame(archive.read(member_name), member_name)

    if normalized_name.endswith(".csv"):
        return pd.read_csv(BytesIO(data), header=None)
    if normalized_name.endswith(".xlsx"):
        return pd.read_excel(BytesIO(data), header=None)
    raise ValueError("仅支持 CSV、XLSX、CSV.GZ 或 ZIP 文件。")


def _histogram_features(
    traces: Sequence[Trace],
    x_range: Tuple[float, float] = (0.0, 2.0),
    y_range: Tuple[float, float] = (-5.0, -2.0),
    bins: int = 28,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for x, y in traces:
        mask = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= x_range[0])
            & (x <= x_range[1])
            & (y >= y_range[0])
            & (y <= y_range[1])
        )
        hist, _, _ = np.histogram2d(
            x[mask],
            y[mask],
            bins=[bins, bins],
            range=[list(x_range), list(y_range)],
        )
        total = float(hist.sum())
        if total > 0:
            hist = hist / total
        vectors.append(np.sqrt(np.clip(hist, 0.0, None)).ravel())
    return np.asarray(vectors, dtype=float)


def _reorder_labels(
    traces: Sequence[Trace],
    labels: np.ndarray,
    n_clusters: int,
    roi_y_range: Tuple[float, float],
) -> np.ndarray:
    means: List[float] = []
    for cluster_id in range(n_clusters):
        values: List[np.ndarray] = []
        for index in np.flatnonzero(labels == cluster_id):
            y = traces[int(index)][1]
            values.append(y[(y >= roi_y_range[0]) & (y <= roi_y_range[1])])
        merged = np.concatenate([item for item in values if item.size]) if values else np.array([])
        means.append(float(np.mean(merged)) if merged.size else -np.inf)
    order = np.argsort(means)[::-1]
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _aggregate_projection(
    traces: Sequence[Trace],
    cluster_labels: Optional[np.ndarray] = None,
    cluster_id: Optional[int] = None,
    y_edges: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    edges = y_edges if y_edges is not None else np.linspace(-7.0, 1.0, 301)
    projection = np.zeros(edges.size - 1, dtype=float)
    for index, (x, y) in enumerate(traces):
        if cluster_id is not None and (cluster_labels is None or int(cluster_labels[index]) != cluster_id):
            continue
        mask = np.isfinite(x) & np.isfinite(y) & (x >= -0.3) & (x <= 2.0)
        projection += np.histogram(y[mask], bins=edges)[0]
    return edges, projection


def _smooth_projection(values: np.ndarray) -> np.ndarray:
    # Match the seven-bin triangular smoothing used by the validated AllTrace
    # method examples; this gives -4.72 for the current 20250310 dataset.
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, 3, mode="edge"), kernel, mode="valid")


def detect_alltrace_boundary(traces: Sequence[Trace]) -> Dict[str, float]:
    """Detect the low valley between the background and platform peaks.

    The platform peak is selected from the molecular-conductance range and the
    background peak from its lower-conductance side. A smoothed projection is
    used for robust peak/valley detection; the returned value is the histogram
    bin center and is subsequently used as a fixed boundary for every trace.
    """
    edges, raw_projection = _aggregate_projection(traces)
    centers = (edges[:-1] + edges[1:]) / 2.0
    projection = _smooth_projection(raw_projection)
    peaks, _ = find_peaks(projection)

    platform_candidates = peaks[(centers[peaks] >= -4.5) & (centers[peaks] <= -2.0)]
    if platform_candidates.size:
        platform_index = int(platform_candidates[np.argmax(projection[platform_candidates])])
    else:
        platform_index = int(np.argmax(np.where((centers >= -4.5) & (centers <= -2.0), projection, -1)))

    background_candidates = peaks[centers[peaks] <= centers[platform_index] - 0.8]
    if background_candidates.size:
        background_index = int(background_candidates[np.argmax(projection[background_candidates])])
    else:
        background_index = int(np.argmax(np.where(centers < centers[platform_index], projection, -1)))
    if background_index >= platform_index:
        raise ValueError("无法在 AllTrace 电导分布中找到平台峰左侧的背景峰和低谷。")

    valley_index = background_index + int(
        np.argmin(projection[background_index : platform_index + 1])
    )
    return {
        "boundary_log_g_g0": float(centers[valley_index]),
        "background_peak_log_g_g0": float(centers[background_index]),
        "platform_peak_log_g_g0": float(centers[platform_index]),
        "background_peak_count": float(projection[background_index]),
        "platform_peak_count": float(projection[platform_index]),
        "boundary_count": float(projection[valley_index]),
        "projection_bin_count": float(centers.size),
    }


def _fit_peak(
    y_centers: np.ndarray,
    counts: np.ndarray,
    fit_range: Tuple[float, float],
) -> Tuple[float, float, float]:
    fit_low, fit_high = sorted((float(fit_range[0]), float(fit_range[1])))
    mask = (y_centers >= fit_low) & (y_centers <= fit_high)
    x = y_centers[mask]
    z = counts[mask]
    if x.size < 3 or float(np.max(z)) <= 0:
        return float(x[np.argmax(z)]) if x.size else np.nan, np.nan, np.nan

    def gaussian(value: np.ndarray, mean: float, sigma: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(-((value - mean) ** 2) / (2.0 * sigma**2))

    try:
        initial_mean = float(x[np.argmax(z)])
        params, _ = curve_fit(
            gaussian,
            x,
            z,
            p0=[initial_mean, 0.3, float(np.max(z))],
            bounds=(
                [fit_low, 0.01, 0.0],
                [fit_high, max(2.0, fit_high - fit_low), np.inf],
            ),
            maxfev=5000,
        )
        return float(params[0]), float(abs(params[1])), float(params[2])
    except Exception:
        weights = np.clip(z, 0.0, None)
        mean = float(np.average(x, weights=weights))
        std = float(np.sqrt(np.average((x - mean) ** 2, weights=weights)))
        return mean, std, float(np.max(z))


def _cluster_stats(
    traces: Sequence[Trace],
    labels: np.ndarray,
    cluster_id: int,
    roi_y_range: Tuple[float, float],
    fit_range: Tuple[float, float],
) -> Dict[str, Any]:
    selected = [traces[i] for i in np.flatnonzero(labels == cluster_id)]
    _, projection = _aggregate_projection(selected)
    centers = np.linspace(-7.0, 1.0, 301)
    centers = (centers[:-1] + centers[1:]) / 2.0
    peak, sigma, amplitude = _fit_peak(centers, projection, fit_range)
    roi_values = np.concatenate(
        [
            y[(y >= roi_y_range[0]) & (y <= roi_y_range[1])]
            for _, y in selected
            if np.any((y >= roi_y_range[0]) & (y <= roi_y_range[1]))
        ]
    ) if selected else np.array([])
    return {
        "cluster_id": cluster_id + 1,
        "count": int(len(selected)),
        "fraction": float(len(selected) / len(traces)),
        "peak_mean": peak,
        "peak_std": sigma,
        "peak_amplitude": amplitude,
        "roi_mean": float(np.mean(roi_values)) if roi_values.size else np.nan,
        "roi_std": float(np.std(roi_values)) if roi_values.size else np.nan,
    }


def calculate_lengths(
    traces: Sequence[Trace],
    labels: np.ndarray,
    boundary_log_g_g0: float,
    start_threshold_g0: float = 0.1,
    trim_points_each_side: int = 10,
    max_abs_slope: float = 8.0,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, int]]]:
    """Calculate rightmost-window platform lengths for all traces."""
    if not np.isfinite(boundary_log_g_g0) or boundary_log_g_g0 >= np.log10(start_threshold_g0):
        raise ValueError("平台-噪音边界必须低于起始阈值。")
    start_log = float(np.log10(start_threshold_g0))
    n_clusters = int(np.max(labels)) + 1 if len(labels) else 0
    accepted: Dict[int, List[Dict[str, Any]]] = {
        cluster_id: [] for cluster_id in range(n_clusters)
    }
    rejected: Dict[int, Dict[str, int]] = {
        cluster_id: {} for cluster_id in range(n_clusters)
    }

    def reject(cluster_id: int, reason: str) -> None:
        rejected[cluster_id][reason] = rejected[cluster_id].get(reason, 0) + 1

    for index, (raw_x, raw_y) in enumerate(traces):
        cluster_id = int(labels[index])
        order = np.argsort(np.asarray(raw_x), kind="stable")
        x = np.asarray(raw_x, dtype=float)[order]
        y = np.asarray(raw_y, dtype=float)[order]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        crossing = np.flatnonzero((y[:-1] > start_log) & (y[1:] <= start_log)) + 1
        if crossing.size == 0:
            reject(cluster_id, "No start threshold crossing")
            continue
        raw_start = int(crossing[0])
        endpoint = np.flatnonzero(
            (np.arange(y.size) >= raw_start)
            & (y >= boundary_log_g_g0)
            & (y <= start_log)
        )
        if endpoint.size == 0:
            reject(cluster_id, "No point in boundary window")
            continue
        raw_end = int(endpoint[-1])
        start = raw_start + int(trim_points_each_side)
        end = raw_end - int(trim_points_each_side)
        if end <= start:
            reject(cluster_id, "Not enough points after edge trimming")
            continue
        x_window, y_window = x[start : end + 1], y[start : end + 1]
        length = float(x_window[-1] - x_window[0])
        if not np.isfinite(length) or length <= 0:
            reject(cluster_id, "Invalid displacement span")
            continue
        slope, intercept = np.polyfit(x_window, y_window, 1)
        fitted = slope * x_window + intercept
        rmse = float(np.sqrt(np.mean((y_window - fitted) ** 2)))
        if abs(float(slope)) >= max_abs_slope:
            reject(cluster_id, "Plateau slope too steep")
            continue
        accepted[cluster_id].append(
            {
                "trace_index": index,
                "trace_id": index + 1,
                "length_nm": length,
                "raw_start_index": raw_start,
                "raw_end_index": raw_end,
                "x_start": float(x_window[0]),
                "x_end": float(x_window[-1]),
                "raw_x_start": float(x[raw_start]),
                "raw_x_end": float(x[raw_end]),
                "slope": float(slope),
                "intercept": float(intercept),
                "rmse": rmse,
                "n_fit_points": int(x_window.size),
                "boundary_log_g_g0": float(boundary_log_g_g0),
            }
        )
    return accepted, rejected


def analyze_traces(
    traces: List[Trace],
    trace_ids: Optional[List[int]] = None,
    boundary: Optional[float] = None,
    n_clusters: int = 3,
    roi_x_range: Tuple[float, float] = (0.0, 2.0),
    roi_y_range: Tuple[float, float] = (-5.0, -2.0),
) -> AnalysisResult:
    """Run the complete clustering and length workflow in memory."""
    n_clusters = int(n_clusters)
    roi_x_range = (float(roi_x_range[0]), float(roi_x_range[1]))
    roi_y_range = (float(roi_y_range[0]), float(roi_y_range[1]))
    if n_clusters < 2:
        raise ValueError("聚类数量 K 必须至少为 2。")
    if len(traces) < n_clusters:
        raise ValueError(f"轨迹数量必须不少于聚类数量 K={n_clusters}。")
    if roi_x_range[0] >= roi_x_range[1]:
        raise ValueError("位移 ROI 下界必须小于上界。")
    if roi_y_range[0] >= roi_y_range[1]:
        raise ValueError("电导 ROI 下界必须小于上界。")
    ids = trace_ids or list(range(1, len(traces) + 1))
    features = _histogram_features(
        traces,
        x_range=roi_x_range,
        y_range=roi_y_range,
    )
    if not np.any(np.sum(features, axis=1) > 0):
        raise ValueError(
            "没有轨迹点落在当前聚类 ROI "
            f"（x={roi_x_range[0]:g}..{roi_x_range[1]:g}, "
            f"logG={roi_y_range[0]:g}..{roi_y_range[1]:g}）内。"
        )
    raw_labels = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=42,
    ).fit_predict(features)
    labels = _reorder_labels(traces, raw_labels, n_clusters, roi_y_range)
    boundary_info = detect_alltrace_boundary(traces) if boundary is None else {
        **detect_alltrace_boundary(traces),
        "boundary_log_g_g0": float(boundary),
    }
    lengths, rejected = calculate_lengths(traces, labels, boundary_info["boundary_log_g_g0"])
    fit_range = (float(boundary_info["boundary_log_g_g0"]), float(np.log10(0.1)))
    stats = [
        _cluster_stats(traces, labels, cluster_id, roi_y_range, fit_range)
        for cluster_id in range(n_clusters)
    ]
    return AnalysisResult(
        traces,
        ids,
        labels,
        features,
        n_clusters,
        roi_x_range,
        roi_y_range,
        boundary_info,
        stats,
        lengths,
        rejected,
    )


def analyze_bytes(
    data: bytes,
    filename: str,
    boundary: Optional[float] = None,
    n_clusters: int = 3,
    roi_x_range: Tuple[float, float] = (0.0, 2.0),
    roi_y_range: Tuple[float, float] = (-5.0, -2.0),
) -> AnalysisResult:
    """Parse an upload and run analysis without persisting user data."""
    traces, ids = parse_traces(data, filename)
    return analyze_traces(
        traces,
        ids,
        boundary=boundary,
        n_clusters=n_clusters,
        roi_x_range=roi_x_range,
        roi_y_range=roi_y_range,
    )


def recalculate_boundary_outputs(
    result: AnalysisResult,
    boundary_log_g_g0: float,
) -> AnalysisResult:
    """Recompute boundary-dependent peak fits and lengths without clustering."""
    boundary = float(boundary_log_g_g0)
    lengths, rejected = calculate_lengths(result.traces, result.labels, boundary)
    fit_range = (boundary, float(np.log10(0.1)))
    stats = [
        _cluster_stats(
            result.traces,
            result.labels,
            cluster_id,
            result.roi_y_range,
            fit_range,
        )
        for cluster_id in range(result.n_clusters)
    ]
    return AnalysisResult(
        result.traces,
        result.trace_ids,
        result.labels,
        result.features,
        result.n_clusters,
        result.roi_x_range,
        result.roi_y_range,
        {**result.boundary, "boundary_log_g_g0": boundary, "manual_override": 1.0},
        stats,
        lengths,
        rejected,
    )
