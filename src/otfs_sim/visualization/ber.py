import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator


COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "x", "*", "v", "P"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def results_to_rows(results):
    """Convert result dict-of-lists into list-of-dicts."""
    keys = list(results.keys())
    if not keys:
        return []
    num_rows = len(results[keys[0]])
    return [{key: results[key][i] for key in keys} for i in range(num_rows)]


def filter_rows(rows, filter_by=None):
    """
    Filter rows by exact values.

    Example:
        filter_by={
            "channel_scenario": ["low_doppler", "high_doppler"],
            "receiver_scenario": "perfect_csi_lmmse",
        }
    """
    if filter_by is None:
        return rows
    filtered = []
    for row in rows:
        keep = True
        for key, allowed in filter_by.items():
            if not isinstance(allowed, (list, tuple, set)):
                allowed = [allowed]
            if row.get(key) not in allowed:
                keep = False
                break
        if keep:
            filtered.append(row)
    return filtered


def make_style_map(rows, field, style_values):
    """Assign colors/linestyles/markers to unique values of a field."""
    values = sorted({row.get(field) for row in rows})
    return {v: style_values[i % len(style_values)] for i, v in enumerate(values)}


def make_label(row, label_fields):
    """Build readable legend label from selected fields."""
    parts = [str(row[f]) for f in label_fields if row.get(f) is not None]
    return " | ".join(parts)


def _apply_cutoff(x, y, cutoff):
    """
    Remove points where y < cutoff (i.e. stop showing once curve goes
    below the cutoff threshold).  Returns masked arrays.
    """
    if cutoff is None:
        return x, y
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = y >= cutoff
    return x[keep], y[keep]


def _smooth_curve(x, y, num_points=300):
    """
    Smooth a BER/BLER curve by cubic spline interpolation in log10-space.

    Strategy
    --------
    * The "floor" (y >= 1.0) is kept as a flat horizontal segment.
    * The falling region (y < 1.0) is interpolated with a cubic spline
      in log10-space for smoothness on a semilogy axis.
    * If there are fewer than 2 falling points the raw data is returned
      unchanged so the caller can still draw a line.

    Returns
    -------
    x_line, y_line   : smooth curve to draw as a line (no markers)
    x_raw,  y_raw    : original points to overlay as markers
    can_smooth : bool — False means not enough data, draw raw line instead
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    falling_mask = y < 1.0

    if falling_mask.sum() < 2:
        return x, y, x, y, False   # can_smooth=False

    x_fall = x[falling_mask]
    y_fall = y[falling_mask]

    # Guard against zero / negative values before log
    y_fall = np.clip(y_fall, 1e-12, None)

    # f = interp1d(x_fall, np.log10(y_fall), kind="cubic", fill_value="extrapolate")
    f = PchipInterpolator(x_fall, np.log10(y_fall), extrapolate=True)
    x_smooth = np.linspace(x_fall.min(), x_fall.max(), num_points)
    y_smooth = 10.0 ** f(x_smooth)

    # Prepend the floor segment (all x where y >= 1.0)
    x_floor = x[~falling_mask]
    if len(x_floor):
        x_smooth = np.concatenate([x_floor, x_smooth])
        y_smooth = np.concatenate([np.ones(len(x_floor)), y_smooth])

    return x_smooth, y_smooth, x, y, True


def _plot_one_curve(ax, x, y, color, linestyle, marker, label, interpolate):
    """
    Draw a single BER/BLER curve on *ax*.

    When interpolate=True:
      - smooth spline drawn as a line (no markers)
      - original points overlaid as markers only (no line, no legend entry)
      - falls back gracefully to a plain line if there aren't enough points
    When interpolate=False:
      - standard line + markers as before
    """
    if not interpolate:
        ax.semilogy(x, y, color=color, linestyle=linestyle, marker=marker,
                    linewidth=2, markersize=6, label=label)
        return

    x_line, y_line, x_raw, y_raw, can_smooth = _smooth_curve(x, y)

    if can_smooth:
        # Smooth line — legend entry here
        ax.semilogy(x_line, y_line, color=color, linestyle=linestyle,
                    linewidth=2, label=label)
        # Raw markers — no legend entry (label=None keeps legend clean)
        ax.semilogy(x_raw, y_raw, color=color, linestyle="none",
                    marker=marker, markersize=6, label=None)
    else:
        # Not enough falling points (curve stays at floor) — plain line
        ax.semilogy(x_raw, y_raw, color=color, linestyle=linestyle,
                    marker=marker, linewidth=2, markersize=6, label=label)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_ber_curves(
    results,
    color_by="receiver_scenario",
    linestyle_by="waveform",
    marker_by="channel_scenario",
    label_fields=None,
    filter_by=None,
    title="BER Curves",
    figsize=(8, 5),
    interpolate=False,
    cutoff=None,
):
    """
    Plot many BER curves with automatic styling.

    Parameters
    ----------
    interpolate : bool
        Smooth each curve with cubic spline interpolation in log-space.
        Original data points are still shown as markers.
    cutoff : float or None
        Hide points below this BER value (e.g. cutoff=1e-4 stops the curve
        once it goes below 1e-4).  None shows all points.
    """
    if label_fields is None:
        label_fields = [color_by, linestyle_by, marker_by]

    rows = results_to_rows(results)
    rows = filter_rows(rows, filter_by)

    if not rows:
        raise ValueError("No result rows available after filtering.")

    color_map     = make_style_map(rows, color_by,     COLORS)
    linestyle_map = make_style_map(rows, linestyle_by, LINESTYLES)
    marker_map    = make_style_map(rows, marker_by,    MARKERS)

    fig, ax = plt.subplots(figsize=figsize)

    for row in rows:
        x, y = _apply_cutoff(row["ebno_db"], row["ber"], cutoff)
        if len(x) == 0:
            continue

        _plot_one_curve(
            ax, x, y,
            color     = color_map[row.get(color_by)],
            linestyle = linestyle_map[row.get(linestyle_by)],
            marker    = marker_map[row.get(marker_by)],
            label     = make_label(row, label_fields),
            interpolate=interpolate,
        )

    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_bler_curves(
    results,
    color_by="receiver_scenario",
    linestyle_by="waveform",
    marker_by="channel_scenario",
    label_fields=None,
    filter_by=None,
    title="BLER Curves",
    figsize=(8, 5),
    interpolate=False,
    cutoff=None,
):
    """
    Plot many BLER curves with automatic styling.

    Parameters
    ----------
    interpolate : bool
        Smooth each curve with cubic spline interpolation in log-space.
        Original data points are still shown as markers.
    cutoff : float or None
        Hide points below this BLER value (e.g. cutoff=1e-4 stops the curve
        once it goes below 1e-4).  None shows all points.
    """
    if label_fields is None:
        label_fields = [color_by, linestyle_by, marker_by]

    rows = results_to_rows(results)
    rows = filter_rows(rows, filter_by)

    if not rows:
        raise ValueError("No result rows available after filtering.")

    color_map     = make_style_map(rows, color_by,     COLORS)
    linestyle_map = make_style_map(rows, linestyle_by, LINESTYLES)
    marker_map    = make_style_map(rows, marker_by,    MARKERS)

    fig, ax = plt.subplots(figsize=figsize)

    for row in rows:
        x, y = _apply_cutoff(row["ebno_db"], row["bler"], cutoff)
        if len(x) == 0:
            continue

        _plot_one_curve(
            ax, x, y,
            color     = color_map[row.get(color_by)],
            linestyle = linestyle_map[row.get(linestyle_by)],
            marker    = marker_map[row.get(marker_by)],
            label     = make_label(row, label_fields),
            interpolate=interpolate,
        )

    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("BLER")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()