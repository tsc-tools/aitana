from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from aitana import whakaari

STYLE = "ggplot"
STYLE_OVERRIDES = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#404040",
    "xtick.color": "#404040",
    "ytick.color": "#404040",
}


def set_style():
    plt.style.use(STYLE)
    plt.rcParams.update(STYLE_OVERRIDES)


def trellis_plot(
    models: dict,
    data: pd.DataFrame,
    plot_uncertainty: bool = False,
    groups=["b", "c", "d", "e"],
    q_min: float = 0.15,
    q_max: float = 0.85,
):
    """Create a trellis plot for the given models and data.

    Parameters
    ----------
    models : dict
        A dictionary containing the models to plot. Each key should be a model name,
        and each value should be a dictionary with keys 'model', 'color', and optionally
        'colorscale' for ensemble models. The 'model' key should contain a xarray DataArray.
    data : pd.DataFrame
        A pandas DataFrame containing the data to plot.
        It should have a datetime index and a 'group' column indicating the group (b, c, d, e)
        for each time point.
    plot_uncertainty : bool, optional
        Whether to plot uncertainty for the models. Default is False.
    q_min : float, optional
        The minimum quantile to use for plotting uncertainty. Default is 0.15.
    q_max : float, optional
        The maximum quantile to use for plotting uncertainty. Default is 0.85.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib figure object containing the trellis plot.
    """
    # Per-row annotations: (text, x_date, y, ha, arrow_to_x)
    # arrow_to_x is a second x date if an arrow points elsewhere, else None
    row_annotations = {
        "b": [
            ("Dome extrusion", "2012-11-24", 1.05, "center", None),
            ("Geysering", "2013-02-15", 1.05, "center", None),
            (
                "Minor steam and mud eruptions",
                "2013-10-04",
                1.05,
                "right",
                "2013-08-17",
            ),
        ],
        "c": [
            ("Banded tremor", "2015-10-13", 1.05, "center", None),
        ],
        "d": [
            ("Non-explosive ash venting", "2016-09-13", 1.05, "center", None),
            ("Earthquake swarm", "2019-04-15", 1.05, "center", None),
            ("Minor ash emissions", "2019-12-31", 1.05, "center", None),
        ],
        "e": [
            ("Lava extrusion", "2020-01-15", 1.05, "center", None),
            ("Minor ash emissions", "2020-11-13", 1.05, "center", None),
            ("Small steam explosions", "2020-12-29", 1.05, "left", None),
            ("Minor ash emissions", "2022-09-18", 1.05, "center", None),
            ("Small steam explosion", "2024-05-24", 1.05, "center", None),
            ("Minor ash emissions", "2024-07-24", 1.05, "left", None),
        ],
    }

    # Per-row vrect spans: list of (x0, x1)
    row_vrects = {
        "b": [
            ("2012-11-22", "2012-12-10"),
            ("2013-01-15", "2013-04-10"),
            ("2013-08-15", "2013-08-18"),
            ("2013-10-01", "2013-10-08"),
        ],
        "c": [
            ("2015-10-13", "2015-10-20"),
        ],
        "d": [
            ("2016-09-13", "2016-09-18"),
            ("2019-04-23", "2019-07-01"),
            ("2019-12-23", "2019-12-29"),
        ],
        "e": [
            ("2020-01-10", "2020-01-20"),
            ("2020-11-13", "2020-12-01"),
            ("2020-12-29", "2021-01-02"),
            ("2022-09-18", "2022-09-24"),
            ("2024-05-24", "2024-05-31"),
            ("2024-07-24", "2024-09-10"),
        ],
    }

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    highlight_color = prop_colors[6 % len(prop_colors)]

    # Auto-assign colors from prop cycle if not provided
    _color_idx = 0
    for name, model in models.items():
        if name not in ("min", "max", "ensemble") and "color" not in model:
            model["color"] = prop_colors[_color_idx % len(prop_colors)]
            _color_idx += 1

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(14, 12), sharex=False, constrained_layout=True
    )

    legend_handles = {}

    for irow, group_name in enumerate(groups):
        ax = axes[irow]
        ax_twin = ax.twinx()
        ax_twin.set_ylim(0, 1)
        ax_twin.set_yticks([])
        ax_twin.set_yticklabels([])

        # Derive the start/end of this group from data's index directly,
        # then mask each model by its own datetime falling within that range.
        group_times = data.index[data.group == group_name]
        t_start, t_end = group_times[0], group_times[-1]

        def model_mask(da):
            """Return a boolean mask over da's datetime dimension for this group."""
            model_times = pd.to_datetime(da["datetime"].values)
            # Normalise timezone: strip tz from model_times if data index is tz-naive,
            # or localise to UTC if data index is tz-aware.
            if t_start.tzinfo is None:
                model_times = model_times.tz_localize(None)
            else:
                model_times = (
                    model_times.tz_localize("UTC")
                    if model_times.tzinfo is None
                    else model_times
                )
            return (model_times >= t_start) & (model_times <= t_end)

        # Keep a reference time axis for eruption markers (from any plotted model)
        time = None

        for name, model in models.items():
            if name in ["min", "max", "ensemble"]:
                continue
            msk = model_mask(model["model"])
            time = pd.to_datetime(model["model"]["datetime"].values)[msk]
            probs = model["model"].values[msk]
            linestyle = {"solid": "-", "dash": "--", "dot": ":"}.get(
                model.get("dash", "solid"), "-"
            )
            (line,) = ax.plot(
                time, probs, color=model["color"], linestyle=linestyle, label=name
            )
            if name not in legend_handles:
                legend_handles[name] = line

        # Uncertainty
        if plot_uncertainty == "quantile" and "min" in models and "max" in models:
            msk = model_mask(models["min"]["model"])
            time = pd.to_datetime(models["min"]["model"]["datetime"].values)[msk]
            probs_min = models["min"]["model"].values[msk]
            probs_max = models["max"]["model"].values[msk]
            ax.fill_between(
                time,
                probs_min,
                probs_max,
                color=models["min"]["color"],
                alpha=0.3,
            )
        elif plot_uncertainty == "ensemble" and "ensemble" in models:
            ens_model = models["ensemble"]["model"]
            colorscale = models["ensemble"]["colorscale"]
            scores = ens_model.model_score.values
            cmap = plt.get_cmap(
                colorscale if isinstance(colorscale, str) else "viridis"
            )
            norm = mcolors.Normalize(vmin=float(scores.min()), vmax=float(scores.max()))
            msk = model_mask(ens_model)
            time = pd.to_datetime(ens_model["datetime"].values)[msk]
            for j in range(len(scores)):
                model_to_plot = ens_model.isel(model_score=j).values[msk]
                color = cmap(norm(float(scores[j])))
                ax.plot(time, model_to_plot, color=color, linewidth=0.1, alpha=1.0)

        # Eruption markers on twin axis
        t1 = pd.Timestamp(time[0], tz="UTC")
        t2 = pd.Timestamp(time[-1], tz="UTC")
        dfe = whakaari.eruptions(end_date=data.index[-1]).loc[t1:t2]
        for i, erupt_time in enumerate(dfe.index):
            label = (
                "Explosive Eruption"
                if "Explosive Eruption" not in legend_handles
                else None
            )
            vline = ax_twin.axvline(
                erupt_time, color="black", linewidth=0.8, label=label
            )
            if label:
                legend_handles["Explosive Eruption"] = vline

        # Highlighted spans
        for x0_str, x1_str in row_vrects.get(group_name, []):
            ax.axvspan(
                pd.Timestamp(x0_str),
                pd.Timestamp(x1_str),
                color=highlight_color,
                alpha=0.2,
                zorder=0,
            )

        # Annotations
        for annot in row_annotations.get(group_name, []):
            text, x_str, y_frac, ha, arrow_x_str = annot
            x_dt = pd.Timestamp(x_str)
            ax.annotate(
                text,
                xy=(pd.Timestamp(arrow_x_str) if arrow_x_str else x_dt, 1.0),
                xytext=(x_dt, y_frac),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                ha=ha,
                fontsize=16,
                arrowprops=dict(arrowstyle="->", color="black")
                if arrow_x_str
                else None,
            )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Shared legend above all subplots
    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=20,
        bbox_to_anchor=(0.5, -0.08),
    )

    return fig


def scoring_plot(
    forecast: pd.Series,
    threshold: float,
    scoring_function: Callable,
    debug: bool = False,
    ax=None,
):
    """
    Plot the forecast probabilities and the evaluation windows.

    Arguments:
    ----------
        forecast: pandas.Series
            The forecast probabilities.
        threshold: float
            The threshold value.
        scoring_function: Callable
            Function that returns (stats, time_windows) given (threshold, trace).
        debug: bool, optional
            Whether to print debug information.
        ax: matplotlib Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
    """
    try:
        forecast.index = pd.to_datetime(forecast.index).tz_localize("UTC")
    except TypeError:
        pass
    time = forecast.index
    eruptions = whakaari.eruptions(2, "0D", end_date=time[-1]).loc[time[0] : time[-1]]
    trace = pd.DataFrame(
        {
            "prob": (forecast - forecast.min()) / (forecast.max() - forecast.min()),
            "eruptions": eruptions["Activity_Scale"].reindex(time, fill_value=0),
        },
        index=time,
    )
    stats_, time_windows = scoring_function(threshold, trace)
    if debug:
        print(stats_)
    if ax is None:
        _, ax = plt.subplots()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n = len(colors)
    cl_ = dict(
        true_positive=colors[5 % n],
        true_negative=colors[1 % n],
        false_positive=colors[4 % n],
        false_negative=colors[0 % n],
    )

    _eruption_label_shown = False
    for t in eruptions.index:
        ax.axvline(
            t,
            ymin=0.0,
            ymax=0.7,
            linewidth=0.7,
            color="black",
            label="Observed Eruption" if not _eruption_label_shown else "_nolegend_",
        )
        _eruption_label_shown = True

    for window in time_windows:
        start = window["start"]
        end = window["end"]
        type_ = window["type"]

        mask = (time >= start) & (time <= end)
        x_window = time[mask]
        y_window = trace["prob"][mask]

        color = cl_[type_]
        ax.fill_between(
            x_window,
            y_window,
            threshold,
            color=color,
            alpha=0.7,
            linewidth=0,
        )

    legend_handles = [
        mpatches.Patch(color=color, label=type_.replace("_", " ").capitalize())
        for type_, color in cl_.items()
    ]

    ax.set_ylim(0, 1)

    return ax, legend_handles
