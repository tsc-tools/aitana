import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server use

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from aitana.access.seismic import FDSNWaveforms, SeismicData, get_instrument_response
from obspy import Stream, UTCDateTime
from pyproj import Geod

seismic_data = SeismicData(
    [
        FDSNWaveforms(url="https://service.geonet.org.nz", fill_value="interpolate"),
        FDSNWaveforms(
            url="https://service-nrt.geonet.org.nz", fill_value="interpolate"
        ),
    ]
)

seismic = [
    ("KRVZ", "10", "EHZ"),
    ("OTVZ", "10", "HHZ"),
    ("WTVZ", "10", "EHZ"),
    ("NGZ", "10", "EHZ"),
    ("COVZ", "10", "HHZ"),
    ("TUVZ", "10", "EHZ"),
    ("FWVZ", "10", "HHZ"),
    ("MAVZ", "10", "HHZ"),
    ("WHVZ", "10", "HHZ"),
    ("TRVZ", "10", "HHZ"),
    ("WNVZ", "11", "EHZ"),
    ("MOVZ", "10", "EHZ"),
    ("MTVZ", "10", "EHZ"),
    ("NTVZ", "10", "HHZ"),
]

acoustic = [
    ("COVZ", "30", "HDF"),
    ("IVVZ", "30", "HDF"),
    ("FWVZ", "30", "HDF"),
    ("WHVZ", "30", "HDF"),
    ("TRVZ", "30", "HDF"),
    ("TOVZ", "30", "HDF"),
    ("MAVZ", "30", "HDF"),
    ("TMVZ", "30", "HDF"),
    ("KRVZ", "30", "HDF"),
    ("OTVZ", "30", "HDF"),
    ("WTVZ", "30", "HDF"),
]

TARGETS = {
    "tongariro": (175.671854359, -39.107850505),
    "ruapehu": (175.564490, -39.281149),
}


def get_data(stations, target, tstart, tend, prefix, new=False):
    fout = os.path.join(
        "/tmp",
        "_".join((prefix, str(target[0]), str(target[1]), str(tstart), str(tend))),
    )
    g = Geod(ellps="WGS84")
    st = Stream()
    if os.path.isfile(fout) and not new:
        with open(fout, "rb") as fh:
            st = pickle.load(fh)
    else:
        for s, loc, cmp in stations:
            print(s, str(loc), cmp, tstart, tend)
            tr = next(
                seismic_data.get_waveforms(
                    net="NZ",
                    site=s,
                    loc=str(loc),
                    comp=cmp,
                    start=tstart,
                    end=tend,
                )
            )
            if len(tr) == 0:
                print(f"No data for {s}.{cmp}")
                continue
            inv = get_instrument_response(net="NZ", site=s, loc=str(loc))
            _s = inv[0][0]
            _, _, d = g.inv(target[0], target[1], _s.longitude, _s.latitude)
            tr.stats.distance = d
            st += tr
        with open(fout, "wb") as fh:
            pickle.dump(st, fh)
    return st


def record_section(target, tstart, tend, red_vel=0.0, new=False, outfile=None):
    if target not in TARGETS:
        raise ValueError(f"Unknown volcano '{target}'. Known: {list(TARGETS)}")
    if outfile is None:
        outfile = f"/tmp/record_section_{target}_{tstart}.png"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    st_seismic = get_data(seismic, TARGETS[target], tstart, tend, "seismic", new=new)
    st_acoustic = get_data(acoustic, TARGETS[target], tstart, tend, "acoustic", new=new)

    for tr in st_seismic:
        data = tr.data[:].astype(float)
        data -= data.mean()
        data /= data.max()
        try:
            id0 = int(tr.stats.distance / (red_vel * tr.stats.delta))
        except ZeroDivisionError:
            id0 = 0
        data = data[id0:]
        times = np.arange(data.size) * tr.stats.delta
        ax.plot(times, data - tr.stats.distance / 1e3, "b")

    for tr in st_acoustic:
        data = tr.data[:].astype(float)
        data -= data.mean()
        data /= data.max()
        try:
            id0 = int(tr.stats.distance / (red_vel * tr.stats.delta))
        except ZeroDivisionError:
            id0 = 0
        data = data[id0:]
        times = np.arange(data.size) * tr.stats.delta
        ax.plot(times, data - tr.stats.distance / 1e3, "r", alpha=0.6)

    yticks = ax.get_yticks()
    ax.set_yticklabels([str(abs(y)) for y in yticks])
    ax.set_ylabel("Distance [km]")
    xticks = ax.get_xticks()
    datetimes = [(tstart + x).strftime("%H:%M:%S") for x in xticks]
    ax.set_xticklabels(datetimes, rotation=45, horizontalalignment="right")
    ax.set_title(
        f"{target.capitalize()}  {tstart.strftime('%Y-%m-%d %H:%M')} – "
        f"{tend.strftime('%H:%M')} UTC"
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=100)
    plt.close(fig)
    return outfile


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a record section, mirroring the /moveout Slack command."
    )
    parser.add_argument(
        "volcano",
        choices=sorted(TARGETS),
        help="Target volcano.",
    )
    parser.add_argument(
        "start_time",
        nargs="?",
        default=None,
        help="Start time, e.g. 2024-01-15T11:30 (default: now minus duration).",
    )
    parser.add_argument(
        "duration_min",
        nargs="?",
        type=float,
        default=20.0,
        help="Duration in minutes (default: 20).",
    )
    parser.add_argument(
        "--red-vel",
        type=float,
        default=0.0,
        help="Reduction velocity (default: 0, i.e. no reduction).",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Bypass the on-disk cache and re-fetch waveform data.",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output image path (default: /tmp/record_section_<volcano>_<tstart>.png).",
    )
    args = parser.parse_args()

    if args.start_time is None:
        tstart = UTCDateTime() - args.duration_min * 60
    else:
        tstart = UTCDateTime(args.start_time)
    tend = tstart + args.duration_min * 60

    out = record_section(
        args.volcano,
        tstart,
        tend,
        red_vel=args.red_vel,
        new=args.new,
        outfile=args.outfile,
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
