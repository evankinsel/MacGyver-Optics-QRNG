
# -*- coding: utf-8 -*-
"""
Dual-SPAD Photon-Counting / Exploratory QRNG Experiment

405 nm laser research | Swabian Time Tagger | Spyder

Features:
- Dual-SPAD photon timestamp acquisition
- Short, full, and continuous baseline recording
- Manual stopping with automatic data saving
- Count rates and A:B detector ratios
- Inter-arrival time distributions
- Exploratory QRNG bit extraction and statistics
- Publication-quality vector PDF plots

IMPORTANT:
Research outputs are not certified cryptographic keys.
The script does not control laser current or temperature.
"""

from __future__ import annotations

import json
import math
import secrets
import time
import socket
import os
import shutil

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import fft

try:
    import scienceplots
except ImportError:
    scienceplots = None

try:
    from Swabian import TimeTagger
except ImportError:
    try:
        import TimeTagger
    except ImportError:
        TimeTagger = None


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MODE = "menu"

SERIAL = "1948000SQ6"

CH_A = 2
CH_B = 3

TRIGGER_A_V = 0.5
TRIGGER_B_V = 0.5

DEAD_TIME_NS = 22.0
WARNING_CPS = 100_000

SHORT_A_EVENTS = 100_000
FULL_A_EVENTS = 2_000_000

SHORT_TIMEOUT_S = 60
FULL_TIMEOUT_S = 180

STREAM_BUFFER_EVENTS = 500_000

# Default output folder
DEFAULT_DOWNLOADS = Path.home() / "Downloads"

if DEFAULT_DOWNLOADS.exists():
    SAVE_ROOT = DEFAULT_DOWNLOADS / "qrng_data"
else:
    SAVE_ROOT = Path("qrng_data")

ANALYZE_FOLDER = None

TOEPLITZ_DEMO = True
SHOW_FIGURES = True

TEMP_SETPOINT_C = None
ITH_MA = None
LASER_BIAS_MA = None

OPERATOR = None
SAMPLE_ID = None
RUN_NOTES = None

USER_CONFIG = Path.home() / ".qrng_config.json"


# ============================================================
# MANUSCRIPT PLOTTING
# ============================================================

PLOT_LAYOUT = "single"

FIGURE_SIZES = {
    "single": (4.0, 3.0),
    "double": (7.0, 4.0)
}


def configure_publication_plots():
    """Configure consistent journal-style figures."""

    if scienceplots is not None:
        plt.style.use(["science", "ieee"])
    else:
        plt.style.use("default")
        print(
            "SciencePlots unavailable; using serif fallback."
        )

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.family": "serif",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        }
    )

    # Use Matplotlib mathtext if external LaTeX is unavailable.
    if shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.fontset"] = "stix"


def manuscript_figure(layout=None):
    """Create a figure sized for a manuscript column."""

    if layout is None:
        layout = PLOT_LAYOUT

    if layout not in FIGURE_SIZES:
        raise ValueError(
            "Layout must be 'single' or 'double'."
        )

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZES[layout],
        constrained_layout=True
    )

    return fig, ax


def save_manuscript_figure(fig, folder, filename):
    """Save a vector PDF of the figure."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        folder / filename,
        format="pdf",
        dpi=600
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


def example_simulated_arrival_plot(folder=None):
    """Optional simulated example, never measured data."""

    configure_publication_plots()

    rng = np.random.default_rng(2026)

    samples_ns = rng.exponential(
        scale=250.0,
        size=20000
    )

    fig, ax = manuscript_figure()

    ax.hist(
        samples_ns,
        bins=65,
        density=True,
        histtype="step",
        label="Simulated arrivals"
    )

    x = np.linspace(0, 1800, 300)

    ax.plot(
        x,
        np.exp(-x / 250.0) / 250.0,
        linestyle="--",
        label="Exponential model"
    )

    ax.set(
        xlabel=r"Inter-arrival time $\Delta t$ (ns)",
        ylabel=r"Probability density (ns$^{-1}$)",
        title="Simulated inter-arrival distribution"
    )

    ax.legend(frameon=False)
    sns.despine(ax=ax)

    if folder is None:
        folder = SAVE_ROOT / "plot_style_example"

    save_manuscript_figure(
        fig,
        folder,
        "simulated_interarrival_example.pdf"
    )

    return Path(folder) / "simulated_interarrival_example.pdf"


# ============================================================
# GENERAL UTILITIES
# ============================================================

def stamp():
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            allow_nan=False
        )


def write_bits(path, bits):
    """Write complete MSB-first bytes only."""

    count = len(bits) // 8 * 8

    np.packbits(
        bits[:count],
        bitorder="big"
    ).tofile(path)

    return count


# ============================================================
# TIME TAGGER ACQUISITION
# ============================================================

def drain_stream(stream, buffers, counters):
    """Read photon timestamps and check event integrity."""

    block = stream.getData()

    timestamps = np.asarray(
        block.getTimestamps(),
        dtype=np.int64
    )

    channels = np.asarray(block.getChannels())

    if timestamps.shape != channels.shape:
        raise RuntimeError(
            "Inconsistent timestamp and channel arrays."
        )

    get_types = getattr(block, "getEventTypes", None)
    get_missed = getattr(block, "getMissedEvents", None)

    if not callable(get_types) or not callable(get_missed):
        raise RuntimeError(
            "Time Tagger driver lacks event-integrity information."
        )

    event_types = np.asarray(get_types())
    missed_tags = np.asarray(get_missed())

    if event_types.shape != timestamps.shape:
        raise RuntimeError("Invalid event-type array.")

    if missed_tags.shape != timestamps.shape:
        raise RuntimeError("Invalid missed-event array.")

    if np.any(missed_tags < 0):
        raise RuntimeError("Negative missed-event count.")

    valid = event_types == 0

    for channel in (CH_A, CH_B):

        selected = timestamps[
            valid & (channels == channel)
        ]

        if selected.size:
            buffers[channel].append(selected.copy())
            counters[channel] += int(selected.size)

    return (
        int(missed_tags.sum()),
        int(np.count_nonzero(~valid))
    )


def collect(target=None, timeout=None):
    """
    Acquire both SPAD channels.

    target=None and timeout=None:
        Continuous acquisition until Ctrl+C.

    A normal manual stop saves all accumulated timestamps.
    """

    if CH_A == CH_B:
        raise ValueError("SPAD channels must differ.")

    if target is not None and target < 2:
        raise ValueError("Invalid event target.")

    if timeout is not None and timeout <= 0:
        raise ValueError("Invalid timeout.")

    if TimeTagger is None:
        raise RuntimeError(
            "Swabian Time Tagger driver is not available."
        )

    buffers = {
        CH_A: [],
        CH_B: []
    }

    counters = {
        CH_A: 0,
        CH_B: 0
    }

    missed = 0
    invalid_types = 0
    overflows = None

    status = "complete"
    error = None

    tagger = None
    stream = None

    start = time.monotonic()

    try:

        tagger = TimeTagger.createTimeTagger(SERIAL)

        tagger.setTriggerLevel(
            CH_A,
            TRIGGER_A_V
        )

        tagger.setTriggerLevel(
            CH_B,
            TRIGGER_B_V
        )

        tagger.clearOverflows()

        stream = TimeTagger.TimeTagStream(
            tagger,
            int(STREAM_BUFFER_EVENTS),
            [CH_A, CH_B]
        )

        stream.start()

        start = time.monotonic()
        next_print = start + 5

        while target is None or counters[CH_A] < target:

            elapsed = time.monotonic() - start

            if timeout is not None and elapsed >= timeout:
                status = "timeout"
                error = "Event target not reached before timeout."
                break

            time.sleep(0.05)

            m, problems = drain_stream(
                stream,
                buffers,
                counters
            )

            missed += m
            invalid_types += problems

            if time.monotonic() >= next_print:

                elapsed = max(
                    time.monotonic() - start,
                    0.001
                )

                print(
                    f"Live: "
                    f"A {counters[CH_A]:,} "
                    f"({counters[CH_A] / elapsed:,.1f} cps) | "
                    f"B {counters[CH_B]:,} "
                    f"({counters[CH_B] / elapsed:,.1f} cps)"
                )

                next_print = time.monotonic() + 5

    except KeyboardInterrupt:

        if target is None:
            status = "stopped_by_user"
            error = "Continuous recording stopped normally."
        else:
            status = "interrupted"
            error = "Stopped before event target."

    except Exception as exc:

        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    finally:

        if stream is not None:

            try:

                stream.stop()

                m, problems = drain_stream(
                    stream,
                    buffers,
                    counters
                )

                missed += m
                invalid_types += problems

            except Exception as exc:

                status = "error"
                error = f"Final stream drain failed: {exc}"

        if tagger is not None:

            try:
                overflows = int(tagger.getOverflows())
            except Exception as exc:
                status = "error"
                error = f"Overflow query failed: {exc}"

            try:
                TimeTagger.freeTimeTagger(tagger)
            except Exception as exc:
                status = "error"
                error = f"Device release failed: {exc}"

    raw = {}

    for channel in (CH_A, CH_B):

        if buffers[channel]:
            raw[channel] = np.concatenate(
                buffers[channel]
            )
        else:
            raw[channel] = np.empty(
                0,
                dtype=np.int64
            )

    if (
        status == "complete"
        and target is not None
        and counters[CH_A] < target
    ):
        status = "incomplete"
        error = "Insufficient channel-A events."

    if (
        status in {"complete", "stopped_by_user"}
        and (missed or invalid_types or overflows)
    ):
        status = "data_loss"
        error = "Lost or invalid event tags detected."

    for channel, timestamps in raw.items():

        if (
            len(timestamps) > 1
            and np.any(np.diff(timestamps) <= 0)
        ):
            status = "timestamp_error"
            error = (
                f"Non-increasing timestamps "
                f"on channel {channel}."
            )

    info = {
        "status": status,
        "error": error,
        "valid": status in {
            "complete",
            "stopped_by_user"
        },
        "raw_events": {
            str(c): len(raw[c])
            for c in raw
        },
        "target_A": target,
        "wall_seconds": round(
            time.monotonic() - start,
            3
        ),
        "acquisition_mode": (
            "continuous"
            if target is None
            else "targeted"
        ),
        "missed_events": missed,
        "non_photon_tags": invalid_types,
        "hardware_overflows": overflows,
        "serial": SERIAL,
        "channels": [CH_A, CH_B],
        "trigger_volts": [
            TRIGGER_A_V,
            TRIGGER_B_V
        ],
        "recorded_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "temperature_setpoint_C": TEMP_SETPOINT_C,
        "threshold_current_mA": ITH_MA,
        "laser_bias_current_mA": LASER_BIAS_MA
    }

    return raw, info


# ============================================================
# COMMON EXPOSURE WINDOW
# ============================================================

def common_window(raw, target=None, exposure_seconds=None):
    """
    Compare both channels over a consistent exposure.

    Continuous baseline:
        Use full available recording and wall-clock duration.

    Targeted acquisition:
        Use channel A's first-to-last event window.
    """

    a = np.asarray(
        raw[CH_A],
        dtype=np.int64
    )

    b = np.asarray(
        raw[CH_B],
        dtype=np.int64
    )

    if target is not None:

        a = a[:target]

        if len(a) < 2:
            return (
                a,
                np.empty(0, dtype=np.int64),
                0.0,
                None
            )

        lo = int(a[0])
        hi = int(a[-1])

        b = b[
            (b >= lo) & (b <= hi)
        ]

        return (
            a,
            b,
            (hi - lo) * 1e-12,
            (lo, hi)
        )

    duration = max(
        0.0,
        float(exposure_seconds or 0)
    )

    if len(a) == 0 and len(b) == 0:
        return (
            a,
            b,
            duration,
            (0, int(duration * 1e12))
        )

    starts = [
        int(arr[0])
        for arr in (a, b)
        if len(arr)
    ]

    lo = min(starts)

    return (
        a,
        b,
        duration,
        (lo, lo + int(duration * 1e12))
    )


# ============================================================
# PHOTON MEASUREMENT AND MANUSCRIPT FIGURES
# ============================================================

def characterize(raw, info, folder):

    a, b, seconds, limits = common_window(
        raw,
        info.get("target_A"),
        info.get("wall_seconds")
    )

    na = len(a)
    nb = len(b)

    if seconds <= 0:

        report = {
            "problem": "No positive exposure window",
            "A_count": na,
            "B_count": nb
        }

        save_json(
            folder / "measurements.json",
            report
        )

        return report

    total = na + nb

    balance = {
        "exposure_seconds": seconds,
        "common_window_ps": list(limits),
        "A_count": na,
        "B_count": nb,
        "A_cps": na / seconds,
        "B_cps": nb / seconds,
        "A_over_B": na / nb if nb else None,
        "A_percent": (
            100 * na / total
            if total else None
        ),
        "B_percent": (
            100 * nb / total
            if total else None
        ),
        "normalized_imbalance": (
            (na - nb) / total
            if total else None
        ),
        "A_above_warning_cps": (
            na / seconds > WARNING_CPS
        ),
        "B_above_warning_cps": (
            nb / seconds > WARNING_CPS
        ),
        "dead_time_violations": {}
    }

    for label, timestamps in (
        ("A", a),
        ("B", b)
    ):

        intervals = np.diff(timestamps)

        balance["dead_time_violations"][label] = (
            int(
                np.count_nonzero(
                    intervals < round(DEAD_TIME_NS * 1000)
                )
            )
            if intervals.size else None
        )

    save_json(
        folder / "measurements.json",
        balance
    )

    # --------------------------------------------------------
    # FIGURE 1: PHOTON COUNT-RATE STABILITY
    # --------------------------------------------------------

    edges = np.linspace(
        0.0,
        seconds,
        max(
            2,
            min(
                2001,
                int(np.ceil(seconds)) + 1
            )
        )
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    bin_widths = np.diff(edges)

    times_a = (
        a - limits[0]
    ) * 1e-12

    times_b = (
        b - limits[0]
    ) * 1e-12

    hist_a = (
        np.histogram(times_a, bins=edges)[0]
        / bin_widths
    )

    hist_b = (
        np.histogram(times_b, bins=edges)[0]
        / bin_widths
    )

    fig, ax = manuscript_figure()

    ax.plot(
        centers,
        hist_a,
        label=f"SPAD A (ch{CH_A})"
    )

    ax.plot(
        centers,
        hist_b,
        label=f"SPAD B (ch{CH_B})"
    )

    ax.set(
        xlabel="Elapsed time (s)",
        ylabel=r"Count rate (s$^{-1}$)",
        title="Photon count-rate stability"
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    sns.despine(ax=ax)

    save_manuscript_figure(
        fig,
        folder,
        "count_rates.pdf"
    )

    # --------------------------------------------------------
    # FIGURE 2: INTER-ARRIVAL DISTRIBUTION
    # --------------------------------------------------------

    fig, ax = manuscript_figure()

    plotted = False

    for label, timestamps in (
        ("A", a),
        ("B", b)
    ):

        if len(timestamps) <= 2:
            continue

        dt = np.diff(timestamps) * 1e-3

        if not (
            np.max(dt) > np.min(dt) > 0
        ):
            continue

        edges2 = np.geomspace(
            float(np.min(dt)),
            float(np.max(dt)) * 1.001,
            85
        )

        density, _ = np.histogram(
            dt,
            bins=edges2,
            density=True
        )

        bin_centers = np.sqrt(
            edges2[:-1] * edges2[1:]
        )

        ax.plot(
            bin_centers,
            density,
            label=f"SPAD {label}"
        )

        plotted = True

    ax.axvline(
        DEAD_TIME_NS,
        linestyle="--",
        color="black",
        label="Nominal dead time"
    )

    ax.set(
        xscale="log",
        yscale="log" if plotted else "linear",
        xlabel=r"Inter-arrival time $\Delta t$ (ns)",
        ylabel=r"Probability density (ns$^{-1}$)",
        title="Inter-arrival distributions"
    )

    ax.legend(frameon=False)
    sns.despine(ax=ax)

    save_manuscript_figure(
        fig,
        folder,
        "interarrival.pdf"
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print("\n--- PHOTON MEASUREMENTS ---")

    print(
        f"Exposure: {seconds:.4f} seconds"
    )

    print(
        f"SPAD A: {na:,} counts | "
        f"{na / seconds:,.2f} cps"
    )

    print(
        f"SPAD B: {nb:,} counts | "
        f"{nb / seconds:,.2f} cps"
    )

    if total:
        print(
            f"A:B split: "
            f"{balance['A_percent']:.2f}% : "
            f"{balance['B_percent']:.2f}%"
        )
    else:
        print("A:B split: N/A (no events)")

    if nb:
        print(
            f"A/B ratio: {balance['A_over_B']:.4f}"
        )
    else:
        print("A/B ratio: N/A")

    print(
        "Dead-time violations:",
        balance["dead_time_violations"]
    )

    if nb < 1000:
        print(
            "Low-count notice: SPAD B has fewer "
            "than 1,000 events."
        )

    if (
        balance["A_above_warning_cps"]
        or balance["B_above_warning_cps"]
    ):
        print(
            "WARNING: Count rate exceeds "
            "configured research threshold."
        )

    return balance


# ============================================================
# EXPLORATORY QRNG EXTRACTION
# ============================================================

def von_neumann_intervals(timestamps):
    """Compare non-overlapping adjacent arrival intervals."""

    differences = np.diff(timestamps)

    pairs = differences[
        :(len(differences) // 2) * 2
    ].reshape(-1, 2)

    return (
        pairs[:, 0] > pairs[:, 1]
    )[
        pairs[:, 0] != pairs[:, 1]
    ].astype(np.uint8)


def toeplitz_hash(bits, output_length):
    """Toeplitz hashing over GF(2)."""

    n = len(bits)
    m = output_length

    if not 0 < m <= n:
        raise ValueError("Invalid Toeplitz dimensions.")

    seed = np.unpackbits(
        np.frombuffer(
            secrets.token_bytes(
                (n + m + 6) // 8
            ),
            dtype=np.uint8
        )
    )[:n + m - 1]

    fft_length = fft.next_fast_len(
        n + m - 1
    )

    convolution = fft.irfft(
        fft.rfft(
            seed.astype(float),
            n=fft_length
        )
        *
        fft.rfft(
            bits.astype(float),
            n=fft_length
        ),
        n=fft_length
    )[
        n - 1:n - 1 + m
    ]

    rounded = np.rint(convolution)

    if np.max(
        np.abs(convolution - rounded)
    ) > 0.25:
        raise ArithmeticError(
            "FFT precision insufficient."
        )

    return (
        (
            rounded.astype(np.int64) & 1
        ).astype(np.uint8),
        seed
    )


def analyze(folder):
    """Analyze saved photon data for research bit statistics."""

    folder = Path(folder)

    info = json.loads(
        (folder / "acquisition.json").read_text(
            encoding="utf-8"
        )
    )

    if not info["valid"]:
        print(
            "Bit extraction blocked:",
            info["status"]
        )
        return

    raw = {
        ch: np.load(
            folder / f"raw_ch{ch}.npy",
            allow_pickle=False
        )
        for ch in (CH_A, CH_B)
    }

    a, b, seconds, _ = common_window(
        raw,
        info.get("target_A"),
        info.get("wall_seconds")
    )

    if seconds <= 0:
        print("No valid exposure for bit analysis.")
        return

    results = {}

    print(
        "\n--- EXPLORATORY QRNG BIT STATISTICS ---"
    )

    for label, timestamps in (
        ("A", a),
        ("B", b)
    ):

        if len(timestamps) < 4:
            print(
                f"SPAD {label}: insufficient photon events."
            )
            continue

        bits = von_neumann_intervals(
            timestamps
        )

        n = len(bits)

        if not n:
            print(
                f"SPAD {label}: no comparison bits."
            )
            continue

        p1 = float(np.mean(bits))
        p0 = 1 - p1

        h = -math.log2(
            max(p0, p1)
        )

        saved = write_bits(
            folder / f"interval_bits_{label}.bin",
            bits
        )

        results[label] = {
            "bits_generated": n,
            "bits_exported": saved,
            "p0": p0,
            "p1": p1,
            "empirical_marginal_hmin": h,
            "yield_bits_per_detection": (
                n / len(timestamps)
            )
        }

        print(
            f"SPAD {label}: "
            f"{n:,} bits | "
            f"P(0)={p0:.5f}, "
            f"P(1)={p1:.5f} | "
            f"empirical Hmin={h:.5f}"
        )

        if label == "A" and TOEPLITZ_DEMO:

            # Exploratory only; no physical entropy
            # certification is implied.

            m = max(
                0,
                min(
                    n,
                    math.floor(n * h - 128)
                )
            )

            if m > 0:

                output, seed = toeplitz_hash(
                    bits,
                    m
                )

                exported = write_bits(
                    folder / "toeplitz_RESEARCH_ONLY.bin",
                    output
                )

                np.packbits(seed).tofile(
                    folder / "toeplitz_RESEARCH_seed.bin"
                )

                results[label][
                    "toeplitz_demo_bits_exported"
                ] = exported

                print(
                    f"SPAD A: Toeplitz demo -> "
                    f"{exported:,} bits."
                )

    results["warning"] = (
        "Empirical bit statistics do not establish "
        "source min-entropy or secure randomness."
    )

    save_json(
        folder / "bit_statistics.json",
        results
    )


# ============================================================
# PERSISTENT CONFIGURATION
# ============================================================

def load_user_config():

    global SAVE_ROOT
    global OPERATOR
    global SAMPLE_ID
    global RUN_NOTES

    try:

        if USER_CONFIG.exists():

            data = json.loads(
                USER_CONFIG.read_text(
                    encoding="utf-8"
                )
            )

            if data.get("save_root"):
                SAVE_ROOT = Path(
                    data["save_root"]
                ).expanduser()

            OPERATOR = data.get(
                "operator",
                OPERATOR
            )

            SAMPLE_ID = data.get(
                "sample_id",
                SAMPLE_ID
            )

            RUN_NOTES = data.get(
                "run_notes",
                RUN_NOTES
            )

    except Exception:
        pass


def save_user_config():

    data = {
        "save_root": str(SAVE_ROOT),
        "operator": OPERATOR,
        "sample_id": SAMPLE_ID,
        "run_notes": RUN_NOTES
    }

    try:

        USER_CONFIG.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

    except Exception:
        pass


def edit_metadata():

    global OPERATOR
    global SAMPLE_ID
    global RUN_NOTES
    global TEMP_SETPOINT_C
    global ITH_MA
    global LASER_BIAS_MA

    print("\n--- EXPERIMENT METADATA ---")

    def text_field(name, current):

        response = input(
            f"{name} [{current or ''}]: "
        ).strip()

        return response if response else current

    def float_field(name, current):

        response = input(
            f"{name} [{current if current is not None else ''}]: "
        ).strip()

        if not response:
            return current

        try:
            return float(response)
        except ValueError:
            print("Invalid number; unchanged.")
            return current

    OPERATOR = text_field(
        "Operator",
        OPERATOR
    )

    SAMPLE_ID = text_field(
        "Sample ID",
        SAMPLE_ID
    )

    RUN_NOTES = text_field(
        "Run notes",
        RUN_NOTES
    )

    TEMP_SETPOINT_C = float_field(
        "TEC setpoint (C)",
        TEMP_SETPOINT_C
    )

    ITH_MA = float_field(
        "Measured threshold current (mA)",
        ITH_MA
    )

    LASER_BIAS_MA = float_field(
        "Laser bias current (mA)",
        LASER_BIAS_MA
    )

    save_user_config()


def set_save_folder():

    global SAVE_ROOT

    print(
        "\nCurrent save folder:",
        SAVE_ROOT
    )

    response = input(
        "New path (Enter = keep, D = Downloads): "
    ).strip()

    if not response:
        return

    if response.upper() == "D":
        candidate = (
            Path.home() / "Downloads" / "qrng_data"
        )
    else:
        candidate = Path(
            response
        ).expanduser()

    candidate.mkdir(
        parents=True,
        exist_ok=True
    )

    SAVE_ROOT = candidate

    save_user_config()

    print(
        "Save folder updated:",
        SAVE_ROOT
    )


# ============================================================
# OFFLINE SELF-TEST
# ============================================================

def selftest():

    from scipy.linalg import toeplitz

    rng = np.random.default_rng(2026)

    for _ in range(8):

        x = rng.integers(
            0,
            2,
            47,
            dtype=np.uint8
        )

        y, seed = toeplitz_hash(
            x,
            23
        )

        dense = toeplitz(
            seed[46:46 + 23],
            seed[46::-1]
        )

        assert np.array_equal(
            y,
            (dense @ x) % 2
        )

    sample = np.array(
        [0, 5, 8, 10, 14]
    )

    expected = np.array(
        [1, 0],
        dtype=np.uint8
    )

    assert np.array_equal(
        von_neumann_intervals(sample),
        expected
    )

    print(
        "Offline self-test PASSED: "
        "Toeplitz and interval comparison."
    )


# ============================================================
# SAVED RUNS AND REANALYSIS
# ============================================================

def find_saved_runs():

    if not SAVE_ROOT.exists():
        return []

    return sorted(
        (
            p
            for p in SAVE_ROOT.iterdir()
            if (
                p.is_dir()
                and (p / "acquisition.json").is_file()
            )
        ),
        key=lambda p: p.name,
        reverse=True
    )


def choose_saved_run():

    if ANALYZE_FOLDER is not None:
        return Path(ANALYZE_FOLDER)

    runs = find_saved_runs()

    if not runs:
        print(
            "No saved runs found in",
            SAVE_ROOT.resolve()
        )
        return None

    print("\nSaved experiment folders:")

    for i, folder in enumerate(
        runs[:8],
        start=1
    ):

        print(
            f"{i}. {folder.name}"
        )

    response = input(
        "Choose 1-8 (Enter = latest), "
        "or enter folder path: "
    ).strip()

    if not response:
        return runs[0]

    if response.isdigit():

        index = int(response)

        if 1 <= index <= min(
            len(runs),
            8
        ):
            return runs[index - 1]

        return None

    folder = Path(
        response
    ).expanduser()

    if not (
        folder / "acquisition.json"
    ).is_file():

        print(
            "acquisition.json not found."
        )

        return None

    return folder


def analyze_saved_run(folder=None):

    if folder is None:
        folder = choose_saved_run()

    if folder is None:
        return

    folder = Path(folder)

    info = json.loads(
        (folder / "acquisition.json").read_text(
            encoding="utf-8"
        )
    )

    if info.get("channels") != [CH_A, CH_B]:
        print(
            "Saved channel settings differ "
            "from the current configuration."
        )
        return

    raw = {
        ch: np.load(
            folder / f"raw_ch{ch}.npy",
            allow_pickle=False
        )
        for ch in (CH_A, CH_B)
    }

    configure_publication_plots()

    characterize(
        raw,
        info,
        folder
    )

    analyze(folder)

    print(
        "Analysis saved to:",
        folder.resolve()
    )


# ============================================================
# RUN ACQUISITION
# ============================================================

def run_acquisition(target=None, timeout=None):

    global SAVE_ROOT

    load_user_config()
    configure_publication_plots()

    SAVE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    folder = SAVE_ROOT / stamp()

    folder.mkdir()

    print(
        "\nRecording BOTH SPAD channels."
    )

    print(
        "Experiment folder:",
        folder.resolve()
    )

    if target is None:

        print(
            "\nCONTINUOUS BASELINE RECORDING"
        )

        print(
            "No photon-count target or time limit."
        )

        print(
            "Press Ctrl+C ONCE in the Spyder console "
            "to stop and save."
        )

        print(
            "Do not use Stop/Restart Kernel."
        )

    acquisition_start = datetime.now(
        timezone.utc
    ).isoformat()

    raw, info = collect(
        target,
        timeout
    )

    acquisition_end = datetime.now(
        timezone.utc
    ).isoformat()

    # Archive raw timestamp arrays before analysis.

    for channel, timestamps in raw.items():

        np.save(
            folder / f"raw_ch{channel}.npy",
            timestamps
        )

    info.update({
        "acquisition_start_utc": acquisition_start,
        "acquisition_end_utc": acquisition_end,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "operator": OPERATOR,
        "sample_id": SAMPLE_ID,
        "run_notes": RUN_NOTES,
        "save_root": str(SAVE_ROOT)
    })

    save_json(
        folder / "acquisition.json",
        info
    )

    print(
        "\nAcquisition status:",
        info["status"]
    )

    if info["error"]:
        print(
            "Details:",
            info["error"]
        )

    characterize(
        raw,
        info,
        folder
    )

    if info["valid"]:
        analyze(folder)
    else:
        print(
            "Raw photon timestamps were saved. "
            "Bit extraction skipped."
        )

    print(
        "\nEXPERIMENT SAVED:",
        folder.resolve()
    )


# ============================================================
# RUN MODES
# ============================================================

def run_mode(mode):

    if mode == "selftest":

        selftest()

    elif mode == "short":

        run_acquisition(
            SHORT_A_EVENTS,
            SHORT_TIMEOUT_S
        )

    elif mode == "full":

        run_acquisition(
            FULL_A_EVENTS,
            FULL_TIMEOUT_S
        )

    elif mode == "analyze":

        analyze_saved_run()

    elif mode == "continuous":

        run_acquisition(
            target=None,
            timeout=None
        )

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )


# ============================================================
# INTERACTIVE EXPERIMENT MENU
# ============================================================

def experiment_menu():

    load_user_config()

    while True:

        print(
            "\n" + "=" * 55
        )

        print(
            "405 nm LASER PHOTON-COUNTING EXPERIMENT"
        )

        print(
            "=" * 55
        )

        print(
            "1. Self-test          Check processing math"
        )

        print(
            f"2. Short run          "
            f"{SHORT_A_EVENTS:,} A events"
        )

        print(
            f"3. Full run           "
            f"{FULL_A_EVENTS:,} A events"
        )

        print(
            "4. Analyze data       Open saved experiment"
        )

        print(
            "5. Edit metadata      Operator and run settings"
        )

        print(
            "6. Set save folder    Change output location"
        )

        print(
            "7. Continuous baseline (manual stop)"
        )

        print(
            "0. Exit"
        )

        print(
            "-" * 55
        )

        print(
            f"SPAD A = channel {CH_A}; "
            f"SPAD B = channel {CH_B}"
        )

        print(
            "Laser settings are controlled manually."
        )

        print(
            "Save folder:",
            SAVE_ROOT
        )

        choice = input(
            "\nSelect 0-7: "
        ).strip()

        if choice in {
            "0",
            "q",
            "quit",
            "exit"
        }:

            print(
                "Experiment menu closed."
            )

            return

        if choice == "5":

            edit_metadata()
            continue

        if choice == "6":

            set_save_folder()
            continue

        choices = {
            "1": "selftest",
            "2": "short",
            "3": "full",
            "4": "analyze",
            "7": "continuous"
        }

        mode = choices.get(
            choice
        )

        if mode is None:

            print(
                "Choose 0, 1, 2, 3, 4, 5, 6, or 7."
            )

            continue

        try:

            run_mode(mode)

        except KeyboardInterrupt:

            print(
                "Interrupted. Returning to menu."
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
            ArithmeticError
        ) as exc:

            print(
                f"Experiment error: "
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if MODE == "menu":
        experiment_menu()

    else:
        run_mode(MODE)
