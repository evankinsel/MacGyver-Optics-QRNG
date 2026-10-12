# -*- coding: utf-8 -*-
"""
Dual-SPAD QRNG research acquisition (v3) | Swabian Time Tagger + Spyder
========================================================================
405 nm diode-laser experiment: simultaneously capture SPAD channels 2 and 3,
archive time tags, verify integrity, characterize count rates / inter-arrival
statistics, compare adjacent intervals on a selected channel, and perform
exploratory Toeplitz hashing.

IMPORTANT SCIENTIFIC LIMITATION
-------------------------------
The 0/1 statistics measured *after* interval comparison do NOT certify a
min-entropy bound for the physical source. Outputs from the default heuristic
mode are research bitstreams, NOT certified cryptographic keys. Deriving an
entropy bound requires a separate source model and validation of physical and
classical noise. Hashing does not repair a compromised or deterministic source.

OPERATION IN SPYDER
-------------------
Edit SETTINGS at the bottom. Run file (F5). Start with RUN_MODE = 'selftest',
then 'short' for ~100k channel-2 tags while both detectors are acquired.
Do not turn on the thermal sweep until I_th(T) is measured and the TEC and
laser-current settings are made manually. This script never controls the laser.

Dependencies: numpy, scipy, matplotlib, and Swabian's Time Tagger driver.
Current Time Tagger installs use `from Swabian import TimeTagger`; older builds
also support `import TimeTagger`. The offline self-test needs no driver.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as sp_fft
from scipy.linalg import toeplitz as dense_toeplitz

try:
    from Swabian import TimeTagger
except ImportError:
    try:
        import TimeTagger  # legacy driver namespace
    except ImportError:
        TimeTagger = None


@dataclass(frozen=True)
class Config:
    serial: str = "1948000SQ6"   # replace if device serial differs
    channel_a: int = 2           # selected QRNG research bitstream
    channel_b: int = 3           # simultaneously captured second SPAD
    trigger_a_v: float = 0.5
    trigger_b_v: float = 0.5
    target_a_events: int = 2_000_000
    max_duration_s: float = 180.0
    buffer_events: int = 500_000
    poll_interval_s: float = 0.05
    status_every_s: float = 5.0
    nominal_dead_time_ps: int = 22_000  # EDIT to actual detector specification
    warning_rate_cps: float = 100_000.0 # experimental ceiling, not a device limit
    min_events_for_processing: int = 1_000
    extractor_epsilon_bits: int = 64
    toeplitz_block_bits: int = 1 << 20
    # None => label h estimate exploratory; a documented independently justified
    # bound may be supplied for source-conditioned entropy calculations.
    justified_entropy_per_bit: Optional[float] = None
    do_exploratory_hash: bool = True
    enable_g2_zero: bool = False
    g2_window_ps: int = 2000  # full bin width, centered at zero; optional diagnostic
    show_plots: bool = False
    out_dir: Path = Path("qrng_data")


CFG = Config()


class TaggerError(RuntimeError):
    pass


class DataIntegrityError(RuntimeError):
    pass


def note(message):
    print("[QRNG] " + message, flush=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def write_json(path: Path, record: dict):
    """Atomic record replacement, used only in an individual run directory."""
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=_plain, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_plain, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def new_run_directory(cfg: Config):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        name += "_" + secrets.token_hex(3)
        path = cfg.out_dir / name
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise OSError("Unable to create a unique acquisition folder")


@contextmanager
def connected_tagger(cfg: Config):
    if TimeTagger is None:
        raise TaggerError("Swabian Time Tagger Python API not imported by this "
                          "Spyder environment. Install/configure the driver first.")
    tagger = None
    try:
        # scanTimeTagger is informational; an empty result is not definitive
        # for already-open devices, so creation is the real check.
        try:
            found = list(TimeTagger.scanTimeTagger())
            if cfg.serial and found and cfg.serial not in found:
                note(f"Device {cfg.serial} absent from discovery {found}; trying anyway")
        except Exception as exc:
            note(f"Device discovery unavailable: {exc}")
        try:
            tagger = TimeTagger.createTimeTagger(cfg.serial)
            tagger.setTriggerLevel(cfg.channel_a, cfg.trigger_a_v)
            tagger.setTriggerLevel(cfg.channel_b, cfg.trigger_b_v)
        except Exception as exc:
            raise TaggerError(f"Cannot open/configure Time Tagger {cfg.serial}: {exc}. "
                              "Check USB, drivers, trigger ranges and other applications.") from exc
        yield tagger
    finally:
        if tagger is not None:
            try:
                TimeTagger.freeTimeTagger(tagger)
            except Exception as exc:
                note(f"WARNING: freeTimeTagger failed: {exc}")


def _collect_data_block(stream, chunks, counts, cfg):
    data = stream.getData()  # explicitly drains stream's internal buffer
    ts = np.asarray(data.getTimestamps(), dtype=np.int64)
    channels = np.asarray(data.getChannels())
    if ts.shape != channels.shape:
        raise DataIntegrityError("Driver returned mismatched timestamp/channel arrays")
    getter = getattr(data, "getMissedEvents", None)
    if not callable(getter):
        missed = None
    else:
        missed_values = np.asarray(getter())
        if missed_values.size and (np.any(missed_values < 0)):
            raise DataIntegrityError("Negative missed-event counter from driver")
        missed = int(np.sum(missed_values, dtype=np.int64))
    # getEventTypes distinguishes genuine photon-edge tags from Error,
    # OverflowBegin/End and MissedEvents. Swabian Tag::Type::TimeTag is 0.
    types_fn = getattr(data, "getEventTypes", None)
    if callable(types_fn):
        types = np.asarray(types_fn())
        if types.shape != ts.shape:
            raise DataIntegrityError("Driver returned mismatched event-type array")
        genuine = (types == 0)
        exceptional_tags = int(np.count_nonzero(~genuine))
    else:
        genuine = np.ones(ts.shape, dtype=bool)
        exceptional_tags = None
    for c in (cfg.channel_a, cfg.channel_b):
        selected = ts[(channels == c) & genuine]
        if selected.size:
            chunks[c].append(selected.copy())
            counts[c] += int(selected.size)
    return missed, exceptional_tags


def acquire_two_channels(cfg: Config):
    """Capture both detectors until A reaches the target or a duration limit.

    The return value includes partial captures if a timeout, Ctrl+C or a stream
    read failure happens *after* a device was opened. All those runs are marked
    invalid for bit extraction but can still be inspected and archived.
    """
    if cfg.channel_a == cfg.channel_b:
        raise ValueError("Choose two different detector channels")
    if cfg.target_a_events < 2 or cfg.max_duration_s <= 0:
        raise ValueError("Invalid acquisition target or time limit")

    chunks = {cfg.channel_a: [], cfg.channel_b: []}
    counts = {cfg.channel_a: 0, cfg.channel_b: 0}
    missed_total, missed_known = 0, True
    exceptional_total, types_known = 0, True
    overflows, overflow_known = None, False
    status, problem = "complete", None
    start_wall = time.monotonic()
    started_utc = utc_now()

    with connected_tagger(cfg) as tagger:
        stream = None
        try:
            if not callable(getattr(tagger, "getOverflows", None)):
                overflow_known = False
            else:
                tagger.clearOverflows()
                overflow_known = True

            stream = TimeTagger.TimeTagStream(
                tagger, int(cfg.buffer_events), [cfg.channel_a, cfg.channel_b]
            )
            stream.start()
            next_status = start_wall + cfg.status_every_s

            while counts[cfg.channel_a] < cfg.target_a_events:
                elapsed = time.monotonic() - start_wall
                if elapsed >= cfg.max_duration_s:
                    status = "timeout"
                    problem = "Channel A target not reached before time limit"
                    break
                time.sleep(cfg.poll_interval_s)
                try:
                    missed, unexpected = _collect_data_block(stream, chunks, counts, cfg)
                    missed_known &= missed is not None
                    types_known &= unexpected is not None
                    missed_total += missed or 0
                    exceptional_total += unexpected or 0
                except Exception as exc:
                    status = "stream_error"
                    problem = f"{type(exc).__name__}: {exc}"
                    break

                if cfg.status_every_s > 0 and time.monotonic() >= next_status:
                    duration = max(time.monotonic() - start_wall, 1e-6)
                    note(f"Recording A(ch{cfg.channel_a}) {counts[cfg.channel_a]:,} "
                         f"({counts[cfg.channel_a]/duration:,.0f} approx cps); "
                         f"B(ch{cfg.channel_b}) {counts[cfg.channel_b]:,} "
                         f"({counts[cfg.channel_b]/duration:,.0f} approx cps)")
                    next_status = time.monotonic() + cfg.status_every_s

        except KeyboardInterrupt:
            status = "interrupted"
            problem = "User interrupted acquisition (Ctrl+C)"
        except Exception as exc:
            status = "setup_or_driver_error"
            problem = f"{type(exc).__name__}: {exc}"
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    # Drain tags already buffered at stop. Do not extend time target.
                    missed, unexpected = _collect_data_block(stream, chunks, counts, cfg)
                    missed_known &= missed is not None
                    types_known &= unexpected is not None
                    missed_total += missed or 0
                    exceptional_total += unexpected or 0
                except Exception as exc:
                    status = "stream_error"
                    problem = f"Final stream drain failed: {exc}"
                stream = None
            if overflow_known:
                try:
                    overflows = int(tagger.getOverflows())
                except Exception as exc:
                    overflow_known = False
                    problem = (problem or "") + f" Overflow query failed: {exc}"

    raw = {c: (np.concatenate(chunks[c]) if chunks[c]
               else np.empty(0, dtype=np.int64)) for c in counts}
    wall_s = time.monotonic() - start_wall
    a = raw[cfg.channel_a]
    if a.size:
        # Common Time Tagger timestamp interval, and both channels measured
        # over the same window. A is trimmed to exactly the requested count.
        a = a[:cfg.target_a_events]
        start_ps, stop_ps = int(a[0]), int(a[-1])
        raw[cfg.channel_a] = a
        b = raw[cfg.channel_b]
        raw[cfg.channel_b] = b[(b >= start_ps) & (b <= stop_ps)]
        duration_s = (stop_ps - start_ps) * 1e-12
    else:
        duration_s = 0.0
        start_ps = stop_ps = None
        raw[cfg.channel_b] = np.empty(0, dtype=np.int64)

    if status == "complete" and a.size < cfg.target_a_events:
        status = "timeout"
        problem = "Incomplete primary stream after final drain"

    for c, arr in raw.items():
        if arr.size > 1 and np.any(np.diff(arr) <= 0):
            status = "bad_timestamps"
            problem = f"Non-increasing timestamps on channel {c}"

    if status == "complete" and (overflows or missed_total or exceptional_total):
        status = "lost_or_error_tags"
        problem = (f"Found {overflows} hardware overflows, {missed_total} missed events "
                   f"and {exceptional_total} non-photon/error tags")
    if status == "complete" and (not overflow_known or not missed_known or not types_known):
        status = "integrity_unverified"
        problem = "Driver does not expose all required loss counters"
    if duration_s <= 0 and status == "complete":
        status = "invalid_window"
        problem = "No positive time window"

    rates = {str(c): (len(raw[c]) / duration_s if duration_s > 0 else None)
             for c in raw}
    second_detector_suspect = len(raw[cfg.channel_b]) < cfg.min_events_for_processing
    if second_detector_suspect:
        note(f"WARNING: SPAD B / channel {cfg.channel_b} has only "
             f"{len(raw[cfg.channel_b]):,} events; check cabling, illumination and trigger level")
    info = {
        "status": status, "problem": problem,
        "integrity_ok": status == "complete",
        "started_utc": started_utc,
        "duration_s": duration_s, "elapsed_wall_s": wall_s,
        "window_start_ps": start_ps, "window_stop_ps": stop_ps,
        "events": {str(c): int(len(raw[c])) for c in raw},
        "rates_cps": rates,
        "hardware_overflows": overflows,
        "missed_events": missed_total if missed_known else None,
        "non_photon_event_tags": exceptional_total if types_known else None,
        "loss_counters_verified": bool(overflow_known and missed_known and types_known),
        "second_detector_low_counts": second_detector_suspect,
        "trigger_volts": {str(cfg.channel_a): cfg.trigger_a_v,
                          str(cfg.channel_b): cfg.trigger_b_v},
        "serial": cfg.serial,
    }
    return raw, info


def save_raw(raw: dict, folder: Path):
    for ch, arr in raw.items():
        np.save(folder / f"timestamps_ch{ch}.npy", np.asarray(arr, dtype=np.int64))


def interarrival_diagnostics(timestamps: np.ndarray, channel: int, cfg: Config,
                             folder: Path):
    if timestamps.size < cfg.min_events_for_processing:
        return {"channel": channel, "events": int(timestamps.size),
                "diagnostics": "insufficient_events"}
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        raise DataIntegrityError(f"Channel {channel} timestamps not strictly increasing")
    violation_count = int(np.count_nonzero(dt < cfg.nominal_dead_time_ps))
    mean_ps = float(np.mean(dt))
    min_ps = int(np.min(dt))

    # A short-window excess is only a *screening indicator*, not a calibrated
    # afterpulsing or bunching estimate (detector response matters).
    short_limit = 10 * cfg.nominal_dead_time_ps
    short_fraction = float(np.mean(dt < short_limit))
    stat = {
        "channel": channel, "events": int(timestamps.size),
        "min_interval_ps": min_ps,
        "mean_interval_us": mean_ps / 1e6,
        "rate_from_mean_cps": 1e12 / mean_ps,
        "deadtime_violations": violation_count,
        "fraction_intervals_below_10x_deadtime": short_fraction,
    }
    if violation_count:
        note(f"WARNING ch{channel}: {violation_count} intervals shorter than configured "
             f"{cfg.nominal_dead_time_ps / 1e3:g} ns detector dead time")

    good = dt[dt > 0]
    if int(good.min()) < int(good.max()):
        edges = np.geomspace(float(good.min()), float(good.max()) + 1.0, 100)
        density, _ = np.histogram(dt, bins=edges)
        # Counts per nanosecond; edges and x-axis are now consistently in ns.
        widths_ns = np.diff(edges) / 1e3
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.stairs(density / widths_ns, edges / 1e3, label=f"Ch{channel}")
        ax.axvline(cfg.nominal_dead_time_ps / 1e3, linestyle="--", color="black",
                   label="Configured dead time")
        ax.set(xscale="log", yscale="log", xlabel="Inter-arrival interval (ns)",
               ylabel="Counts per ns", title=f"SPAD {channel} inter-arrival distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / f"interarrival_ch{channel}.png", dpi=160)
        if cfg.show_plots:
            plt.show()
        plt.close(fig)
    return stat


def count_rate_trace(a: np.ndarray, b: np.ndarray, info: dict, cfg: Config,
                     folder: Path):
    """Save a shared-window trace for A/B, in tabular and plot form.

    Time-bin rates are helpful for finding drift or sudden alignment changes;
    they are *not* a replacement for independent dark-count measurements.
    """
    seconds = info.get("duration_s") or 0.0
    if seconds <= 0:
        return None
    # Aim for one-second bins; adapt short runs to give at least a few bins.
    n_bins = max(1, min(300, int(math.ceil(seconds / max(0.2, min(1.0, seconds / 10))))))
    edges = np.linspace(0, seconds, n_bins + 1)
    widths = np.diff(edges)
    origin = info["window_start_ps"]
    a_seconds = (a - origin) * 1e-12
    b_seconds = (b - origin) * 1e-12
    counts_a, _ = np.histogram(a_seconds, bins=edges)
    counts_b, _ = np.histogram(b_seconds, bins=edges)
    rates_a, rates_b = counts_a / widths, counts_b / widths
    center = (edges[:-1] + edges[1:]) / 2
    table = np.column_stack([center, rates_a, rates_b, counts_a, counts_b])
    np.savetxt(folder / "count_rate_trace.csv", table, delimiter=",",
               header=f"time_s,rate_ch{cfg.channel_a}_cps,rate_ch{cfg.channel_b}_cps,"
                      f"counts_ch{cfg.channel_a},counts_ch{cfg.channel_b}", comments="")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(center, rates_a, where="mid", label=f"SPAD A / ch{cfg.channel_a}")
    ax.step(center, rates_b, where="mid", label=f"SPAD B / ch{cfg.channel_b}")
    ax.set(xlabel="Seconds since start of common window", ylabel="Count rate (cps)",
           title="Dual-SPAD count-rate stability")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "count_rate_trace.png", dpi=160)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    combined = len(a) + len(b)
    return {"bins": n_bins, "bin_seconds": seconds / n_bins,
            "relative_count_imbalance": (len(a)-len(b))/combined if combined else None,
            "channel_a_rate_cv": float(np.std(rates_a) / np.mean(rates_a)) if np.mean(rates_a) else None,
            "channel_b_rate_cv": float(np.std(rates_b) / np.mean(rates_b)) if np.mean(rates_b) else None,
            "note": "CV includes shot noise; imbalance is not corrected for detector efficiencies"}


def interval_pair_bits(timestamps: np.ndarray):
    """Pair successive *non-overlapping* adjacent inter-arrival times.

    dt[0] > dt[1] -> 1, dt[0] < dt[1] -> 0; equal intervals discarded.
    This resembles a von Neumann comparator but unbiased/independent output is
    NOT guaranteed without assumptions on the detector and source process.
    """
    if timestamps.size < 3:
        return np.empty(0, np.uint8)
    intervals = np.diff(timestamps)
    pairs = intervals[: 2 * (intervals.size // 2)].reshape(-1, 2)
    x, y = pairs[:, 0], pairs[:, 1]
    return np.asarray(x[x != y] > y[x != y], dtype=np.uint8)


def entropy_diagnostics(bits: np.ndarray):
    """Exploratory statistics, not an SP 800-90B assessment or formal H_inf."""
    n = len(bits)
    if n < 16:
        return {"n_bits": n, "p_one": None,
                "bit_mcv_heuristic": 0.0, "block_mcv_heuristic": 0.0,
                "heuristic_per_bit": 0.0}
    p_one = float(np.mean(bits))
    p_largest = max(p_one, 1 - p_one)
    z = 2.576

    def upper_confidence(value, n_samples):
        return min(1.0, value + z * math.sqrt(value * (1.0 - value) / (n_samples - 1)))

    bit_h = -math.log2(upper_confidence(p_largest, n))
    groups = n // 8
    if groups >= 100:
        symbols = np.packbits(bits[:8 * groups], bitorder="big")
        symbols_counts = np.bincount(symbols, minlength=256)
        symbol_p = float(np.max(symbols_counts) / groups)
        block_h = -math.log2(upper_confidence(symbol_p, groups)) / 8.0
    else:
        block_h = 0.0
    return {
        "n_bits": n,
        "p_one": p_one,
        "bit_mcv_heuristic": bit_h,
        "block_mcv_heuristic": block_h,
        "heuristic_per_bit": min(bit_h, block_h),
        "warning": "Post-comparator empirical estimates are not certified source entropy",
    }


def random_seed_bits(num_bits: int):
    if num_bits < 0:
        raise ValueError("num_bits must be nonnegative")
    token = secrets.token_bytes((num_bits + 7) // 8)
    return np.unpackbits(np.frombuffer(token, dtype=np.uint8))[:num_bits].copy()


def toeplitz_hash(x: np.ndarray, m: int, seed: Optional[np.ndarray] = None):
    """FFT Toeplitz y_i = sum_j s[n-1+i-j] x_j (mod 2).

    A truncated cyclic convolution of length L >= len(s)=n+m-1 is exact for
    the requested segment [n-1:n-1+m]: wraparound terms only affect earlier
    indices, not that segment. Self-test checks against a dense GF(2) matrix.
    """
    x = np.asarray(x, dtype=np.uint8)
    n = int(x.size)
    if not (0 <= m <= n) or np.any(x > 1):
        raise ValueError("Invalid Toeplitz dimensions or nonbinary input")
    if m == 0:
        return np.empty(0, np.uint8), np.empty(0, np.uint8)
    required = n + m - 1
    seed = random_seed_bits(required) if seed is None else np.asarray(seed, dtype=np.uint8)
    if seed.size != required or np.any(seed > 1):
        raise ValueError(f"Toeplitz seed must have {required} binary bits")
    length = sp_fft.next_fast_len(required, real=True)
    fs = sp_fft.rfft(seed.astype(np.float64), n=length)
    fx = sp_fft.rfft(x.astype(np.float64), n=length)
    values = sp_fft.irfft(fs * fx, n=length)[n-1:n-1+m]
    rounded = np.rint(values)
    if values.size and np.max(np.abs(values-rounded)) > 0.25:
        raise FloatingPointError("FFT convolution precision check failed")
    return (rounded.astype(np.int64) & 1).astype(np.uint8), seed


def extract_blocks(bits: np.ndarray, h_per_bit: float, cfg: Config):
    if not (0 <= h_per_bit <= 1):
        raise ValueError("Entropy bound must be between 0 and 1 bit/bit")
    out_blocks, seed_blocks, details = [], [], []
    for i in range(0, len(bits), cfg.toeplitz_block_bits):
        block = bits[i:i + cfg.toeplitz_block_bits]
        size = max(0, math.floor(len(block) * h_per_bit - 2 * cfg.extractor_epsilon_bits))
        if size == 0:
            details.append({"input_bits": len(block), "output_bits": 0, "seed_bits": 0})
            continue
        hashed, seed = toeplitz_hash(block, size)
        out_blocks.append(hashed)
        seed_blocks.append(seed)
        details.append({"input_bits": len(block), "output_bits": len(hashed),
                        "seed_bits": len(seed)})
    empty = np.empty(0, dtype=np.uint8)
    return (np.concatenate(out_blocks) if out_blocks else empty,
            np.concatenate(seed_blocks) if seed_blocks else empty, details)


def write_packed_bits(bits: np.ndarray, target: Path):
    """Write full bytes only (no invented padding), then verify on disk."""
    bits = np.asarray(bits, dtype=np.uint8)
    if np.any(bits > 1):
        raise ValueError("Only binary arrays can be exported")
    n = len(bits) - len(bits) % 8
    packed = np.packbits(bits[:n], bitorder="big")
    tmp = target.with_name(target.name + ".tmp")
    packed.tofile(tmp)
    os.replace(tmp, target)
    back = np.fromfile(target, dtype=np.uint8)
    if not np.array_equal(back, packed):
        raise IOError(f"Binary verification failed: {target}")
    return n


def g2_at_zero_diagnostic(a: np.ndarray, b: np.ndarray, window_ps: int):
    """Simple raw correlation bin about zero, not a jitter/delay corrected g2(0).

    Coincidences / (rate_A * rate_B * full_bin_seconds * exposure_seconds).
    Use only as a preliminary HBT indicator, never a definitive g2(0).
    """
    if window_ps <= 0:
        raise ValueError("g2_window_ps must be > 0")
    if a.size < 2 or b.size < 2:
        return {"g2_zero_raw": None, "coincidences": 0}
    exposure_s = (a[-1] - a[0]) * 1e-12
    left = -(window_ps // 2)
    right = left + window_ps
    events_per_a = np.searchsorted(b, a + right, side="left") - np.searchsorted(
        b, a + left, side="left")
    found = int(np.sum(events_per_a, dtype=np.int64))
    expected = (len(a) / exposure_s) * (len(b) / exposure_s) * (window_ps * 1e-12) * exposure_s
    return {"g2_zero_raw": float(found / expected) if expected > 0 else None,
            "coincidences": found, "expected_accidental_coincidences": expected,
            "window_ps": window_ps,
            "warning": "Uncorrected for channel timing offset, jitter, dark counts, drift"}


def process_saved_run(folder: Path, cfg: Config = CFG, acq_info: Optional[dict] = None,
                      force_diagnostics_only=False):
    folder = Path(folder)
    if acq_info is None:
        info_file = folder / "acquisition.json"
        if not info_file.exists():
            raise FileNotFoundError("Reprocessing requires acquisition.json in the run folder")
        acq_info = json.loads(info_file.read_text(encoding="utf-8"))
    a = np.load(folder / f"timestamps_ch{cfg.channel_a}.npy", allow_pickle=False)
    b = np.load(folder / f"timestamps_ch{cfg.channel_b}.npy", allow_pickle=False)

    stats = {}
    for channel, arr in ((cfg.channel_a, a), (cfg.channel_b, b)):
        stats[str(channel)] = interarrival_diagnostics(arr, channel, cfg, folder)
        rate = acq_info.get("rates_cps", {}).get(str(channel))
        if rate is None:
            note(f"Ch{channel}: {len(arr):,} events, no valid common-window count rate")
        else:
            note(f"Ch{channel}: {len(arr):,} events | {rate:,.1f} cps")
            if rate > cfg.warning_rate_cps:
                note(f"Ch{channel} WARNING: above configured {cfg.warning_rate_cps:,.0f} cps ceiling")

    summary = {"analysis_timestamp_utc": utc_now(),
               "run_folder": str(folder), "acquisition": acq_info,
               "channel_diagnostics": stats, "extractor": None,
               "research_bitstream_only": True}
    summary["shared_count_rate_trace"] = count_rate_trace(a, b, acq_info, cfg, folder)
    if cfg.enable_g2_zero and acq_info.get("integrity_ok"):
        summary["g2_preliminary"] = g2_at_zero_diagnostic(a, b, cfg.g2_window_ps)
    else:
        summary["g2_preliminary"] = None

    if force_diagnostics_only or not acq_info.get("integrity_ok", False):
        summary["extractor_skipped"] = "Acquisition not verified or diagnostics-only requested"
    elif len(a) < cfg.min_events_for_processing:
        summary["extractor_skipped"] = "Insufficient timestamps"
    else:
        bits = interval_pair_bits(a)
        estimate = entropy_diagnostics(bits)
        num_comparator_bits = write_packed_bits(bits, folder / "comparison_bits.bin")
        summary["extractor"] = {
            "comparison_bits_generated": int(len(bits)),
            "comparison_bits_exported": num_comparator_bits,
            "comparison_yield_per_primary_event": len(bits) / len(a),
            "diagnostics": estimate,
        }
        if cfg.do_exploratory_hash and estimate["heuristic_per_bit"] > 0:
            if cfg.justified_entropy_per_bit is None:
                entropy = estimate["heuristic_per_bit"]
                basis = "exploratory_post_comparator_heuristic_NOT_a_security_bound"
            else:
                entropy = cfg.justified_entropy_per_bit
                basis = "externally_supplied_bound_NOT_independently_verified_by_script"
            hashed, seed, blocks = extract_blocks(bits, entropy, cfg)
            n_hash = write_packed_bits(hashed, folder / "toeplitz_output_RESEARCH.bin")
            # Save exactly the concatenated seed bit length in metadata. Packbits
            # pads only the archived seed file, never an extractor output file.
            np.packbits(seed, bitorder="big").tofile(folder / "toeplitz_seeds.bin")
            summary["extractor"].update({
                "toeplitz_output_bits_generated": len(hashed),
                "toeplitz_output_bits_exported": n_hash,
                "toeplitz_seed_bits": len(seed),
                "seed_file_padded_bits": (-len(seed)) % 8,
                "blocks": blocks,
                "entropy_per_comparator_bit_used": entropy,
                "entropy_basis": basis,
                "eps_per_block": f"2^-{cfg.extractor_epsilon_bits}",
                "warning": "Research output; not certified quantum entropy or secure keys",
            })
            note(f"Exported {n_hash:,} Toeplitz research bits and {len(seed):,} seed bits")
    write_json(folder / "analysis.json", summary)
    return summary


def run_point(cfg: Config = CFG, temperature_c=None, ith_ma=None,
              measured_temperature_c=None):
    folder = new_run_directory(cfg)
    note(f"Run directory: {folder.resolve()}")
    note(f"Capturing channels {cfg.channel_a} AND {cfg.channel_b}; "
         f"stop target is {cfg.target_a_events:,} events on channel {cfg.channel_a}")
    try:
        raw, info = acquire_two_channels(cfg)
    except Exception as exc:
        failure = {"utc": utc_now(), "error": f"{type(exc).__name__}: {exc}",
                   "temperature_c": temperature_c, "ith_ma": ith_ma}
        write_json(folder / "connection_error.json", failure)
        raise
    info.update({"temperature_setpoint_c": temperature_c,
                 "temperature_measured_c": measured_temperature_c,
                 "threshold_current_ma": ith_ma,
                 "nominal_bias_current_ma": (1.02 * ith_ma if ith_ma is not None else None),
                 "bias_current_note": "Metadata only; script never sets laser current",
                 "configuration": asdict(cfg),
                 "numpy_version": np.__version__,
                 "swabian_api_version": (TimeTagger.getVersion()
                      if TimeTagger is not None and hasattr(TimeTagger, "getVersion") else None)})
    save_raw(raw, folder)  # SAVE RAW BEFORE ANY ANALYSIS, EVEN FOR AN INVALID RUN
    write_json(folder / "acquisition.json", info)
    note(f"Acquisition status: {info['status']}")
    if info["problem"]:
        note(f"Reason: {info['problem']}")
    try:
        summary = process_saved_run(folder, cfg, info)
        record = {"run_folder": str(folder), "acquisition_status": info["status"],
                  "counts": info["events"], "rates_cps": info["rates_cps"],
                  "output_bits": (summary.get("extractor") or {}).get(
                      "toeplitz_output_bits_exported", 0)}
        append_jsonl(cfg.out_dir / "experiments.jsonl", record)
    except Exception as exc:
        note(f"Analysis failed, but raw timestamps remain at {folder.resolve()}: {exc}")
        write_json(folder / "analysis_error.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    return summary


def selftest():
    """Offline reproducible tests of hashing, I/O and both-channel mock DAQ."""
    rng = np.random.default_rng(1948)
    for n, m in ((53, 29), (204, 100), (257, 256), (1023, 751)):
        for _ in range(3):
            x = rng.integers(0, 2, n, dtype=np.uint8)
            seed = rng.integers(0, 2, n+m-1, dtype=np.uint8)
            mat = dense_toeplitz(seed[n-1:n-1+m], seed[n-1::-1]).astype(np.int64)
            reference = (mat @ x.astype(np.int64)) % 2
            result, _ = toeplitz_hash(x, m, seed)
            assert np.array_equal(reference, result), f"Toeplitz mismatch for {n}x{m}"
    assert np.array_equal(interval_pair_bits(np.array([0, 10, 13, 20, 22])),
                          np.array([1, 1], np.uint8))
    note("Offline Toeplitz GF(2) self-test: PASS (12 matrix checks)")
    note("Interval-comparison self-test: PASS")
    test_mock_acquisition()


def test_mock_acquisition():
    """Simulate TimeTagStream with two detector channels; no physical device used."""
    global TimeTagger
    original = TimeTagger

    class Buffer:
        def __init__(self, ts, channels):
            self.ts, self.channels = ts, channels
        def getTimestamps(self):
            return self.ts
        def getChannels(self):
            return self.channels
        def getMissedEvents(self):
            return np.zeros(len(self.ts), dtype=np.int64)
        def getEventTypes(self):
            return np.zeros(len(self.ts), dtype=np.int32)

    class Stream:
        def __init__(self, tagger, n, channels):
            self.used = False
        def start(self):
            pass
        def stop(self):
            pass
        def getData(self):
            if self.used:
                return Buffer(np.empty(0, np.int64), np.empty(0, np.int32))
            self.used = True
            # A: [0, 10, 20, 30, 40], B: [-10, 5, 25, 35, 55]
            return Buffer(np.array([-10, 0, 5, 10, 20, 25, 30, 35, 40, 55], np.int64),
                          np.array([3, 2, 3, 2, 2, 3, 2, 3, 2, 3], np.int32))

    class Device:
        def setTriggerLevel(self, ch, v):
            pass
        def clearOverflows(self):
            pass
        def getOverflows(self):
            return 0

    class Driver:
        @staticmethod
        def scanTimeTagger():
            return ["MOCK"]
        @staticmethod
        def createTimeTagger(serial):
            return Device()
        @staticmethod
        def freeTimeTagger(tagger):
            pass
        TimeTagStream = Stream

    try:
        TimeTagger = Driver
        fake_cfg = replace(CFG, serial="MOCK", target_a_events=5,
                           max_duration_s=1, status_every_s=100)
        result, info = acquire_two_channels(fake_cfg)
        assert info["integrity_ok"] and info["status"] == "complete", info
        assert result[2].tolist() == [0, 10, 20, 30, 40]
        assert result[3].tolist() == [5, 25, 35], result[3]
        assert info["events"] == {"2": 5, "3": 3}
        assert info["rates_cps"]["2"] == 5 / (40e-12)
        note("Offline dual-channel mock acquisition: PASS")
    finally:
        TimeTagger = original


# %% SETTINGS: edit here; F5 to run one mode in Spyder.
RUN_MODE = "selftest"  # selftest | short | point | sweep | reprocess
TEMPERATURE_C = None  # e.g., 25.0, or leave None if not logged
ITH_MA = None          # measured threshold current at this temperature, or None
MEASURED_TEMPERATURE_C = None
SWEEP_POINTS = []     # [(20.0, measured_ith20), (22.5, measured_ith225), ...]
REPROCESS_FOLDER = None  # e.g. Path('qrng_data/20261011T...')


if __name__ == "__main__":
    if RUN_MODE == "selftest":
        selftest()
    elif RUN_MODE == "short":
        short = replace(CFG, target_a_events=100_000, max_duration_s=60)
        result = run_point(short, TEMPERATURE_C, ITH_MA, MEASURED_TEMPERATURE_C)
    elif RUN_MODE == "point":
        result = run_point(CFG, TEMPERATURE_C, ITH_MA, MEASURED_TEMPERATURE_C)
    elif RUN_MODE == "sweep":
        if not SWEEP_POINTS:
            raise ValueError("Provide measured (temperature_C, I_th_mA) in SWEEP_POINTS")
        for temp, threshold in SWEEP_POINTS:
            note(f"MANUALLY set TEC to {temp} C, allow thermal soak, and set bias "
                 f"current per approved procedure (nominal 1.02*I_th={1.02*threshold:.3f} mA)")
            input("Press Enter only after the operator has confirmed the laser settings...")
            result = run_point(CFG, temp, threshold)
    elif RUN_MODE == "reprocess":
        if REPROCESS_FOLDER is None:
            raise ValueError("Set REPROCESS_FOLDER to an existing run directory")
        result = process_saved_run(Path(REPROCESS_FOLDER), CFG)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")
