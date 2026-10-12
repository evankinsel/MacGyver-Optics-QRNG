# -*- coding: utf-8 -*-
"""Lean dual-SPAD photon-counting / exploratory QRNG experiment (Spyder).

Run with MODE="menu" and choose a numbered option in the Spyder console.
Self-test needs no hardware; acquisition modes require the Time Tagger.
Both detectors are recorded simultaneously. Acquisition runs save raw timestamps,
counts, shared-window A:B balance, and two plots. Analysis uses SAVED DATA:
inter-arrival comparisons, empirical bias/min-entropy, optional Toeplitz demo.

IMPORTANT: The extracted bits are RESEARCH DATA, NOT certified random keys.
Neither measured bit bias nor NIST tests prove physical-source min-entropy.
This script does not change laser current, TEC temperature, or optical alignment.

Requires numpy, scipy and matplotlib; hardware modes require Swabian's driver.
Timestamps and instrument delay/dead-time values are in PICOSECONDS.
"""
from __future__ import annotations

import json
import math
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

try:
    from Swabian import TimeTagger
except ImportError:
    try:
        import TimeTagger
    except ImportError:
        TimeTagger = None


# ========= EDIT THESE SETTINGS IN SPYDER =========
MODE = "menu"               # menu | selftest | short | full | analyze
SERIAL = "1948000SQ6"       # Time Tagger serial
CH_A, CH_B = 2, 3          # physical Tagger inputs; confirm wiring
TRIGGER_A_V = 0.5           # check SPAD output specifications
TRIGGER_B_V = 0.5
DEAD_TIME_NS = 22.0         # nominal ONLY: verify actual SPAD dead time
WARNING_CPS = 100_000
SHORT_A_EVENTS = 100_000
FULL_A_EVENTS = 2_000_000
SHORT_TIMEOUT_S = 60
FULL_TIMEOUT_S = 180
STREAM_BUFFER_EVENTS = 500_000
SAVE_ROOT = Path("qrng_data")
ANALYZE_FOLDER = None        # None: choose a saved run from the menu
TOEPLITZ_DEMO = True        # research demonstration, NOT validated key material
SHOW_FIGURES = True         # display figures in Spyder; always save PNG files
TEMP_SETPOINT_C = None      # enter TEC temperature if measured
ITH_MA = None               # measured threshold current at this temperature
LASER_BIAS_MA = None        # manually set current; script DOES NOT set it
# ================================================


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def write_bits(path, bits):
    """Save MSB-first bytes; truncate fractional last byte (no zero padding)."""
    count = len(bits) // 8 * 8
    np.packbits(bits[:count], bitorder="big").tofile(path)
    return count


def _drain(stream, buffers, counters):
    """Drain TimeTagStream; count only valid photon tags; detect losses."""
    block = stream.getData()
    ts = np.asarray(block.getTimestamps(), dtype=np.int64)
    ch = np.asarray(block.getChannels())
    if ts.shape != ch.shape:
        raise RuntimeError("Swabian returned inconsistent timestamps and channels")
    get_types = getattr(block, "getEventTypes", None)
    get_missed = getattr(block, "getMissedEvents", None)
    if not callable(get_types) or not callable(get_missed):
        raise RuntimeError("Swabian driver lacks event-type or loss information")
    event_types = np.asarray(get_types())
    missed_tags = np.asarray(get_missed())
    if event_types.shape != ts.shape or missed_tags.shape != ts.shape:
        raise RuntimeError("Swabian loss arrays have unexpected dimensions")
    if np.any(missed_tags < 0):
        raise RuntimeError("Negative missed-event counter")
    mask = event_types == 0    # Swabian Tag::Type::TimeTag is 0
    for channel in (CH_A, CH_B):
        selected = ts[mask & (ch == channel)]
        if selected.size:
            buffers[channel].append(selected.copy())
            counters[channel] += int(selected.size)
    return int(missed_tags.sum()), int(np.count_nonzero(~mask))


def collect(target, timeout):
    """Acquire both channels; return FULL raw arrays + validity information.

    Even timeout/stream failures return partial data for archiving. Counts/rates
    are evaluated later on the SAME [first A, last A] timestamp window.
    """
    if CH_A == CH_B or target < 2 or timeout <= 0:
        raise ValueError("Check channel choices, event target, and timeout")
    if TimeTagger is None:
        raise RuntimeError("Cannot import Swabian TimeTagger in this Spyder environment")

    buffers = {CH_A: [], CH_B: []}
    counters = {CH_A: 0, CH_B: 0}
    missed, invalid_types, overflows = 0, 0, None
    status, error = "complete", None
    tagger, stream = None, None
    start = time.monotonic()
    try:
        tagger = TimeTagger.createTimeTagger(SERIAL)
        tagger.setTriggerLevel(CH_A, TRIGGER_A_V)
        tagger.setTriggerLevel(CH_B, TRIGGER_B_V)
        tagger.clearOverflows()
        stream = TimeTagger.TimeTagStream(
            tagger, int(STREAM_BUFFER_EVENTS), [CH_A, CH_B]
        )
        stream.start()
        start = time.monotonic()
        next_print = start + 5
        while counters[CH_A] < target:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                status, error = "timeout", "Channel A did not reach event target"
                break
            time.sleep(0.05)
            m, problems = _drain(stream, buffers, counters)
            missed += m
            invalid_types += problems
            if time.monotonic() >= next_print:
                dt = max(time.monotonic() - start, 0.001)
                print(f"Live: A {counters[CH_A]:,} ({counters[CH_A]/dt:,.0f} cps) | "
                      f"B {counters[CH_B]:,} ({counters[CH_B]/dt:,.0f} cps)")
                next_print += 5
    except KeyboardInterrupt:
        status, error = "interrupted", "Interrupted from Spyder"
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if stream is not None:
            try:
                stream.stop()
                m, problems = _drain(stream, buffers, counters)
                missed += m
                invalid_types += problems
            except Exception as exc:
                status, error = "error", f"Could not drain stopped stream: {exc}"
            stream = None
        if tagger is not None:
            try:
                overflows = int(tagger.getOverflows())
            except Exception as exc:
                status, error = "error", f"Could not read hardware overflows: {exc}"
            try:
                TimeTagger.freeTimeTagger(tagger)
            except Exception as exc:
                status, error = "error", f"Device release failed: {exc}"

    raw = {c: np.concatenate(buffers[c]) if buffers[c]
           else np.empty(0, dtype=np.int64) for c in (CH_A, CH_B)}
    if status == "complete" and counters[CH_A] < target:
        status, error = "incomplete", "Fewer A photon events than requested"
    if status == "complete" and (missed or invalid_types or overflows):
        status, error = "data_loss", "Hardware missed/overflow/error tags occurred"
    for c, ts in raw.items():
        if len(ts) > 1 and np.any(np.diff(ts) <= 0):
            status, error = "timestamp_error", f"Non-increasing channel {c} timestamps"
    info = {
        "status": status, "error": error, "valid": status == "complete",
        "raw_events": {str(c): len(raw[c]) for c in raw},
        "target_A": target, "wall_seconds": round(time.monotonic()-start, 3),
        "missed_events": missed, "non_photon_tags": invalid_types,
        "hardware_overflows": overflows, "serial": SERIAL,
        "channels": [CH_A, CH_B], "trigger_volts": [TRIGGER_A_V, TRIGGER_B_V],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_setpoint_C": TEMP_SETPOINT_C,
        "threshold_current_mA": ITH_MA,
        "laser_bias_current_mA": LASER_BIAS_MA
    }
    return raw, info


def common_window(raw, target=None):
    """Ensure BOTH count rates and the A:B ratio use identical exposure."""
    a = np.asarray(raw[CH_A], dtype=np.int64)
    b = np.asarray(raw[CH_B], dtype=np.int64)
    if target is not None:
        a = a[:target]
    if len(a) < 2:
        return a, np.empty(0, dtype=np.int64), 0., None
    lo, hi = int(a[0]), int(a[-1])
    b = b[(b >= lo) & (b <= hi)]
    return a, b, (hi-lo)*1e-12, (lo, hi)


def characterize(raw, info, folder):
    a, b, seconds, limits = common_window(raw, info.get("target_A"))
    if seconds <= 0:
        report = {"problem": "No positive shared exposure window"}
        save_json(folder / "measurements.json", report)
        return report
    na, nb = len(a), len(b)
    total = na+nb
    balance = {
        "exposure_seconds": seconds,
        "common_window_ps": list(limits),
        "A_count": na, "B_count": nb,
        "A_cps": na/seconds, "B_cps": nb/seconds,
        "A_over_B": na/nb if nb else None,
        "A_percent": 100*na/total if total else None,
        "B_percent": 100*nb/total if total else None,
        "normalized_imbalance": (na-nb)/total if total else None,
        "A_above_warning_cps": na/seconds > WARNING_CPS,
        "B_above_warning_cps": nb/seconds > WARNING_CPS,
        "dead_time_violations": {},
    }
    for label, ts in (("A",a),("B",b)):
        intervals = np.diff(ts)
        balance["dead_time_violations"][label] = int(
            np.count_nonzero(intervals < round(DEAD_TIME_NS*1000))
        ) if intervals.size else None
    save_json(folder / "measurements.json", balance)

    # 1 second bins, both detectors on the same time axis
    edges = np.r_[np.arange(0., seconds, 1.), seconds]
    time_bins = (edges[:-1]+edges[1:])/2
    hist_a = np.histogram((a-limits[0])*1e-12, bins=edges)[0]/np.diff(edges)
    hist_b = np.histogram((b-limits[0])*1e-12, bins=edges)[0]/np.diff(edges)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time_bins,hist_a,label=f"SPAD A (ch{CH_A})")
    ax.plot(time_bins,hist_b,label=f"SPAD B (ch{CH_B})")
    ax.set(xlabel="Elapsed time (s)",ylabel="Counts/second",title="Photon count-rate stability")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(folder / "count_rates.png",dpi=160)
    if SHOW_FIGURES: plt.show()
    plt.close(fig)

    # Two inter-arrival distributions (screening for dead-time anomalies)
    fig, ax = plt.subplots(figsize=(8,4))
    for label, ts in (("A",a),("B",b)):
        if len(ts)>2:
            dt = np.diff(ts)*1e-3   # ps -> ns
            if np.max(dt)>np.min(dt)>0:
                edges2 = np.geomspace(float(np.min(dt)),float(np.max(dt))*1.001,85)
                density,_ = np.histogram(dt,bins=edges2,density=True)
                centers = np.sqrt(edges2[:-1]*edges2[1:])
                ax.plot(centers,density,label=f"SPAD {label}")
    ax.axvline(DEAD_TIME_NS,ls="--",color="black",label="Nominal dead time")
    ax.set(xscale="log",yscale="log",xlabel="Within-channel interval (ns)",
           ylabel="Probability density (1/ns)",title="Inter-arrival distributions")
    ax.legend(); fig.tight_layout()
    fig.savefig(folder / "interarrival.png",dpi=160)
    if SHOW_FIGURES: plt.show()
    plt.close(fig)

    print("\n--- PHOTON MEASUREMENTS (same time window) ---")
    print(f"Exposure:  {seconds:.4f} seconds")
    print(f"SPAD A:    {na:,} counts | {na/seconds:,.1f} cps")
    print(f"SPAD B:    {nb:,} counts | {nb/seconds:,.1f} cps")
    print(f"A:B split: {balance['A_percent']:.2f}% : {balance['B_percent']:.2f}%")
    print(f"A/B ratio: {balance['A_over_B']:.4f}" if nb else "A/B ratio: N/A (B=0)")
    print("Dead-time violations (A/B):",balance["dead_time_violations"])
    print("Note: detector count split is not calibrated beamsplitter transmission;"
          " efficiencies/alignment also matter.")
    if nb < 1000:
        print("WARNING: B recorded few photons; check illumination, cabling and trigger.")
    if balance["A_above_warning_cps"] or balance["B_above_warning_cps"]:
        print("WARNING: One or both detectors exceed the proposed 100 kcps ceiling")
    return balance


def von_neumann_intervals(ts):
    """Adjacent NON-overlapping inter-arrival comparisons (not photon routing)."""
    dif = np.diff(ts)
    pairs = dif[:(len(dif)//2)*2].reshape(-1,2)
    return (pairs[:,0]>pairs[:,1])[pairs[:,0]!=pairs[:,1]].astype(np.uint8)


def toeplitz_hash(x,m):
    """True Toeplitz GF(2) product by FFT convolution; returns bits and random seed."""
    n=len(x)
    if not 0<m<=n: raise ValueError("Output bits m must satisfy 0 < m <= n")
    seed=np.unpackbits(np.frombuffer(secrets.token_bytes((n+m+6)//8),dtype=np.uint8))[:n+m-1]
    length=fft.next_fast_len(n+m-1)
    convolution=fft.irfft(fft.rfft(seed.astype(float),n=length)*
                           fft.rfft(x.astype(float),n=length),n=length)[n-1:n-1+m]
    rounded=np.rint(convolution)
    if np.max(np.abs(convolution-rounded))>0.25:
        raise ArithmeticError("FFT precision insufficient: reject hashed output")
    return (rounded.astype(np.int64)&1).astype(np.uint8),seed


def analyze(folder):
    """Offline analysis; refused for incomplete/lossy hardware acquisitions."""
    folder=Path(folder)
    info=json.loads((folder/"acquisition.json").read_text(encoding="utf-8"))
    if not info["valid"]:
        print("Extraction blocked: run has status",info["status"],"; raw files remain available")
        return
    raw={ch:np.load(folder/f"raw_ch{ch}.npy",allow_pickle=False)
         for ch in (CH_A,CH_B)}
    a,b,sec,_=common_window(raw,info["target_A"])
    if sec<=0: raise ValueError("Invalid shared time window")
    results={}
    print("\n--- RESEARCH-ONLY BIT STATISTICS (NOT SECURE KEY VALIDATION) ---")
    for label,ts in (("A",a),("B",b)):
        if len(ts)<4:
            print(f"SPAD {label}: too few events")
            continue
        bits=von_neumann_intervals(ts)
        n=len(bits)
        if not n:
            print(f"SPAD {label}: no surviving comparison bits")
            continue
        p1=float(np.mean(bits)); p0=1-p1
        h=-math.log2(max(p0,p1))       # EMPIRICAL marginal min-entropy only
        saved=write_bits(folder/f"interval_bits_{label}.bin",bits)
        results[label]={"bits_generated":n,"bits_exported":saved,
                        "p0":p0,"p1":p1,"empirical_marginal_hmin":h,
                        "yield_bits_per_detection":n/len(ts)}
        print(f"SPAD {label}: {n:,} bits | P(0)={p0:.5f}, P(1)={p1:.5f} | "
              f"empirical Hmin={h:.5f} bits/bit | yield={n/len(ts):.4f}")
        if label=="A" and TOEPLITZ_DEMO:
            # Empirical bias is NOT a justified entropy bound; this length is
            # illustrative only and must NOT be treated as cryptographically safe.
            m=max(0,min(n,math.floor(n*h-128)))
            if m>0:
                y,seed=toeplitz_hash(bits,m)
                results[label]["toeplitz_demo_bits_exported"]=write_bits(
                    folder/"toeplitz_RESEARCH_ONLY.bin",y)
                np.packbits(seed).tofile(folder/"toeplitz_RESEARCH_seed.bin")
                results[label]["toeplitz_seed_length_bits"]=len(seed)
                print(f"SPAD A: Toeplitz GF(2) demo -> {m:,} bits (research only)")
    results["warning"]="Observed bias is NOT a source min-entropy bound; this is NOT a secure QRNG."
    save_json(folder/"bit_statistics.json",results)


def run_acquisition(target, timeout):
    SAVE_ROOT.mkdir(parents=True,exist_ok=True)
    folder=SAVE_ROOT/stamp()
    folder.mkdir()
    print("Recording BOTH SPAD inputs; saving to",folder.resolve())
    raw,info=collect(target,timeout)
    for ch,ts in raw.items():
        np.save(folder/f"raw_ch{ch}.npy",ts)  # archive even incomplete/lossy runs
    save_json(folder/"acquisition.json",info)
    print("Acquisition status:",info["status"], "|",info["error"] or "no reported errors")
    characterize(raw,info,folder)
    if info["valid"]:
        analyze(folder)
    else:
        print("Raw timestamps saved. Bit extraction blocked until a valid acquisition.")
    print("Results:",folder.resolve())


def selftest():
    from scipy.linalg import toeplitz
    rng=np.random.default_rng(2026)
    for _ in range(8):
        x=rng.integers(0,2,47,dtype=np.uint8)
        # The same random seed cannot be injected into the public hash API,
        # so check the actual GF(2) output using the returned seed.
        y,s=toeplitz_hash(x,23)
        dense=toeplitz(s[46:46+23],s[46::-1])
        assert np.array_equal(y,(dense@x)%2)
    # Deterministic comparator examples: intervals 5,3 -> 1; 2,4 -> 0
    assert np.array_equal(von_neumann_intervals(np.array([0,5,8,10,14])),
                          np.array([1,0],dtype=np.uint8))
    print("Offline self-test passed: Toeplitz GF(2) and interval comparison")



def _find_saved_runs():
    """List runs with an acquisition record, most recent first."""
    if not SAVE_ROOT.exists():
        return []
    return sorted((p for p in SAVE_ROOT.iterdir()
                   if p.is_dir() and (p / "acquisition.json").is_file()),
                  key=lambda p: p.name, reverse=True)


def choose_saved_run():
    """Select a previous experiment for offline analysis; no hardware needed."""
    if ANALYZE_FOLDER is not None:
        return Path(ANALYZE_FOLDER)
    runs = _find_saved_runs()
    if not runs:
        print("No saved acquisition runs found in", SAVE_ROOT.resolve())
        return None
    print("\nSaved experiment folders (newest first):")
    for i, folder in enumerate(runs[:8], start=1):
        try:
            info = json.loads((folder / "acquisition.json").read_text(encoding="utf-8"))
            status = info.get("status", "unknown")
        except (ValueError, OSError):
            status = "metadata unreadable"
        print(f"  {i}. {folder.name}  [{status}]")
    response = input("Choose 1-8 (Enter = latest), or type a folder path: ").strip()
    if not response:
        return runs[0]
    if response.isdigit():
        index = int(response)
        if 1 <= index <= min(len(runs), 8):
            return runs[index - 1]
        print("No run with that number.")
        return None
    path = Path(response).expanduser()
    if not (path / "acquisition.json").is_file():
        print("Cannot find acquisition.json inside", path)
        return None
    return path


def analyze_saved_run(folder=None):
    """Show optics measurements/figures plus exploratory QRNG results from disk."""
    if folder is None:
        folder = choose_saved_run()
    if folder is None:
        return
    folder = Path(folder)
    info = json.loads((folder / "acquisition.json").read_text(encoding="utf-8"))
    if info.get("channels") != [CH_A, CH_B]:
        print("Saved channel wiring differs from current CH_A/CH_B settings.")
        print("Recorded:", info.get("channels"), "Current:", [CH_A, CH_B])
        print("Update the channel settings to match the saved experiment.")
        return
    raw = {ch: np.load(folder / f"raw_ch{ch}.npy", allow_pickle=False)
           for ch in (CH_A, CH_B)}
    print("\nRe-analyzing", folder.resolve())
    characterize(raw, info, folder)
    analyze(folder)     # blocks bit extraction on incomplete or lossy runs
    print("Results saved to", folder.resolve())


def _run_mode(mode):
    if mode == "selftest":
        selftest()
    elif mode == "short":
        run_acquisition(SHORT_A_EVENTS, SHORT_TIMEOUT_S)
    elif mode == "full":
        run_acquisition(FULL_A_EVENTS, FULL_TIMEOUT_S)
    elif mode == "analyze":
        analyze_saved_run()
    else:
        raise ValueError("Unknown mode: " + str(mode))


def experiment_menu():
    """Small text menu: an acquisition helper, not a laser-control program."""
    while True:
        print("\n" + "=" * 53)
        print("   405 nm LASER PHOTON-COUNTING EXPERIMENT")
        print("=" * 53)
        print("  1. Self-test     Check processing math (no hardware)")
        print(f"  2. Short run     {SHORT_A_EVENTS:,} A events; record both SPADs")
        print(f"  3. Full run      {FULL_A_EVENTS:,} A events; record both SPADs")
        print("  4. Analyze data  Open a previously saved experiment")
        print("  0. Exit")
        print("-" * 53)
        print(f"Inputs: SPAD A = tagger ch{CH_A}; B = ch{CH_B}")
        print("Laser current, temperature and alignment are set manually.")
        choice = input("Select 0-4: ").strip()
        if choice in {"0", "q", "quit", "exit"}:
            print("Experiment menu closed.")
            return
        mode = {"1":"selftest", "2":"short", "3":"full", "4":"analyze"}.get(choice)
        if mode is None:
            print("Choose one of 0, 1, 2, 3 or 4.")
            continue
        try:
            _run_mode(mode)
        except KeyboardInterrupt:
            print("Interrupted. Returning to the experiment menu.")
        except (OSError, ValueError, RuntimeError, ArithmeticError) as exc:
            print(f"Experiment could not finish: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    if MODE == "menu":
        experiment_menu()
    else:
        _run_mode(MODE)
