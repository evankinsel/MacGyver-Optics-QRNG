# Thermal and Temporal Characterization of a Near-Threshold 405 nm Diode Laser in an HBT Setup

## Project Overview
This repository contains the data processing, metadata logging, and statistical characterization pipeline for an undergraduate benchtop study on a near-threshold 405 nm diode laser in a Hanbury Brown and Twiss (HBT) setup[cite: 1]. The project evaluates how laser diode operating conditions, mount temperature variations (20.0°C to 35.0°C), and SPAD detector response (dead time and afterpulsing) affect photon arrival timing statistics and pairwise random bit extraction[cite: 1].

> **Claim Boundary:** This is an undergraduate physical characterization experiment rather than a finished security product[cite: 1]. Statistical metrics (such as min-entropy estimates and selected NIST SP 800-22 diagnostic tests) serve as descriptive evaluation tools, not formal proofs of quantum origin or cryptographic deployment readiness[cite: 1].

## Hardware Architecture
* **Optical Source:** 405 nm diode laser mounted on a temperature-controlled stage with precision current driving[cite: 1].
* **Thermal Control:** TEC controller supporting documented thermal setpoints (e.g., sweeping from 20.0°C to 35.0°C in 2.5°C steps with thermal soak times)[cite: 1].
* **Interferometer Setup (HBT):** Hanbury Brown and Twiss arrangement utilizing a 50:50 beam splitter to route light onto dual SPAD channels for single-channel timing and two-detector cross-correlation measurements[cite: 1].
* **Attenuation & Shielding:** Neutral density (ND) filters setting single-detector count rates below a 100 kcps operating guardrail to prevent SPAD saturation and mitigate detector memory effects[cite: 1].
* **Detection:** Dual Single-Photon Avalanche Diodes (SPADs) with active quenching for photon counting and coincidence detection[cite: 1].
* **Time-Tagging:** Swabian Time Tagger Ultra recording photon arrival timestamps with picosecond resolution across both channels[cite: 1].
* **Power Diagnostics:** Power meter or monitor photodiode for L-I curve characterization (or documented count response vs. current if an optical power monitor is unavailable)[cite: 1].

## Research Phases & Workflow

| Phase | Description & Key Activities |
| :--- | :--- |
| **Phase 0: Feasibility & Baseline** | Verify current/temperature access, log laser-blocked dark count distributions, and collect room-temperature baselines below, near, and above threshold[cite: 1]. |
| **Phase 1: Operating Point Selection** | Map the L-I curve (or attenuated count response vs. current) around the knee to select a stable low-flux region[cite: 1]. |
| **Phase 2: Timing Acquisition** | Execute repeated timestamp acquisitions (~2×10⁶ events per run) across thermal setpoints (20.0°C to 35.0°C) and conduct multi-day stability repeats[cite: 1]. |
| **Phase 3: Analysis & Extraction** | Check rate stability, plot inter-arrival distributions, evaluate interval autocorrelation, and execute pairwise bit extraction[cite: 1]. |
| **Phase 4: HBT Coincidence Study** | Measure two-detector cross-correlation and normalized coincidence histograms $g^{(2)}(\tau)$ around zero delay[cite: 1]. |

## Software & Extraction Pipeline
The Python analysis pipeline handles timestamp ingestion, interval calculations, bit generation, and run tracking[cite: 1]:

1. **Timestamp Import & Filtering:** Ingests raw timestamp streams alongside structured CSV metadata (current, temperature, filters, duration, channel)[cite: 1].
2. **Interval Calculation:** Computes inter-arrival intervals $\Delta t_i = t_i - t_{i-1}$ and verifies short-delay limits against detector dead time[cite: 1].
3. **Pairwise Interval Extraction:** Evaluates sequential non-overlapping interval pairs ($\Delta t_{2k}$, $\Delta t_{2k+1}$)[cite: 1]:
   * If $\Delta t_{2k} > \Delta t_{2k+1} \longrightarrow 1$[cite: 1]
   * If $\Delta t_{2k} < \Delta t_{2k+1} \longrightarrow 0$[cite: 1]
   * Discards ties ($\Delta t_{2k} = \Delta t_{2k+1}$) and logs retained bit yield and residual bias[cite: 1].
4. **Diagnostics Suite:** Evaluates count rate drift over time, generates inter-arrival histograms, computes min-entropy ($H_\infty$), and executes selected NIST SP 800-22 tests as descriptive diagnostics[cite: 1].
