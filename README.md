# MacGyver Optics: 405nm Diode Laser QRNG

**Automated DAQ & Extraction Pipeline for a Time-Correlated Quantum Random Number Generator**

## Project Overview
This repository contains the data acquisition, entropy extraction, and live bias monitoring pipeline for an ultra-low-cost (under $1,000), room-temperature Quantum True Random Number Generator (TRNG)[cite: 1]. 

Standard optics approaches often view laser diode threshold noise and thermal drift as physical flaws[cite: 1]. This project embraces the "MacGyver Optics" philosophy by turning those imperfections into a cryptographic asset[cite: 1]. We quantify how quantum spontaneous emission noise behaves under thermal variations (20.0°C to 35.0°C) and SPAD detector dead time constraints[cite: 1].

## Hardware Architecture
* **Optical Source:** Thorlabs 405 nm Diode Laser[cite: 1].
* **Thermal Control:** Thorlabs TEC Controller sweeping from 20.0°C to 35.0°C in 2.5°C steps[cite: 1].
* **Biasing:** Precision DC Current Driver injecting current at the threshold knee ($I \approx 1.02 \times I_{\text{th}}$)[cite: 1].
* **Detectors:** Dual SPAD Units (Channel 1 & Channel 2) with active quenching (~22 ns dead time)[cite: 1]. Count rates are strictly bounded below 100 kcps using Neutral Density (ND) filters to eliminate afterpulse memory effects[cite: 1].
* **Time-Tagging:** Swabian Time Tagger Ultra (Serial: 1948000SQ6) logging photon arrival times with picosecond resolution[cite: 1].

## Software Pipeline
The primary script, `trng_ase_acquisition.py`, automates the single-channel timestamp acquisition and key generation phase[cite: 1]. 
1. **Timestamping:** Captures $2 \times 10^6$ arrival timestamps[cite: 1].
2. **Interval Calculation:** Computes inter-arrival intervals $\Delta t_i = t_i - t_{i-1}$[cite: 1].
3. **Extraction:** Executes vectorized Von Neumann pair extraction where $\Delta t_{2k} > \Delta t_{2k+1} \longrightarrow 1$ and $\Delta t_{2k} < \Delta t_{2k+1} \longrightarrow 0$[cite: 1].
4. **Validation:** Checks for dead-time violations below 22 ns and calculates Min-Entropy ($H_\infty$) before passing keys to the NIST SP 800-22 suite[cite: 1].
