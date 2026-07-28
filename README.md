# Data-Validation-Framework
# Phase 1: DM H-TFET Biosensor TCAD Modeling & Monte Carlo Simulation Framework

## Overview
This repository contains the Phase 1 codebase and TCAD simulation scripts for modeling a **Gate-Overlapped Heterojunction Tunnel Field-Effect Transistor (DM H-TFET)** dielectric-modulated biosensor[cite: 1, 2]. The primary objective of Phase 1 is to capture the nonlinear physical behavior of nanoscale biosensors under non-uniform biomolecule filling distributions and execute large-scale Monte Carlo statistical TCAD simulations to create a dataset for variability analysis[cite: 1, 2].

---

## Device Architecture & Specifications

The simulated device utilizes a **Ge–Si heterojunction** with a line-tunneling gate-overlapped source region to maximize band-to-band tunneling (BTBT) probability and electrostatic control[cite: 1, 2].
            +-------------------------------------------------+
            |                  Gate Metal                     |
            +-------------------------------------------------+
            |             High-k Dielectric (HfO2)            |
            +------------------------+------------------------+
            |     Nanocavity         |   Interfacial Oxide    |
            | (Biomolecules 90% Fill)|        (SiO2)          |
+-----------+------------------------+------------------------+
|   Source  |                       Channel                   | Drain
|   (p+ Ge) |                      (p- Si)                    | (n+ Si)
+-----------+-------------------------------------------------+---------+
### Key Geometrical & Doping Parameters
* **Source Region**: Heavily doped p-type Germanium ($1 \times 10^{20} \text{ cm}^{-3}$), length = 200 nm[cite: 2].
* **Channel Region**: Lightly doped p-type Silicon ($1 \times 10^{15} \text{ cm}^{-3}$), length = 200 nm[cite: 2].
* **Drain Region**: Heavily doped n-type Silicon ($1 \times 10^{20} \text{ cm}^{-3}$), length = 100 nm[cite: 2].
* **Gate Overlap**: 100 nm overlap onto the source region to promote line tunneling[cite: 2].
* **Gate Stack**: 10 nm $\text{HfO}_2$ high-k dielectric layer over a 1 nm $\text{SiO}_2$ interfacial oxide layer[cite: 2].
* **Gate Work Function**: Fixed at 4.2 eV[cite: 2].
* **Nanocavity Dimensions**: Nominal length = 150 nm, nominal height = 10 nm, nominal body thickness = 10 nm[cite: 2].

---

## Physical Transport Models

Silvaco ATLAS TCAD is configured with physics models to capture nanoscale tunneling dynamics[cite: 1, 2]:
* **Non-local Band-to-Band Tunneling (BTBT)**: Accounts for line and point tunneling across the Ge–Si heterojunction[cite: 2].
* **Shockley-Read-Hall (SRH)**: Accounts for carrier recombination[cite: 2].
* **Mobility Models**: Concentration-dependent (`CONMOB`) and field-dependent (`FLDMOB`) mobility models[cite: 2].
* **Carrier Statistics**: Fermi-Dirac statistics for degenerate doping regions[cite: 2].

---

## Biomolecule Filling Profiles

To overcome the assumption of ideal uniform cavity coverage, four realistic biomolecule filling profiles are modeled under a **90% cavity-filled condition**[cite: 1, 2]:

1. **Concave Distribution**: Biomolecule concentration is higher near cavity edges than in the middle[cite: 1, 2].
2. **Convex Distribution**: Peak fill density located at the center of the cavity[cite: 1, 2].
3. **Ramp-Up Distribution**: Biomolecule density linearly increases from source to drain[cite: 1, 2].
4. **Ramp-Down Distribution**: Biomolecule density linearly decreases from source to drain[cite: 1, 2].

---

## Data Generation Pipeline

Monte Carlo statistical TCAD simulations vary cavity dimensions and silicon body thickness[cite: 1, 2]:
* **Total Generated Samples**: $\sim 50,000$ to $70,000$ simulation samples[cite: 1, 2].
* **Varied Parameters**: Cavity length, cavity height/width, and silicon body thickness[cite: 1, 2].
* **Extracted Primary Outputs**:
  * **ON-Current Sensitivity ($I_D$ Sensitivity)**[cite: 1, 2]
  * **Threshold Voltage Sensitivity ($V_{th}$ Sensitivity)**[cite: 1, 2]

---

## Repository Structure

```text
├── tcad_scripts/
│   ├── htfet_base_structure.in      # Silvaco ATLAS deck for nominal device setup
│   ├── profile_concave.in           # Non-uniform dielectric mesh script
│   ├── profile_convex.in            # Convex density modulation script
│   ├── profile_rampup.in            # Ramp-up profile script
│   └── profile_rampdown.in          # Ramp-down profile script
├── monte_carlo_runner/
│   ├── batch_generator.py           # Python script to parameterize TCAD input decks
│   └── extract_sensitivity.py       # Extract ID and Vth sensitivities from log files
├── data/
│   ├── raw_tcad_outputs/            # Generated TCAD output logs
│   └── compiled_dataset.csv         # Structured dataset containing ~50,000 Monte Carlo samples
└── README.md
