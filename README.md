**Phase 1: DM H-TFET Biosensor TCAD Modeling & Monte Carlo Simulation Framework**

*Overview*

This repository contains the Phase 1 codebase and TCAD simulation scripts for modeling a Gate-Overlapped Heterojunction Tunnel Field-Effect Transistor (DM H-TFET) dielectric-modulated biosensor. The primary objective of Phase 1 is to capture the nonlinear physical behavior of nanoscale biosensors under non-uniform biomolecule filling distributions and execute large-scale Monte Carlo statistical TCAD simulations to create a dataset for variability analysis.

*Device Architecture & Specifications*

The simulated device utilizes a Ge–Si heterojunction with a line-tunneling gate-overlapped source region to maximize band-to-band tunneling (BTBT) probability and electrostatic control.      


                  +-------------------------------------------------+
                  |                   Gate Metal                    |
                  +-------------------------------------------------+
                  |            High-k Dielectric (HfO2)             |
                  +------------------------+------------------------+
                  |       Nanocavity       |   Interfacial Oxide    |
                  | (Biomolecules 90% Fill)|         (SiO2)         |
+-----------------+------------------------+------------------------+-----------------+
|     Source      |                        Channel                  |      Drain      |
|     (p+ Ge)     |                        (p- Si)                  |     (n+ Si)     |
+-----------------+-------------------------------------------------+-----------------+


*Key Geometrical & Doping Parameters*

Source Region: Heavily doped p-type Germanium ($1 \times 10^{20} \text{ cm}^{-3}$), length = 200 nm

Channel Region: Lightly doped p-type Silicon ($1 \times 10^{15} \text{ cm}^{-3}$), length = 200 nm

Drain Region: Heavily doped n-type Silicon ($1 \times 10^{20} \text{ cm}^{-3}$), length = 100 nm

Gate Overlap: 100 nm overlap onto the source region to promote line tunneling

Gate Stack: 10 nm $\text{HfO}_2$ high-k dielectric layer over a 1 nm $\text{SiO}_2$ interfacial oxide layer

Gate Work Function: Fixed at 4.2 eV

Nanocavity Dimensions: Nominal length = 150 nm, nominal height = 10 nm, nominal body thickness = 10 nm

*Physical Transport Models*
Silvaco ATLAS TCAD is configured with physics models to capture nanoscale tunneling dynamics:Non-local Band-to-Band Tunneling (BTBT): Accounts for line and point tunneling across the Ge–Si heterojunctionShockley-Read-Hall (SRH): Accounts for carrier recombinationMobility Models: Concentration-dependent (CONMOB) and field-dependent (FLDMOB) mobility modelsCarrier Statistics: Fermi-Dirac statistics for degenerate doping regions.

*Biomolecule Filling Profiles*
To overcome the assumption of ideal uniform cavity coverage, four realistic biomolecule filling profiles are modeled under a 90% cavity-filled condition:Concave Distribution: Biomolecule concentration is higher near cavity edges than in the middleConvex Distribution: Peak fill density located at the center of the cavityRamp-Up Distribution: Biomolecule density linearly increases from source to drainRamp-Down Distribution: Biomolecule density linearly decreases from source to drain.

*Data Generation Pipeline*
Monte Carlo statistical TCAD simulations vary cavity dimensions and silicon body thickness:Total Generated Samples: $\sim 50,000$ to $70,000$ simulation samplesVaried Parameters: Cavity length, cavity height/width, and silicon body thicknessExtracted Primary Outputs:ON-Current Sensitivity ($I_D$ Sensitivity)Threshold Voltage Sensitivity ($V_{th}$ Sensitivity)

*Repository Structure*
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
*Quickstart Guide*
*Prerequisites*
Silvaco ATLAS TCAD (v5.20.0.R or higher)Python 3.8+ with pandas, numpy, and subprocess

*Execution Steps*
Run Monte Carlo Batch Simulations:python monte_carlo_runner/batch_generator.py --samples 50000 --profile all
Extract & Compile Sensitivities:python monte_carlo_runner/extract_sensitivity.py --input_dir data/raw_tcad_outputs/ --output data/compiled_dataset.csv
