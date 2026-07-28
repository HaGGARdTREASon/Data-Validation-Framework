**Phase 1: DM H-TFET Biosensor TCAD Modeling & Monte Carlo Simulation Framework**

*Overview*

This repository contains the Phase 1 codebase and TCAD simulation scripts for modeling a Gate-Overlapped Heterojunction Tunnel Field-Effect Transistor (DM H-TFET) dielectric-modulated biosensor. The primary objective of Phase 1 is to capture the nonlinear physical behavior of nanoscale biosensors under non-uniform biomolecule filling distributions and execute large-scale Monte Carlo statistical TCAD simulations to create a dataset for variability analysis.

*Device Architecture & Specifications*

The simulated device utilizes a Ge–Si heterojunction with a line-tunneling gate-overlapped source region to maximize band-to-band tunneling (BTBT) probability and electrostatic control.      


<img width="727" height="290" alt="image" src="https://github.com/user-attachments/assets/3b1823b5-ede0-4230-b9cd-82fe545e7706" />



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

<img width="806" height="325" alt="image" src="https://github.com/user-attachments/assets/f024b331-2897-47f6-826b-b4e3949312bf" />


*Quickstart Guide*

*Prerequisites*

Silvaco ATLAS TCAD (v5.20.0.R or higher)Python 3.8+ with pandas, numpy, and subprocess

*Execution Steps*
Run Monte Carlo Batch Simulations:python monte_carlo_runner/batch_generator.py --samples 50000 --profile all
Extract & Compile Sensitivities:python monte_carlo_runner/extract_sensitivity.py --input_dir data/raw_tcad_outputs/ --output data/compiled_dataset.csv


**Phase 2: Machine Learning Predictive Framework for Biosensor Performance & Variability Analysis**


*Overview*


Phase 2 implements a data-driven predictive modeling framework using supervised machine learning regression to model the nonlinear dependencies of biosensor sensitivity on structural variations across multiple biomolecule filling profiles.

By substituting exhaustive Technology Computer-Aided Design (TCAD) simulations with trained regression estimators, the framework reduces computational overhead by $\sim 99\%$ while maintaining prediction accuracy above $99.8\%$.

*Machine Learning Architecture*

The pipeline evaluates 8 individual regression tasks (4 biomolecule profiles $\times$ 2 target metrics):

Input Features: Structural and cavity geometric parameters (e.g., Cavity Width, Cavity Length, Silicon Body Thickness).

Target Outputs:

$I_D$ Sensitivity (ON-current sensitivity)

$V_{th}$ Sensitivity (Threshold voltage sensitivity)

Evaluated Algorithms:

Random Forest Regression (RFR) (Primary Architecture)

Gradient Boosting Regression (GBR)

Extreme Gradient Boosting Regression (XGBoost / XGBR)

K-Nearest Neighbors Regression (KNN)

Support Vector Regression (SVR)

Preprocessing & Scaling Pipeline

To avoid data leakage, data preprocessing follows a strict sequential pipeline:

Data Cleaning: Null-value filtering across input feature and sensitivity target columns.

Min-Max Normalization: Target sensitivity values are scaled to the range $[0, 1]$:


$$\hat{y} = \frac{y - y_{\min}}{y_{\max} - y_{\min}}$$

Dataset Partitioning: Evaluated across multi-ratio split configurations ($50:50$ up to $90:10$) with $80:20$ serving as the standard working benchmark.

Inverse Transformation: Predicted outputs are scaled back to original physical units for domain-specific accuracy computation:


$$y = \hat{y} \cdot (y_{\max} - y_{\min}) + y_{\min}$$

<img width="800" height="385" alt="image" src="https://github.com/user-attachments/assets/31500d96-429b-4f4b-9170-827a2e8873fe" />



*Formal Evaluation Metrics*


Model performance is evaluated using standard regression metrics on normalized targets:

Mean Squared Error (MSE): 

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Root Mean Squared Error (RMSE): 

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Mean Absolute Error (MAE): 

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \vert{}y_i - \hat{y}_i\vert{}$$

Coefficient of Determination ($R^2$): 

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Physical Accuracy Metric: 

$$\text{Accuracy (\%)} = \left( 1 - \frac{1}{n} \sum_{i=1}^{n} \frac{\vert{}y_i - \hat{y}_i\vert{}}{y_i} \right) \times 100$$


*Repository Structure*

<img width="735" height="277" alt="image" src="https://github.com/user-attachments/assets/4d490673-290b-41e3-9a13-1e463e44a6a3" />



**Setup & Quickstart**

**Prerequisites & Dependencies**

*Ensure Python 3.8+ is installed along with the required machine learning packages:*

scikit-learn>=1.0.0
xgboost>=1.5.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0


**Installation & Execution**

*Install dependencies:*
pip install -r requirements.txt


*Train and evaluate the models across target profiles:*
python src/models.py --data_path ../data/compiled_dataset.csv --split 0.8 --model rfr


*Generate tolerance-based spatial scatter plots:*
python src/visualization.py --profile convex --target Id_sensitivity --tolerance 0.08
