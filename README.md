# MagSurrogate

Machine-Learning Surrogate Models for Temperature- and Size-Dependent Magnetic Constitutive Relations

This repository provides the dataset, source code, benchmarking workflow, and physics-analysis tools.
---

## Overview

MagSurrogate is a physics-aware machine-learning framework for constructing surrogate models of finite-temperature magnetic constitutive relations of FePt nanograins.

The repository includes:

- Polynomial Regression
- Cubic Spline Interpolation
- Random Forest (RF)
- Gradient Boosting (GB)
- Extreme Gradient Boosting (XGB)
- K-Nearest Neighbor (KNN)

The framework supports:

- Leave-One-Diameter-Out Cross Validation (LODO-CV)
- Unseen geometry generalization analysis
- Curie temperature extraction
- Finite-size scaling analysis
- Universal finite-size scaling collapse

---

## Dataset

The dataset contains magnetic properties generated using atomistic spin simulations.

### Input Variables

| Variable | Description | Unit |
|-----------|------------|------|
| Temperature | Temperature | K |
| Diameter | Grain diameter | nm |

### Output Variables

| Variable | Description |
|-----------|------------|
| Mx | Equilibrium magnetization |
| Chi_longitudinal | Longitudinal susceptibility |
| Chi_trans | Transverse susceptibility |

Dataset file:

```text
fept_dataset.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/komkritc/MagSurrogate.git
cd MagSurrogate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the complete workflow:

```bash
python run_all.py
```

---

## License

This project is released under the MIT License.

---

## Author

Komkrit Chooruang

Department of Electrical Engineering, Faculty of Engineering, 
Nakhon Phanom University, Nakhon Phanom 48000, Thailand.
