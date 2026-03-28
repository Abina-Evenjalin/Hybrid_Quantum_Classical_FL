# Quantum Federated Learning Platform
## Hybrid Quantum-Classical Federated Learning Research Platform

---

**Author:** R.A. Evenjalin — S16420
**Supervisor:** Dr. U.P. Liyanage
**Degree:** BSc (Hons) in Information Technology and Management
**Institution:** University of Colombo
**Version:** 2.0.0

---

## Download Executable

The standalone Windows executable is too large for GitHub hosting.
Download the full application here:

**[Download QuantumFL_App.zip (Google Drive)](https://drive.google.com/file/d/1IVQGCfUBojj_J9hSAJ4ZI6iRrKPPRgU0/view?usp=sharing)**

> Unzip and double-click `quantum_fl_app.exe` to launch. No Python or Anaconda required.

---

## Repository Contents

```
Hybrid_Quantum_Classical_FL/
├── source_code/                 <- Full Python source code
│   ├── quantum_fl_app.py        <- Main application script (3000+ lines)
│   └── resources/
│       └── hfl_icon.ico         <- Application icon
├── test_datasets/               <- Sample datasets for testing
│   ├── AND.xlsx                 <- Numerical: AND logic gate dataset
│   ├── Circles.xlsx             <- Numerical: Circles classification dataset
│   ├── Moons.xlsx               <- Numerical: Moons classification dataset
│   ├── Non_IID_XOR.xlsx         <- Numerical: Non-IID XOR dataset
│   ├── OR.xlsx                  <- Numerical: OR logic gate dataset
│   ├── XOR.xlsx                 <- Numerical: XOR logic gate dataset
│   ├── medical_diagnosis.csv    <- Numerical: Medical diagnosis dataset
│   ├── product_reviews.csv      <- Text: Product reviews sentiment dataset
│   └── shapes/                  <- Image: Geometric shapes dataset
│       └── geometric_shapes/
│           ├── circles/         <- Circle images
│           ├── squares/         <- Square images
│           └── triangles/       <- Triangle images
└── README.md                    <- This file
```

---

## How to Run the Application

### Requirements
- Windows 10 or Windows 11 (64-bit)
- No Python or Anaconda installation required
- The entire `QuantumFL_App\` folder must remain intact after unzipping

### Steps to Launch
1. Download `QuantumFL_App.zip` from the Google Drive link above
2. Unzip the folder
3. Open the unzipped `QuantumFL_App\` folder
4. Double-click `quantum_fl_app.exe`
5. Wait up to **60 seconds** on first launch (unpacking bundled libraries)
6. The application window will appear automatically

> **Note:** Do not move `quantum_fl_app.exe` out of its folder. It requires the `_internal\` folder alongside it to run.

---

## How to Use the Platform

The platform follows a 5-step workflow:

### Step 1 — Quantum Setup
- Select quantum backend: **Aer Simulator** (local, no setup needed) or **IBM Quantum Cloud** (requires API token)
- Test the quantum circuit to verify it is working
- Default: Aer Simulator (recommended for testing)

### Step 2 — Dataset Upload
- Click **Browse Files/Folders** or drag and drop your dataset
- Select data type when prompted:
  - **Numerical** — upload any `.xlsx` or `.csv` file from `test_datasets\`
  - **Image** — upload the `test_datasets\shapes\geometric_shapes\` folder
  - **Text** — upload `test_datasets\product_reviews.csv`
- The platform automatically detects, validates, and preprocesses the data

### Step 3 — Configuration
- AI-suggested parameters are auto-filled based on your dataset
- Adjust if needed:
  - Number of Qubits (2-10)
  - Number of Federated Clients (2-10)
  - Training Rounds (1-50)
  - Local Epochs (1-20)
  - Learning Rate
- Choose model type: **Quantum-Classical Hybrid** or **Classical Baseline**

### Step 4 — Training
- Click **Start Training**
- Monitor live accuracy, loss, and round progress
- Training runs in background — UI remains responsive

### Step 5 — Results
- Click **Generate Report** for full training summary
- Click **Plot Results** for visual charts
- Click **Export Data** to save results as CSV

---

## Test Datasets Guide

| Dataset | Type | Classes | Recommended For |
|---|---|---|---|
| AND.xlsx | Numerical | 2 | Quick testing, binary classification |
| OR.xlsx | Numerical | 2 | Quick testing, binary classification |
| XOR.xlsx | Numerical | 2 | Non-linear classification |
| Circles.xlsx | Numerical | 2 | Non-linear classification |
| Moons.xlsx | Numerical | 2 | Non-linear classification |
| Non_IID_XOR.xlsx | Numerical | 2 | Non-IID federated learning testing |
| medical_diagnosis.csv | Numerical | Multi | Real-world medical data |
| product_reviews.csv | Text | Multi | Sentiment analysis, NLP |
| geometric_shapes/ | Image | 3 | Image classification (circles/squares/triangles) |

> **Recommended for first run:** `AND.xlsx` — smallest dataset, fastest training

---

## Platform Features

- **Multi-Modal Support** — Numerical, Image, and Text data
- **Quantum Circuits** — Real parameterized variational quantum circuits via Qiskit
- **Federated Learning** — Privacy-preserving distributed training across multiple clients
- **Federated Averaging** — Weighted FedAvg aggregation algorithm
- **Hybrid Model** — Quantum-Classical neural network combination
- **Classical Baseline** — Pure neural network for quantum advantage comparison
- **CNN Feature Extraction** — ResNet-18 for image data (via PyTorch)
- **BERT Embeddings** — Sentence transformers for text data
- **IBM Quantum Cloud** — Optional real quantum hardware integration
- **Live Monitoring** — Real-time training metrics and progress
- **Export** — CSV export of training results

---

## Technical Stack

| Component | Technology |
|---|---|
| GUI Framework | PyQt5 |
| Quantum Computing | Qiskit + Qiskit-Aer |
| Deep Learning | PyTorch 2.2.2 |
| ML / Preprocessing | scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Text Embeddings | sentence-transformers 2.7.0 |
| Image Features | torchvision (ResNet-18) |
| Python Version | 3.11 |

---

## Running from Source Code

If you prefer to run from source directly:

```bash
# Create conda environment
conda create -n quantumfl_env python=3.11 -y
conda activate quantumfl_env

# Install dependencies
pip install PyQt5 numpy pandas matplotlib scikit-learn seaborn openpyxl
pip install torch==2.2.2 torchvision pillow
pip install qiskit qiskit-aer qiskit-ibm-runtime
pip install sentence-transformers==2.7.0 transformers==4.40.0

# Run the application
python source_code/quantum_fl_app.py
```

---

## Troubleshooting

**App takes long to open on first launch**
This is normal. PyTorch and other libraries unpack on first run. Wait up to 90 seconds.

**App crashes on launch**
Make sure the entire `QuantumFL_App\` folder is intact and `_internal\` folder is present next to the exe.

**Image data not loading**
Ensure folder structure is: `parent_folder\class_name\images.png` — the platform expects class subfolders.

**Text data not loading**
CSV must have a `text` column and a `label` column, or use folder of `.txt` files organized by class.

---

## Research Context

This platform was developed as part of a BSc (Hons) thesis investigating the potential of hybrid quantum-classical models in federated learning environments. The platform explores whether quantum-enhanced models can outperform classical baselines in distributed, privacy-preserving machine learning scenarios across multiple data modalities.

---

*Quantum Federated Learning Platform v2.0 — University of Colombo — 2026*
