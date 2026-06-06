# Fashion-MNIST: Shallow vs Deep Networks

> **Course:** SST Neural Network & Intro to Computer Vision (ML III)  
> **Assignment:** Section 2 — 17 Shallow vs. Deep Networks  
> **Dataset:** Fashion-MNIST  
> **Framework:** PyTorch  

**Team: Group 8**
- Ujjwal Jain (10173)  
- Pratham Onkar (10136)  
- Aditya Kumar Rai (10178)  
- Dhairya Motta (10202)  
- Arman Barbhuiya (10196)  
- Lakshay Jagga (10398)  
- Piyush Kumar Gupta (10332)  
- Iyad Farooq (10116)  

---

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Does increasing depth improve performance under a fixed parameter budget? |
| RQ2 | How does depth affect gradient flow? |
| RQ3 | What role does activation choice play? |
| RQ4 | Can BatchNorm restore trainability? |

---

## Project Structure

```
fashion-mnist-depth-study/
│
├── notebooks/
│   └── Fashion_MNIST_Depth_Study.ipynb   ← Primary deliverable (Colab-ready)
│
├── src/
│   ├── __init__.py
│   ├── models.py     — ConfigurableMLP
│   ├── train.py      — Training loop + ExperimentRunner
│   ├── utils.py      — CONFIG, seeding, data loading, CSV I/O
│   └── plots.py      — All 7 required figures
│
├── results/
│   ├── results.csv           ← Epoch-level metrics (auto-generated)
│   ├── summary.csv           ← Per-run summary (auto-generated)
│   └── figures/              ← All PNG figures (auto-generated)
│       ├── fig1_val_accuracy.png
│       ├── fig2_val_loss.png
│       ├── fig3_test_accuracy_bar.png
│       ├── fig4a_gradient_by_layer.png
│       ├── fig4b_gradient_over_epochs.png
│       ├── fig5_activation_heatmap.png
│       └── fig6_batchnorm_recovery.png
│
├── report/
│   └── report.md             ← Final Submission Report
│
├── requirements.txt
├── PHASE_3_VALIDATION_CHECKLIST.md
└── README.md
```

---

## Setup

### Option A — Google Colab (Recommended)

1. Upload the project zip to Google Drive (or clone to Colab storage):
   ```bash
   !git clone https://github.com/Ujjwaljain16/fashion-mnist-depth-study.git
   %cd fashion-mnist-depth-study
   ```
2. Open `notebooks/Fashion_MNIST_Depth_Study.ipynb` in Colab.
3. Set `Runtime → Change runtime type → T4 GPU` for best performance.
4. Run all cells top-to-bottom (`Runtime → Run all`).

### Option B — Local Environment

```bash
# 1. Clone or unzip project
git clone https://github.com/Ujjwaljain16/fashion-mnist-depth-study.git
cd fashion-mnist-depth-study

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook notebooks/Fashion_MNIST_Depth_Study.ipynb
```

> **Windows note:** The notebook automatically detects Windows and sets `num_workers=0`
> to avoid Jupyter multiprocessing errors.

---

## Reproducing Results

### Full Run (submission quality)

Open the notebook and confirm:
```python
FAST_MODE   = False   # 3 seeds × 50 epochs
AUTO_RESUME = True    # skips already-completed runs
```
Then `Run all cells`. Expected runtime: ~8–15 min on Colab T4 GPU.

### Quick Test Run

```python
FAST_MODE   = True    # 1 seed × 10 epochs (~2 min on CPU)
AUTO_RESUME = True
```

### Re-generating Figures Only

If results CSVs already exist, skip to Section 7 in the notebook and run
the figure cells. Results are loaded from CSV, not from memory.

---

## Experiment Summary

| Experiment | Models | Purpose |
|-----------|--------|---------|
| 1 | 2L, 4L, 8L ReLU | Depth comparison under equal parameter budget |
| 2 | 2L/4L/8L × ReLU/Sigmoid | Activation effect on gradient flow |
| 3 | 8L Sigmoid ± BatchNorm | BatchNorm as a trainability fix |

---

## Architecture: Equal Parameter Budget

Widths are adjusted so all models have approximately the same number of parameters (~500K):

| Depth | Width | Actual Parameters |
|-------|-------|------------------|
| 2L    | 413   | 499,327          |
| 4L    | 296   | 499,066          |
| 8L    | 215   | 496,015          |
| 8L+BN | 215   | 499,455          |

Actual parameter counts are logged automatically during training.

---

## Key Files

| File | Purpose |
|------|---------|
| [`src/models.py`](src/models.py) | `ConfigurableMLP` class |
| [`src/train.py`](src/train.py) | Training loop + experiment orchestration |
| [`src/utils.py`](src/utils.py) | `CONFIG` dict (single source of truth) |
| [`src/plots.py`](src/plots.py) | All figure generation functions |
| [`report/report.md`](report/report.md) | Final submission report with empirical data |

---

## Reproducibility

Three random seeds are used: **42, 123, 7**.

All results are reported as **mean ± std** across seeds.

Seed control:
- `random.seed`, `numpy.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `cudnn.deterministic = True`, `cudnn.benchmark = False`

The train/validation split (54K/6K) uses a fixed generator seed of 42 across all runs.

---

## Team Contributions

This project was a collaborative effort by all members of Group 8:
Ujjwal Jain, Pratham Onkar, Aditya Kumar Rai, Dhairya Motta, Arman Barbhuiya, Lakshay Jagga, Piyush Kumar Gupta, and Iyad Farooq. 

All members contributed equally to the experimental design, code implementation, execution, data analysis, and final report writing.
