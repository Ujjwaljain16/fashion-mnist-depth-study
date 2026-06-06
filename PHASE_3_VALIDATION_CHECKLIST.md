# Fashion-MNIST: Shallow vs Deep Networks
### Phase 3 Validation Checklist

> **Instructions:** Run through every item after completing the notebook.
> All items must pass before submission.
> Mark each as ✅ PASS or ❌ FAIL with notes.

---

## 1. Notebook Execution

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1.1 | Notebook runs top-to-bottom without errors (FAST_MODE=True first) | ✅ PASS | |
| 1.2 | Notebook runs top-to-bottom without errors (FAST_MODE=False) | ✅ PASS | |
| 1.3 | All cells have saved output (not just blank cells) | ✅ PASS | |
| 1.4 | No TODO comments or placeholder text in any cell | ✅ PASS | |
| 1.5 | FAST_MODE and AUTO_RESUME are exposed in a single config cell at the top | ✅ PASS | |

---

## 2. Parameter Count Verification

| # | Check | Expected | Actual | Status |
|---|-------|----------|--------|--------|
| 2.1 | 2L MLP (width=413) parameter count | ~499K | `499,327` | ✅ PASS |
| 2.2 | 4L MLP (width=296) parameter count | ~499K | `499,066` | ✅ PASS |
| 2.3 | 8L MLP (width=215) parameter count | ~496K | `496,015` | ✅ PASS |
| 2.4 | BatchNorm overhead (8L, width=215) | 3,440 | `3,440` | ✅ PASS |
| 2.5 | Parameter count table displayed in notebook Section 3 | ✅ PASS | | |

---

## 3. Dataset Verification

| # | Check | Expected | Status |
|---|-------|----------|--------|
| 3.1 | Train split size | 54,000 | ✅ PASS |
| 3.2 | Validation split size | 6,000 | ✅ PASS |
| 3.3 | Test split size | 10,000 | ✅ PASS |
| 3.4 | Normalisation applied (mean=0.5, std=0.5) | [-1.0, 1.0] range | ✅ PASS |
| 3.5 | Sample visualisation (20 images, 2 per class) displayed | ✅ PASS | |

---

## 4. CSV Logging

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 4.1 | `results/results.csv` created after Experiment 1 | ✅ PASS | |
| 4.2 | `results/summary.csv` created after Experiment 1 | ✅ PASS | |
| 4.3 | Each completed run has exactly `epochs` rows in results.csv | ✅ PASS | |
| 4.4 | gradient columns layer_0_grad … layer_7_grad present in results.csv | ✅ PASS | |
| 4.5 | Summary CSV has columns: run_id, seed, depth, width, activation, batchnorm, parameter_count, test_acc, convergence_epoch, gradient_attenuation_ratio | ✅ PASS | |
| 4.6 | AUTO_RESUME correctly skips existing run_ids on re-run | ✅ PASS | Test by running Exp 1 twice |
| 4.7 | No duplicate run_ids in results.csv | ✅ PASS | |

---

## 5. Experiment Coverage

| # | Experiment | Config | Status |
|---|-----------|--------|--------|
| 5.1 | Exp 1: 2L ReLU, seeds [42, 123, 7] | 3 runs | ✅ PASS |
| 5.2 | Exp 1: 4L ReLU, seeds [42, 123, 7] | 3 runs | ✅ PASS |
| 5.3 | Exp 1: 8L ReLU, seeds [42, 123, 7] | 3 runs | ✅ PASS |
| 5.4 | Exp 2: 2L/4L/8L Sigmoid, seeds [42, 123, 7] | 9 runs (+ 9 ReLU reused) | ✅ PASS |
| 5.5 | Exp 3: 8L Sigmoid no-BN (reused), seeds [42, 123, 7] | 3 reused | ✅ PASS |
| 5.6 | Exp 3: 8L Sigmoid + BN, seeds [42, 123, 7] | 3 new runs | ✅ PASS |
| 5.7 | Total unique runs = 21 (12 reused + 9 new) | ✅ PASS | |

---

## 6. Gradient Logging

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 6.1 | Gradient norms captured per-epoch for all hidden layers | ✅ PASS | |
| 6.2 | Gradient capture occurs AFTER backward(), BEFORE step() | ✅ PASS | |
| 6.3 | Gradient attenuation ratio = first_grad / last_grad (higher = more vanishing) | ✅ PASS | |
| 6.4 | 8L Sigmoid shows gradient_attenuation_ratio >> 1 vs 8L ReLU ≈ 1 | ✅ PASS | Expected result |
| 6.5 | Gradient values non-zero (no dead-gradient rows in CSV) | ✅ PASS | |

---

## 7. Figure Generation

| # | Figure | File | Status |
|---|--------|------|--------|
| 7.1 | Fig 1: Val Accuracy vs Epoch (2L/4L/8L ReLU) | fig1_val_accuracy.png | ✅ PASS |
| 7.2 | Fig 2: Val Loss vs Epoch | fig2_val_loss.png | ✅ PASS |
| 7.3 | Fig 3: Test Accuracy Bar Chart (all 6 configs) | fig3_test_accuracy_bar.png | ✅ PASS |
| 7.4 | Fig 4A: Gradient Norm vs Layer Depth (8L, final epoch) | fig4a_gradient_by_layer.png | ✅ PASS |
| 7.5 | Fig 4B: Gradient Norm vs Epoch (first + last layer) | fig4b_gradient_over_epochs.png | ✅ PASS |
| 7.6 | Fig 5: Activation Heatmap (3×2 grid) | fig5_activation_heatmap.png | ✅ PASS |
| 7.7 | Fig 6: BatchNorm Recovery (2-panel) | fig6_batchnorm_recovery.png | ✅ PASS |
| 7.8 | All figures have axis labels, titles, and legends | ✅ PASS | |
| 7.9 | All figures use error bars / ± shading across seeds | ✅ PASS | |

---

## 8. Summary Table

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 8.1 | Summary table computed from summary.csv (not hard-coded) | ✅ PASS | |
| 8.2 | All results shown as mean ± std across seeds | ✅ PASS | |
| 8.3 | Columns present: Depth, Width, Activation, BatchNorm, Parameters, Test Acc, Std, Convergence Epoch, Grad Ratio | ✅ PASS | |
| 8.4 | n_seeds column shows number of completed seeds per config | ✅ PASS | |

---

## 9. Report Completeness

| # | Check | Status |
|---|-------|--------|
| 9.1 | Abstract written (≤150 words) | ✅ PASS |
| 9.2 | Introduction covers: depth, expressivity, vanishing gradients | ✅ PASS |
| 9.3 | Methodology section: dataset, architectures, parameter budget, controls | ✅ PASS |
| 9.4 | All 7 figures included with captions | ✅ PASS |
| 9.5 | Summary table present with actual numbers from experiments | ✅ PASS |
| 9.6 | Discussion section: interpret results, connect to theory | ✅ PASS |
| 9.7 | Threats to Validity section: all 4 mandatory threats included | ✅ PASS |
| 9.8 | Conclusion directly answers all 4 research questions | ✅ PASS |
| 9.9 | References section present (≥5 citations) | ✅ PASS |
| 9.10 | Report is 5–6 pages | ✅ PASS |

---

## 10. Seed Reproducibility

| # | Check | Status |
|---|-------|--------|
| 10.1 | `set_seed()` called before each run (model init + training) | ✅ PASS |
| 10.2 | `set_seed()` called for dataset split with fixed seed=42 | ✅ PASS |
| 10.3 | Re-running same seed + config produces identical results | ✅ PASS |
| 10.4 | `cudnn.deterministic=True`, `cudnn.benchmark=False` set | ✅ PASS |

---

## 11. Viva Preparation

| # | Check | Status |
|---|-------|--------|
| 11.1 | 25 questions generated with Critical/Likely/Advanced ranking | ✅ PASS |
| 11.2 | Each question has: Ideal Answer, Common Mistake, Follow-Up | ✅ PASS |
| 11.3 | Topics covered: vanishing gradients, ReLU, Sigmoid, BatchNorm, parameter budget, expressivity | ✅ PASS |

---

## 12. Final Submission Readiness

| # | Check | Status |
|---|-------|--------|
| 12.1 | Notebook saved with all outputs (not cleared) | ✅ PASS |
| 12.2 | results/ directory contains results.csv, summary.csv, figures/ | ✅ PASS |
| 12.3 | README.md contains setup + reproduction instructions | ✅ PASS |
| 12.4 | requirements.txt contains all dependencies | ✅ PASS |
| 12.5 | report/report.md filled with actual experiment numbers | ✅ PASS |

---

## Confidence Score

After completing all checks:

```
Total items:     69
Items passed:    69/69
Confidence:      100/100
```

> [!IMPORTANT]
> A score below 90/100 indicates items requiring attention before submission.
> All Critical items (Sections 1–6, 10) must pass for submission readiness.
