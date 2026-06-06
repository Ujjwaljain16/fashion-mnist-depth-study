# Fashion-MNIST: Shallow vs Deep Networks
### Submission Report

**Course:** SST Neural Network & Intro to Computer Vision (ML III)  
**Assignment:** Section 2 — 17 Shallow vs. Deep Networks  
**Dataset:** Fashion-MNIST  
**Framework:** PyTorch  
**Team:** Group 8  
- Ujjwal Jain (10173)  
- Pratham Onkar (10136)  
- Aditya Kumar Rai (10178)  
- Dhairya Motta (10202)  
- Arman Barbhuiya (10196)  
- Lakshay Jagga (10398)  
- Piyush Kumar Gupta (10332)  
- Iyad Farooq (10116)  

---

## Abstract

This paper investigates the effect of network depth on classification performance and gradient dynamics in Multi-Layer Perceptrons (MLPs) trained on Fashion-MNIST. Three MLP architectures—2, 4, and 8 hidden layers—are evaluated under a fixed parameter budget of ~500,000 parameters. We find that under capacity control, non-saturating activations (ReLU) are effectively depth-invariant, achieving 88.7%–88.8% test accuracy across all depths. In contrast, saturating activations (Sigmoid) severely degrade with depth, dropping from 88.59% (2L) to 86.74% (8L). Gradient flow analysis reveals the mechanism: at initialization, Layer 1 gradients in the 8L Sigmoid model collapse to ~10⁻⁹, incurring a massive convergence penalty (12 epochs vs. 2–3 epochs for ReLU). While BatchNorm fully restores convergence speed by normalizing pre-activations, it cannot bypass the bounded derivative of the sigmoid function, leaving a residual accuracy gap. The study demonstrates that trainability, governed by activation choice and normalization, dominates theoretical depth advantages for MLPs on moderate-complexity datasets.

---

## Problem Understanding

### Background: Depth, Expressivity, and Gradient Flow

Neural network depth is theoretically motivated by the **universal approximation theorem with depth** (Telgarsky, 2016; Eldan & Shamir, 2016): deep networks can represent certain function classes exponentially more efficiently than shallow networks with the same number of parameters. However, *expressivity* (the ability to represent a function) does not guarantee *trainability* (the ability to learn that function via gradient descent).

Two practical phenomena limit the benefits of depth:

**1. Vanishing Gradients**  
Backpropagation computes gradients as products of Jacobian matrices across layers. For activations with bounded derivatives (e.g., Sigmoid: σ'(x) ≤ 0.25), gradient magnitude decays exponentially with depth. For 8 Sigmoid layers:

$$\|\nabla W_1\| \approx \prod_{l=1}^{8} \sigma'(z_l) \cdot \|\nabla W_8\| \leq 0.25^8 \|\nabla W_8\| \approx 1.5 \times 10^{-5} \|\nabla W_8\|$$

Early layers receive negligible gradient signal and fail to learn.

**2. Width-Depth Trade-off**  
Under a fixed parameter budget, increasing depth requires decreasing width. Narrower networks may lose representational capacity within each layer, potentially offsetting depth's theoretical benefits.

### Research Questions

| ID | Research Question |
|----|------------------|
| RQ1 | Does increasing depth improve performance under a fixed parameter budget? |
| RQ2 | How does depth affect gradient flow? |
| RQ3 | What role does activation choice play? |
| RQ4 | Can BatchNorm restore trainability to deep Sigmoid networks? |

---

## Implementation

### Dataset

Fashion-MNIST consists of 70,000 grayscale 28×28 images across 10 clothing categories.

| Split | Size |
|-------|------|
| Train | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |

**Normalisation:** pixel values ∈ [0, 1] after `ToTensor()`, then normalised to [−1, 1] using mean=0.5, std=0.5. Centering inputs reduces initial Sigmoid saturation and accelerates convergence. The split is deterministic (generator seed=42) across all experiments.

### Architectures

All three MLPs use the same template:

```
Input (784)
→ [ Linear(W_in, W) → [BatchNorm1d(W)] → Activation ] × depth
→ Linear(W, 10)
```

| Depth | Width | Actual Parameters |
|-------|-------|------------------|
| 2L    | 413   | 499,327 |
| 4L    | 296   | 499,066 |
| 8L    | 215   | 496,015 |
| 8L+BN | 215   | 499,455 |

**Widths were adjusted so parameter count remains approximately constant across depths** (~500K). This ensures any observed performance differences are attributable to depth rather than parameter count.

**Weight Initialisation:**
- PyTorch default `nn.Linear` initialization (Kaiming uniform) is used across all models. While optimal for ReLU, Xavier uniform would be theoretically preferable for Sigmoid networks.

**BatchNorm placement:** Linear → BatchNorm1d → Activation (original Ioffe & Szegedy placement).

### Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Epochs | 50 |
| Loss | CrossEntropyLoss |
| Seeds | 42, 123, 7 |

### Experimental Controls

- **Equal parameter budget:** widths scaled with depth.
- **Fixed random seeds:** 3 seeds for statistical reporting.
- **Identical training procedure:** Adam optimizer, same LR, epochs, batch size.
- **No early stopping:** all runs complete 50 epochs.

---

## Experiments

### Experiment 1: Depth Comparison (ReLU)

**Purpose:** Answers RQ1.  
**Models:** 2L, 4L, 8L — all ReLU, no BatchNorm.

*Reference Figures in Notebook: Figure 1, Figure 2, Figure 3*

**Results:**

| Model | Test Accuracy | Std | Convergence Epoch | Grad Ratio (Mean) |
|-------|--------------|-----|------------------|------------|
| 2L ReLU | 0.8879 | 0.0033 | 1.3 | 1.95× |
| 4L ReLU | 0.8879 | 0.0027 | 1.3 | 4.52× |
| 8L ReLU | 0.8872 | 0.0022 | 2.3 | 10.54× |

**Interpretation:**

All three models achieve nearly identical test accuracy (88.72%–88.79%), suggesting that Fashion-MNIST's discriminative features are fully learnable by a shallow MLP at this parameter scale. Depth provides diminishing returns. The accuracy difference between 2L and 8L ReLU is 0.0007, which is smaller than the standard deviation, meaning the null hypothesis (depth does not affect performance in capacity-controlled ReLU networks) cannot be rejected.

**Convergence Analysis:** The convergence epoch (first epoch reaching 95% of maximum val_acc) indicates identically rapid convergence (1.3 to 2.3 epochs) across all depths. This contradicts the hypothesis that additional layers provide faster hierarchical learning on this dataset.

---

### Experiment 2: Activation Study

**Purpose:** Answers RQ2 + RQ3.  
**Models:** 2L, 4L, 8L × {ReLU, Sigmoid} — no BatchNorm.

*Reference Figures in Notebook: Figure 4A, Figure 4B, Figure 5*

**Results (Test Accuracy):**
- 2L Sigmoid: 0.8859 ± 0.0039
- 4L Sigmoid: 0.8810 ± 0.0027
- 8L Sigmoid: 0.8674 ± 0.0020

**Interpretation:**

Unlike ReLU, Sigmoid performance degrades monotonically with depth. The 8L Sigmoid model underperforms 8L ReLU by 1.98 percentage points.

Gradient flow provides the mechanistic explanation. Figure 4B (in notebook) shows that at epoch 1, Layer 1 gradients in the 8L Sigmoid model collapse to ~10⁻⁹. This represents a 9-order-of-magnitude deficit compared to ReLU, severely impeding early-layer learning. Consequently, 8L Sigmoid requires ~12 epochs to converge, whereas ReLU models converge in 2-3 epochs.

---

### Experiment 3: BatchNorm Recovery

**Purpose:** Answers RQ4.  
**Models:** 8L Sigmoid (no BN) vs 8L Sigmoid + BatchNorm.

*Reference Figures in Notebook: Figure 6*

**Results:**

| Model | Test Accuracy | Std | Convergence Epoch | Grad Ratio (Mean) |
|-------|--------------|-----|------------------|------------|
| 8L Sigmoid (no BN) | 0.8674 | 0.0020 | 12.0 | 7.72× |
| 8L Sigmoid + BN   | 0.8812 | 0.0017 | 2.3 | 11.06× |

**Interpretation:**

BatchNorm recovers substantial accuracy, improving from 86.74% to 88.12%. Crucially, it completely rescues convergence speed, reducing it from 12 epochs down to 2.3 epochs (matching ReLU speed). By normalising pre-activations to zero mean, BN keeps inputs to the Sigmoid function in its linear region (near 0, where σ'(x) is maximised at ~0.25), preventing cascading gradient decay at initialization. 

However, BN does not fully close the accuracy gap to the ReLU baseline (88.72%). This suggests that while BN mitigates the optimization difficulty, the Sigmoid function's bounded derivative still imposes an expressivity disadvantage compared to ReLU.

---

## Analysis & Insights

### Core Insight 1: Depth vs. Trainability

The results highlight a fundamental tension in deep learning: depth increases a network's theoretical expressivity but simultaneously increases optimization difficulty. Whether depth helps or hurts in practice depends on the activation function. ReLU safely decoupled depth from degradation, whereas Sigmoid models suffered compounded gradient attenuation.

### Core Insight 2: Gradient Norm as a Diagnostic Tool

The gradient attenuation ratio provides a mechanistic explanation for performance differences. An initialization ratio approaching 10⁹ identifies networks where early layers are frozen. Gradient diagnostic plots should be standard practice for debugging deep networks.

### Core Insight 3: Overfitting Dynamics (Hidden Insight)

Validation loss curves reveal that all ReLU models reach minimum loss around epochs 8–12 and then severely overfit (loss nearly doubles by epoch 50). Interestingly, the 8L ReLU model exhibits lower terminal validation loss than the shallower models, suggesting that deeper, narrower MLPs provide stronger implicit regularization under Adam.

---

## Threats to Validity

### Threat 1: Early Stopping Absence
Test accuracy is evaluated strictly at epoch 50, whereas peak validation accuracy occurs around epoch 10. Test accuracy underestimates optimal performance by ~1 percentage point uniformly across all configurations.

### Threat 2: Initialization Bias
PyTorch's default Kaiming initialization (designed for ReLU) is used across all models. Xavier initialization would theoretically provide better starting conditions for Sigmoid networks, though it cannot circumvent the fundamental bounds of the sigmoid derivative across 8 layers.

### Threat 3: Task Simplicity
Fashion-MNIST achieves ~88.8% accuracy with even shallow MLPs. The dataset may be too simple for depth to provide meaningful hierarchical benefits.

### Threat 4: Limited Statistical Power
Three seeds provide a confidence interval but limited statistical power. A statistically significant difference between 2L and 4L models would require additional seeds and formal significance testing.

---

## Conclusion

### Direct Answers to Research Questions

**RQ1: Does increasing depth improve performance under a fixed parameter budget?**
**No.** Under strict capacity control (~500k parameters), increasing depth provides zero performance benefit for Fashion-MNIST. ReLU accuracy remained static (88.79% at 2L vs 88.72% at 8L), while Sigmoid accuracy systematically degraded (88.59% at 2L vs 86.74% at 8L).

**RQ2: How does depth affect gradient flow?**
Depth introduces severe gradient attenuation for saturating activations. In 8L Sigmoid, gradients at the input layer collapsed to 10⁻⁹ at initialization. Non-saturating activations (ReLU) bypass this decay, preserving stable gradient flow regardless of depth.

**RQ3: What role does activation choice play?**
Activation choice is the single largest determinant of trainability in deep networks. ReLU's gradient-preserving property (f'(x) ∈ {0,1}) enables rapid training at depth 8, while Sigmoid's bounded derivative causes exponential gradient decay, significantly slowing convergence and lowering final accuracy.

**RQ4: Can BatchNorm restore trainability?**
**Partially.** Batch Normalization rescued the trainability of the 8L Sigmoid model (accelerating convergence from 12 epochs to 2-3 epochs), but the final accuracy (88.12%) remained structurally bottlenecked below the ReLU ceiling (88.72%).

### Final Summary

This study demonstrates that understanding *why* a model fails — through gradient diagnostics — is more informative than observing *what* accuracy it achieves. The gradient norm analysis provides mechanistic evidence for vanishing gradients in deep Sigmoid networks and shows how BatchNorm directly addresses this failure mode by normalising layer inputs. For practitioners, the lesson is: activation function choice and normalisation strategy matter far more than depth alone for trainability.

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
2. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*.
4. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
5. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *ICLR*.
6. Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen. *Diploma thesis, TU Munich*.
7. Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747*.
