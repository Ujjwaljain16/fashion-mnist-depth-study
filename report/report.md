# Fashion-MNIST: Shallow vs Deep Networks
### Submission Report

**Course:** Deep Learning I  
**Assignment:** Section 2 — Network Depth Experiments  
**Dataset:** Fashion-MNIST  
**Framework:** PyTorch  
**Date:** [Fill in submission date]  
**Team:** [Fill in team members]  

---

## Abstract

> *(~150 words — fill in after experiments are complete)*

This paper investigates the effect of network depth on classification performance and gradient dynamics in Multi-Layer Perceptrons (MLPs) trained on Fashion-MNIST. Three MLP architectures — 2, 4, and 8 hidden layers — are compared under a fixed parameter budget of approximately 500,000 parameters, with layer widths adjusted accordingly. We conduct three controlled experiments: a depth comparison under ReLU activation, an activation function study contrasting ReLU with Sigmoid, and a BatchNorm recovery experiment targeting the identified failure mode. All experiments use three random seeds; results are reported as mean ± standard deviation. Gradient norms per layer are logged to provide mechanistic insight into training dynamics beyond accuracy alone. [Add 2–3 sentences summarising key findings after running experiments.] The study demonstrates that the interaction between depth, activation function, and normalisation strategy is the primary determinant of whether a deep MLP is effectively trainable.

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
| 2L    | 413   | [Fill from notebook] |
| 4L    | 296   | [Fill from notebook] |
| 8L    | 215   | [Fill from notebook] |
| 8L+BN | 215  | [Fill from notebook] |

**Widths were adjusted so parameter count remains approximately constant across depths** (~500K). This ensures any observed performance differences are attributable to depth rather than parameter count.

**Weight Initialisation:**
- ReLU models: Kaiming Uniform (He et al., 2015) — accounts for ReLU's half-zeroing
- Sigmoid models: Xavier Uniform (Glorot & Bengio, 2010) — designed for symmetric saturating activations

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

- **Equal parameter budget:** widths scaled with depth
- **Fixed random seeds:** 3 seeds for statistical reporting
- **Identical training procedure:** Adam optimizer, same LR, epochs, batch size
- **Identical data split:** fixed generator seed for val partition
- **No early stopping:** all runs complete 50 epochs

---

## Experiments

### Experiment 1: Depth Comparison (ReLU)

**Purpose:** Answers RQ1.  
**Models:** 2L, 4L, 8L — all ReLU, no BatchNorm.

[Figure 1: Validation Accuracy vs Epoch — place here]  
[Figure 2: Validation Loss vs Epoch — place here]  
[Figure 3: Test Accuracy Bar Chart — place here]

**Results:**

> *(Fill after experiments — paste from summary table)*

| Model | Test Accuracy | Std | Convergence Epoch | Grad Ratio |
|-------|--------------|-----|------------------|------------|
| 2L ReLU | | | | |
| 4L ReLU | | | | |
| 8L ReLU | | | | |

**Interpretation:**

> *(Choose the applicable interpretation after observing results)*

*If depth helps:* The [N]L model achieved the highest test accuracy ([X]% ± [Y]%), suggesting Fashion-MNIST contains sufficient complexity to benefit from hierarchical feature learning, even under equal parameter budgets.

*If depth is neutral:* All three models achieve similar test accuracy ([range]%), suggesting that Fashion-MNIST's discriminative features are fully learnable by a shallow MLP at this parameter scale. Depth provides diminishing returns.

*If depth hurts:* The 8L model underperforms (test acc = [X]% vs [Y]% for 2L), consistent with optimisation difficulties in deep networks even with ReLU, despite a well-initialised network.

**Convergence Analysis:** The convergence epoch (first epoch reaching 95% of maximum val_acc) indicates [faster/similar/slower] convergence for deeper models. This [supports/contradicts] the hypothesis that additional layers provide faster hierarchical learning.

---

### Experiment 2: Activation Study

**Purpose:** Answers RQ2 + RQ3.  
**Models:** 2L, 4L, 8L × {ReLU, Sigmoid} — no BatchNorm.

[Figure 4A: Gradient Norm vs Layer Depth — place here]  
[Figure 4B: Gradient Norm vs Epoch (supplemental) — place here]  
[Figure 5: Activation Heatmap — place here]

**Gradient Attenuation Results:**

> *(Fill after experiments)*

| Model | Gradient Attenuation Ratio |
|-------|--------------------------|
| 8L ReLU | |
| 8L Sigmoid | |

**Interpretation:**

Figure 4A shows gradient L2 norms plotted against layer index (1 = input side, 8 = output side) at the final training epoch.

*If Sigmoid shows decay:* The Sigmoid gradient norms decay from layer 8 toward layer 1 by a factor of approximately [X], while ReLU maintains [relatively stable/gently varying] norms across layers. This directly demonstrates vanishing gradients: early layers in the 8L Sigmoid network receive gradient signal [X]× weaker than the output-side layers, severely impeding their ability to learn. The gradient attenuation ratio of [X] (Sigmoid) vs [Y] (ReLU) quantifies this disparity.

The theoretical expectation is σ'(x) ≤ 0.25 per layer → 0.25^8 ≈ 1.5×10⁻⁵ cumulative attenuation. The observed attenuation is [consistent with / less severe than] this bound because Adam's adaptive learning rate partially compensates for small gradients.

---

### Experiment 3: BatchNorm Recovery

**Purpose:** Answers RQ4.  
**Models:** 8L Sigmoid (no BN) vs 8L Sigmoid + BatchNorm.

[Figure 6: BatchNorm Recovery — place here]

**Results:**

> *(Fill after experiments)*

| Model | Test Accuracy | Std | Convergence Epoch | Grad Ratio |
|-------|--------------|-----|------------------|------------|
| 8L Sigmoid (no BN) | | | | |
| 8L Sigmoid + BN   | | | | |

**Interpretation:**

*If BN helps significantly:* BatchNorm recovers substantial accuracy from [X]% to [Y]%, closing [Z]% of the gap to the 8L ReLU baseline. The gradient attenuation ratio drops from [A] to [B], confirming that BatchNorm directly addresses the gradient flow problem. By normalising pre-activations to zero mean, BN keeps inputs to Sigmoid in its linear region (near 0, where σ'(x) is maximised at ~0.25), preventing cascading gradient decay.

*If BN helps partially:* BatchNorm improves accuracy from [X]% to [Y]% but does not fully close the gap to ReLU. This suggests that while BN mitigates gradient vanishing, Sigmoid's saturating nature and bounded derivative still impose a training disadvantage compared to ReLU at this depth.

---

## Analysis & Insights

### Core Insight 1: Depth vs. Trainability

The results highlight a fundamental tension in deep learning: depth increases a network's theoretical *expressivity* (ability to represent complex functions) but simultaneously increases *optimisation difficulty* (harder to train via gradient descent). Whether depth helps or hurts in practice depends on:

1. **Dataset complexity** — simple datasets may not require hierarchical representations
2. **Activation function** — whether gradient flow is preserved across layers
3. **Architectural mitigations** — normalisation strategies that stabilise gradient flow

### Core Insight 2: Gradient Norm as a Diagnostic Tool

The gradient attenuation ratio (first_layer_grad / last_layer_grad) provides a mechanistic explanation for performance differences. A ratio >> 1 identifies networks where early layers are effectively frozen despite ongoing weight updates later in the network. This metric should be standard practice in deep network debugging.

### Core Insight 3: BatchNorm's Mechanism

BatchNorm does not directly add representational capacity (the weight overhead is minimal: 2W parameters per layer). Its benefit is entirely mechanistic: by normalising layer inputs, it keeps activations in the gradient-transmitting region of the activation function. This is particularly valuable for Sigmoid, which saturates and kills gradients at ±∞.

---

## Threats to Validity

### Threat 1: Task Simplicity
Fashion-MNIST achieves 90%+ accuracy with even shallow MLPs. The dataset may be too simple for depth to provide meaningful benefits, making the depth-performance comparison less informative for harder, real-world tasks (e.g., CIFAR-10, ImageNet).

### Threat 2: Width-Depth Confound
Controlling for parameter count requires unequal widths across depths. Narrower networks may learn qualitatively different feature representations, introducing a confound: differences in performance may partially reflect the width change rather than the depth change.

### Threat 3: Limited Statistical Power
Three seeds provide a confidence interval but limited statistical power. A statistically significant difference between 2L and 4L models would require additional seeds and formal significance testing (e.g., Welch's t-test), which is outside this study's scope.

### Threat 4: Scope Limitation
Results apply only to fully-connected MLPs. CNNs have fundamentally different depth-performance relationships due to parameter sharing, local receptive fields, and built-in translational invariance. Conclusions should not be extrapolated to convolutional architectures.

---

## Conclusion

### Direct Answers to Research Questions

**RQ1: Does increasing depth improve performance under a fixed parameter budget?**

[Fill after experiments: "Yes, modestly — the [N]L model outperformed the 2L baseline by [X]%, suggesting [explanation]." OR "No — all depths achieve similar accuracy (~[X]%), indicating that Fashion-MNIST complexity is within 2L MLP capacity at this parameter scale."]

**RQ2: How does depth affect gradient flow?**

Depth amplifies gradient attenuation. Even with ReLU, deeper networks show [slightly/substantially] larger gradient attenuation ratios. With Sigmoid, 8 layers produce gradient signals at layer 1 that are [X]× weaker than at layer 8, making early-layer learning effectively impossible without architectural intervention.

**RQ3: What role does activation choice play?**

Activation choice is the single largest determinant of trainability in deep networks. ReLU's gradient-preserving property (f'(x) ∈ {0,1}) enables training at depth 8, while Sigmoid's bounded derivative (σ'(x) ≤ 0.25) causes exponential gradient decay, making the 8L Sigmoid model [dramatically underperform / fail to converge] compared to its ReLU counterpart.

**RQ4: Can BatchNorm restore trainability?**

[Fill after experiments: "Yes — 8L Sigmoid + BN achieves [X]% vs [Y]% without BN, recovering [Z]% of the accuracy gap to the ReLU baseline. Mechanistically, BatchNorm reduces the gradient attenuation ratio from [A] to [B], confirming that gradient flow is the bottleneck addressed." OR note if BN helps only partially.]

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
