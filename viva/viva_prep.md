# Viva Preparation — Fashion-MNIST Depth Study
### Top 25 Questions Most Likely To Be Asked

**Format per question:**
- ❓ Question
- ✅ Ideal Answer
- ⚠️ Common Mistake
- ➕ Follow-Up

---

## 🔴 CRITICAL — Must Know (8–10 questions almost certain)

---

### Q1. Why did you control for parameter count across depths?

✅ **Ideal Answer:**  
Without controlling parameters, deeper models would have more total capacity, making any accuracy gain attributable to parameter count rather than depth itself. We kept parameters constant (~500K) by reducing width as depth increases (widths: 2L→413, 4L→296, 8L→215). This isolates depth as the independent variable.

⚠️ **Common Mistake:**  
Saying "we controlled parameters to be fair" without explaining the confound it prevents. The evaluator wants to hear the word "confound" or "isolated variable."

➕ **Follow-Up:** "But wouldn't different widths change what the network learns?"  
→ Yes — this is acknowledged as Threat #2 (width-depth confound). Width affects the feature space at each layer. Equal parameters is the best feasible control but doesn't eliminate this confound entirely.

---

### Q2. Mechanistically, why do deep Sigmoid networks fail?

✅ **Ideal Answer:**  
The sigmoid derivative σ'(x) = σ(x)(1−σ(x)) is bounded above by 0.25, occurring only at x=0. As we backpropagate through 8 layers, the chain rule multiplies these derivatives together: the cumulative gradient factor is at most 0.25^8 ≈ 1.5×10⁻⁵. Early layers receive near-zero gradient signal and effectively stop learning. The gradient attenuation ratio (first_layer_grad / last_layer_grad) quantifies this empirically.

⚠️ **Common Mistake:**  
Saying "sigmoid is old and bad." The evaluator wants the mathematical mechanism — bounded derivative → multiplicative decay through chain rule → vanishing gradient.

➕ **Follow-Up:** "But Adam has adaptive learning rates — doesn't that fix the problem?"  
→ Partially. Adam divides by a running average of squared gradients. If gradients are consistently tiny, the effective step size is rescaled — but the direction of the update still depends on the gradient sign, which becomes unreliable when the gradient magnitude approaches machine precision. Experimentally, 8L Sigmoid still underperforms 8L ReLU.

---

### Q3. Why does ReLU not suffer from vanishing gradients?

✅ **Ideal Answer:**  
In ReLU's active region (x > 0), f'(x) = 1 exactly. The chain rule product is 1^L = 1 — gradients pass through unchanged. Only dead neurons (permanently x < 0) block gradient flow, and these don't compound multiplicatively. This makes ReLU depth-scalable in a way sigmoid is not.

⚠️ **Common Mistake:**  
Saying "ReLU avoids saturation." Correct, but incomplete. Specifically say: derivative is 1 in active region → no multiplicative decay in backprop.

➕ **Follow-Up:** "What about the dying ReLU problem?"  
→ When a neuron's pre-activation is always negative, its gradient is always 0 — it never updates. This is a different problem (dead neurons) unrelated to vanishing gradients. Mitigated by Kaiming init (which our models use), but not our focus in this study.

---

### Q4. How does BatchNorm restore trainability in the Sigmoid network?

✅ **Ideal Answer:**  
BatchNorm normalises each layer's pre-activation to zero mean and unit variance: ẑ = (z − μ_B) / √(σ²_B + ε). This ensures inputs to each Sigmoid layer are centered near 0, where σ'(0) = 0.25 — the maximum value of the sigmoid derivative. By keeping inputs away from the saturation zones (|x| >> 0), BN prevents the derivative from collapsing to 0. The chain rule product is no longer dominated by near-zero terms, so gradients flow more freely.

⚠️ **Common Mistake:**  
Saying "BN normalises gradients." BN normalises *activations*, not gradients. The gradient improvement is an indirect consequence.

➕ **Follow-Up:** "BN adds parameters — doesn't that make the comparison unfair?"  
→ The 8L BN model adds 2×W×depth = 2×215×8 = 3,440 parameters vs ~496K total — less than 0.7% overhead. This is negligible and does not meaningfully change the comparison.

---

### Q5. What is the gradient attenuation ratio and what does it tell you?

✅ **Ideal Answer:**  
The gradient attenuation ratio is defined as: first_hidden_layer_gradient_norm / last_hidden_layer_gradient_norm. A ratio of 1 means uniform gradient flow — each layer receives equal signal. A ratio >> 1 means early layers receive much weaker gradients than late layers — the defining signature of vanishing gradients. We expect this ratio to be dramatically larger for 8L Sigmoid than 8L ReLU.

⚠️ **Common Mistake:**  
Confusing direction. We measure first/last (NOT last/first). Higher ratio = more vanishing. Lower ratio = healthier gradient flow.

➕ **Follow-Up:** "Why not use the average gradient across all layers?"  
→ The ratio of extremes (first vs last) is the most informative diagnostic because it captures the full range of attenuation across the network. The average would obscure whether the attenuation is monotonic or concentrated at specific layers.

---

### Q6. What is the difference between capacity and expressivity?

✅ **Ideal Answer:**  
**Capacity** is quantified by parameter count — how many different functions a network can fit, in a VC-dimension sense. **Expressivity** (or representational power) refers to the *type* of functions efficiently representable. Deep networks are more expressive than shallow ones of equal parameter count: they can represent certain hierarchical functions with exponentially fewer parameters (Telgarsky, 2016). However, we control capacity (parameter count) specifically to isolate expressivity differences.

⚠️ **Common Mistake:**  
Using capacity and expressivity interchangeably. They differ: equal-capacity networks can have different expressivity based on depth.

➕ **Follow-Up:** "Does Fashion-MNIST require high expressivity?"  
→ Probably not. Fashion-MNIST is relatively simple, and even 2L MLPs achieve 88–90% accuracy. Deep networks may offer higher expressivity but the task may not require it — which is one of the study's threats to validity.

---

### Q7. Why might depth NOT improve performance on Fashion-MNIST?

✅ **Ideal Answer:**  
Three reasons: (1) **Task simplicity** — Fashion-MNIST's discriminative features may be fully captured by a shallow representation. (2) **Width-depth tradeoff** — deeper models here use narrower layers (min width=215), reducing each layer's feature space. (3) **Optimisation difficulty** — even with ReLU, deeper networks have more complex loss landscapes and may converge to worse local minima. The dataset may simply not require the hierarchical representations that depth enables.

⚠️ **Common Mistake:**  
Only giving one reason. Provide all three to demonstrate depth of understanding.

➕ **Follow-Up:** "Where would depth help more?"  
→ Harder datasets with genuine hierarchical structure: CIFAR-10 (textures → shapes → objects), ImageNet, NLP tasks (characters → words → sentences). Fashion-MNIST images are relatively low-resolution and low-complexity.

---

### Q8. How did you ensure experimental reproducibility?

✅ **Ideal Answer:**  
Four mechanisms: (1) Fixed seeds: `random.seed(s)`, `numpy.seed(s)`, `torch.manual_seed(s)`, `torch.cuda.manual_seed_all(s)`. (2) Deterministic CUDA: `cudnn.deterministic=True`, `cudnn.benchmark=False`. (3) Fixed val split: `torch.Generator().manual_seed(42)` for train/val split — same 6K samples for all runs. (4) Explicit re-initialisation: `set_seed()` called before both model init and training loop, so weight init and batch ordering are both deterministic.

⚠️ **Common Mistake:**  
Saying "we set torch.manual_seed" without mentioning numpy and random, or not mentioning the val split seed is separate.

➕ **Follow-Up:** "Why not use `torch.use_deterministic_algorithms(True)`?"  
→ This flag raises RuntimeError for some CUDA operations (e.g., certain interpolation ops) in Colab GPU environments. `cudnn.deterministic=True` achieves equivalent determinism for all operations in our network without risking crashes.

---

## 🟡 LIKELY — High Probability of Being Asked

---

### Q9. Walk me through your model architecture.

✅ **Ideal Answer:**  
Each model is: Flatten (784) → [Linear(in, W) → (optional BN) → Activation] × depth → Linear(W, 10). The output layer has no BatchNorm or activation — it produces raw logits for CrossEntropyLoss. Width W is chosen so total parameters ≈ 500K: W=413 for 2L, 296 for 4L, 215 for 8L. Kaiming init for ReLU models, Xavier for Sigmoid.

⚠️ **Common Mistake:**  
Forgetting to mention the output layer is bare (no activation). CrossEntropy applies softmax internally.

➕ **Follow-Up:** "Where is BatchNorm placed relative to the activation?"  
→ Linear → BatchNorm1d → Activation. Standard placement per Ioffe & Szegedy (2015), normalising before the activation to keep pre-activations near zero.

---

### Q10. Why Adam and not SGD?

✅ **Ideal Answer:**  
Adam is adaptive — it scales the learning rate per parameter based on gradient history. This partially compensates for vanishing gradients in early layers by amplifying small gradient updates. SGD with a fixed LR would suffer more severely from gradient vanishing and require careful LR tuning per layer. Adam also converges faster in practice. For this study, Adam provides a better controlled comparison because it gives each model a fair shot at convergence.

⚠️ **Common Mistake:**  
"Adam is better than SGD." Technically incorrect in general — SGD with momentum + LR scheduling often outperforms Adam for image classification (Wilson et al., 2017). Adam is chosen here for stability and comparability, not superiority.

➕ **Follow-Up:** "Does Adam mask the vanishing gradient problem?"  
→ Partially — this is acknowledged in the discussion. The 8L Sigmoid may perform better than it would with SGD, which could make the "failure" less dramatic. However, gradient attenuation is still visible in the gradient norm analysis, and the mechanistic story holds regardless.

---

### Q11. What normalisation did you use and why?

✅ **Ideal Answer:**  
Pixel normalisation: mean=0.5, std=0.5, mapping [0,1] → [−1,1]. Centering inputs around zero is beneficial because: (1) symmetric activations like Sigmoid have maximum gradient at x=0, so centered inputs initialise neurons in the high-gradient region. (2) Zero-centered inputs avoid systematic bias in weight updates. The exact FashionMNIST statistics (mean≈0.286, std≈0.353) would be more precise, but 0.5/0.5 is widely accepted and mandated by the assignment spec.

⚠️ **Common Mistake:**  
"We normalised because that's standard." Give the mechanistic reason: gradient preservation for Sigmoid, zero-centering for fast convergence.

---

### Q12. What is the convergence epoch metric and how did you define it?

✅ **Ideal Answer:**  
Convergence epoch is defined as the first epoch where val_acc ≥ 0.95 × max(val_acc across all epochs). This captures when the model has reached 95% of its best performance. We prefer this over "99% of final accuracy" because: (1) final accuracy may dip due to overfitting, making "99% of final" retroactively pessimistic; (2) 95% of max is robust to post-plateau fluctuations. A lower convergence epoch indicates faster learning.

⚠️ **Common Mistake:**  
Using epoch of highest val_acc — this is the optimal epoch, not the convergence epoch.

---

### Q13. What were your three experiments and what did each answer?

✅ **Ideal Answer:**  
Exp 1 (Depth Comparison, ReLU): Answers RQ1 — does depth help under equal parameters? All 3 ReLU depths compared. Exp 2 (Activation Study): Answers RQ2+RQ3 — how does activation affect gradient flow? 3 depths × 2 activations (ReLU, Sigmoid). Exp 3 (BatchNorm Recovery): Answers RQ4 — can BN fix 8L Sigmoid? 8L Sigmoid ±BN. All experiments use 3 seeds; results are mean ± std.

⚠️ **Common Mistake:**  
Not mapping experiments to research questions explicitly.

---

### Q14. Explain Figure 4A (Gradient Norm vs Layer Depth).

✅ **Ideal Answer:**  
The x-axis is layer index (1=input, 8=output). The y-axis is the L2 norm of the weight gradient at each layer, captured at the final training epoch. For 8L ReLU, norms should be approximately stable across layers. For 8L Sigmoid, norms should decay from right to left — the further a layer is from the output, the weaker its gradient signal. This visually demonstrates vanishing gradients. The gradient attenuation ratio (layer 1 norm / layer 8 norm) quantifies the disparity.

⚠️ **Common Mistake:**  
Reading the x-axis from the wrong direction. Layer 1 is the INPUT-side layer.

---

### Q15. What is the purpose of using three random seeds?

✅ **Ideal Answer:**  
A single run is insufficient to claim any result because the outcome depends on random weight initialisation and mini-batch ordering. Three seeds allow us to report mean ± std, giving a rough confidence interval. If standard deviations are small relative to differences between models, the differences are likely real. If std is large, no strong conclusion can be drawn. This is standard practice for controlled deep learning experiments.

⚠️ **Common Mistake:**  
"Three seeds make results reproducible." Seeds make results reproducible — that's different. Multiple seeds quantify *stability* and allow *statistical reporting*.

---

## 🟢 ADVANCED — Demonstrates Deep Understanding (differentiates top marks)

---

### Q16. You showed BatchNorm improves Sigmoid networks. Why not just use ReLU?

✅ **Ideal Answer:**  
ReLU is simpler, computationally cheaper, and typically outperforms Sigmoid even with BatchNorm. The reason we study 8L Sigmoid + BN is pedagogical: it directly answers whether BN addresses the vanishing gradient failure mode (gradient flow) or other issues (capacity, optimization landscape). The comparison isolates BN's contribution. In practice, ReLU is the correct choice for deep MLPs.

---

### Q17. What is the relationship between batch norm and internal covariate shift?

✅ **Ideal Answer:**  
Internal covariate shift (Ioffe & Szegedy, 2015) refers to the change in the distribution of each layer's inputs during training, as earlier layers update. BN addresses this by normalising each layer's inputs to zero mean, unit variance. Note: subsequent research (Santurkar et al., 2018) showed BN's main benefit may be smoothing the loss landscape rather than reducing ICS per se. For our purposes, the practical effect — preventing Sigmoid saturation — is well-established regardless of the theoretical explanation.

---

### Q18. What does the log scale on the gradient plots tell you?

✅ **Ideal Answer:**  
A log scale is necessary when gradient magnitudes span multiple orders of magnitude (e.g., 10⁻¹ to 10⁻⁵). On a linear scale, small values would be invisible. The log scale reveals whether the decay is *exponential* (linear on log scale) which is the signature of vanishing gradients. For Sigmoid networks, an approximately linear decline on the log-y plot confirms exponential gradient decay — consistent with the σ'(x)^L theoretical bound.

---

### Q19. What would you expect if you used Xavier init for ReLU models?

✅ **Ideal Answer:**  
Xavier initialisation assumes symmetric activations (like Tanh/Sigmoid) where neurons are in their linear regime. For ReLU, Xavier underestimates the variance needed because ReLU zeros out negative half of the input distribution. This would cause signal variance to halve with each layer (gradient variance would also decay), leading to slower convergence. Kaiming init (He et al., 2015) corrects for this by scaling by √(2/fan_in). Using Xavier for ReLU would reduce initial gradient norms and potentially slow training.

---

### Q20. How would you extend this experiment to be more rigorous?

✅ **Ideal Answer:**  
Several improvements: (1) More seeds (5–10) for better statistical power and Welch's t-tests for significance. (2) Multiple datasets (CIFAR-10) to test generalisability. (3) Hyperparameter search for learning rate (flat LR=0.001 may not be optimal for all depths). (4) Additional baselines: ResNet-style connections (residual connections are another solution to vanishing gradients). (5) Activation statistics monitoring (mean/std of activations per layer) alongside gradient norms.

---

### Q21. What is the information bottleneck theory and how does it relate to depth?

✅ **Ideal Answer:**  
The information bottleneck (Tishby & Schwartz-Ziv, 2017) proposes that deep networks successively compress input information, retaining only task-relevant features. Deeper networks may create better bottlenecks by composing simple transformations. However, this theory is contested (Saxe et al., 2018) and applies to specific training dynamics. For our purposes: depth theoretically enables better feature composition, but empirically on Fashion-MNIST, simple features suffice.

---

### Q22. Why might equal parameter count not be the best control variable?

✅ **Ideal Answer:**  
Equal parameters ensures similar total capacity but may not be the most meaningful control. Alternative controls: (1) Equal FLOPS (compute cost) — deeper networks with same params use more FLOPs per forward pass. (2) Equal width — some papers argue width is the natural comparand for depth. (3) Equal test accuracy on a baseline — find configs achieving the same accuracy, then compare training efficiency. Each choice tells a different story; equal parameters is conventional and simple but not uniquely correct.

---

### Q23. Sigmoid's derivative is ≤0.25 — but did your 8L Sigmoid actually fail to train?

✅ **Ideal Answer:**  
With Adam optimiser, 8L Sigmoid may still achieve reasonable accuracy (e.g., 80–85%) because Adam's per-parameter adaptive LR partially compensates for small gradients. The "failure" is relative: (1) It underperforms 8L ReLU, demonstrating gradient-flow disadvantage. (2) The gradient attenuation ratio reveals that early layers are learning much slower than late layers, regardless of final accuracy. (3) Convergence is slower (higher convergence epoch). This is why we use gradient norms as the primary evidence, not just accuracy.

---

### Q24. What is BatchNorm's effect during inference vs training?

✅ **Ideal Answer:**  
During training, BN uses batch statistics (mean, variance of the current mini-batch). During inference, it uses running statistics (exponential moving average of batch means/variances accumulated during training). This distinction matters: (1) BN behaves differently with batch size=1 at inference. (2) If the test distribution differs from training (covariate shift), running stats may be stale. (3) We call `model.eval()` before test evaluation, which switches BN to inference mode — using running statistics.

⚠️ **Common Mistake:**  
Not knowing that `model.eval()` changes BN behavior. Many students forget this distinction.

---

### Q25. If you had to explain this project in 2 minutes to a non-ML audience, what would you say?

✅ **Ideal Answer:**  
"We tested whether making a neural network deeper always makes it smarter, using clothing image classification as the task. We kept the total size of the network constant and compared networks with 2, 4, and 8 layers. The key finding is: making a network deeper is not free. With the wrong building block (sigmoid function), deeper networks actually become harder to train because useful learning signals get too weak to reach the early layers — like a telephone game where the message is garbled by the time it arrives. Switching to a better building block (ReLU) fixes this. Adding batch normalisation also helps by preventing the signal from fading. Depth is only beneficial if the training signal can flow freely through the entire network."

---

## Quick Reference: Critical Formulae

| Formula | Meaning |
|---------|---------|
| `σ'(x) = σ(x)(1−σ(x)) ≤ 0.25` | Sigmoid derivative bound |
| `0.25^8 ≈ 1.5×10⁻⁵` | Max gradient factor after 8 sigmoid layers |
| `f'(x) = 1 if x>0, else 0` | ReLU derivative |
| `ratio = ‖∇W₁‖ / ‖∇W₈‖` | Gradient attenuation ratio (higher = more vanishing) |
| `ẑ = (z−μ_B)/√(σ²_B+ε)` | BatchNorm normalisation |
| `params ≈ W_in×W + (D−1)×W² + W×C` | MLP parameter count formula |

---

## Key Papers to Know by Name

| Paper | Key Contribution |
|-------|----------------|
| Hochreiter (1991) | First formal analysis of vanishing gradients |
| Glorot & Bengio (2010) | Xavier initialisation; saturation in deep sigmoid networks |
| He et al. (2015) | Kaiming initialisation; ReLU depth scaling |
| Ioffe & Szegedy (2015) | Batch Normalisation |
| Kingma & Ba (2014) | Adam optimiser |
