> **Reliable and Balanced Transfer Learning for Generalized Multimodal Face Anti-Spoofing**  (MMDG++)
> (Under Review)

---

## 🔍 Abstract

Face Anti-Spoofing (FAS) is essential for securing face recognition systems against presentation attacks.  
Advancements in sensor technology and multimodal learning have led to the development of multimodal FAS methods.  
However, these methods often struggle to generalize to unseen attacks and diverse environments due to:

1. **Modality unreliability** – sensors like depth and infrared may suffer from domain shifts, hindering reliable cross-modal fusion.
2. **Modality imbalance** – over-reliance on a dominant modality weakens robustness against certain attacks.

To address these challenges, we propose **MMDG++**, a multimodal domain-generalized FAS framework built upon the vision-language model **CLIP**.

- We design the **U-Adapter++** (Uncertainty-Guided Cross-Adapter++) to filter out unreliable regions within each modality, enabling robust multimodal interactions.
- We introduce **ReGrad** (Rebalanced Modality Gradient Modulation), which adaptively balances modality-specific gradients for stable convergence.
- We further enhance generalization with **ADPs** (Asymmetric Domain Prompts), which leverage CLIP’s language priors to learn generalized decision boundaries across modalities.

We also construct a new **multimodal FAS benchmark** to evaluate generalizability under varied deployment conditions. Extensive experiments demonstrate that our method achieves **state-of-the-art performance** and superior generalization across multiple FAS tasks.

---

## 📁 Contents

- `/models` – Main implementation of MMDG++
- `/MMDG-Benchmark` – Benchmark splits and evaluation protocols
  
---

## 📄 Citation

Coming soon.

---

## 📝 License

This repository will be released under an open-source license upon paper acceptance.

---

## 📬 Contact

For questions or collaborations, please contact:  
[Xun Lin] – [linxun@buaa.edu.cn]
