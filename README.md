# ConvMamba-SLAM 🚀

An independent research project focused on evaluating and integrating **Mamba-based global context** into the **DROID-SLAM** framework. This study investigates the trade-offs between dense geometric structures and state-space model (SSM) mechanisms for visual odometry.

This research was conducted at the **Dynamic Networks and Control Laboratory, The University of Texas at Arlington**.

📄 **[Read the full Technical Report here](https://docs.google.com/document/d/1zOgWIjg4yOpx_tMVxqnB0578jk_RZahDPcdbfN7Xdfw/edit?tab=t.0#heading=h.10y9jieiukup)

## 📌 Key Frameworks & Methodology

This project explores two core architecture modifications to the baseline DROID-SLAM:
1. **JEGO Scan Integration:** Injecting global awareness based on selected edge pairs using JamMa's JEGO scan.
2. **Mamba-based Update Operators:** Replacing conventional ConvGRU blocks with four Mamba blocks to manage residual refinement and hidden state updates.

---

## 💻 Training & Evaluation Setup

* **Hardware:** Trained utilizing H100 GPUs provided by the **Texas Advanced Computing Center (TACC)**.
* **Training Data:** Strictly followed the original DROID-SLAM training protocol using the **TartanAir dataset** to ensure a fair and direct comparison.
* **Evaluation Benchmarks:** Evaluated the model's odometry and tracking performance across diverse environments using the **TUM dataset** and **Bonn dataset**.

---

## 📊 Experimental Results & Insights

Unlike superficial integrations, this project delivers critical quantitative evaluations on the alignment between baseline geometric principles and new deep learning architectures.

### 1. JEGO Scan Integration
* **Static Scenes:** Achieved a **58% reduction in average RMSE**, proving that global awareness significantly enhances accuracy in stable environments.
* **Efficiency:** Experienced a **60% decrease in overall FPS**. 
* **Analysis:** DROID-SLAM's dense edge-pair structure is inherently sub-optimal for JEGO scan's original pairwise matching mechanism, causing significant bottlenecks in GPU optimization.

### 2. Mamba-based Update Operators
* **Efficiency:** Observed a **7% improvement in FPS**, demonstrating the computational efficiency of the Mamba blocks.
* **Dynamic Scenes:** Resulted in a **70% increase in RMSE**.
* **Analysis:** For fine-grained, pixel-wise residual corrections, maintaining a local receptive field (as in ConvGRU) is vastly more advantageous than relying on Mamba's global awareness.

### 💡 Core Takeaway
> **Superficial integration of novel models fails without a profound, underlying alignment between the baseline framework's geometric principles and the new architecture's mechanism.**
