# `ssa-scRNA`: Strategy Architecture & Design Philosophy

`ssa-scRNA` (Semi-Supervised Annotation for Single-Cell RNA) is built around a highly modular, decoupled **Strategy Pattern**. Instead of locking users into a rigid, monolithic pipeline, the library treats every algorithmic step—from weak seed generation to final ensemble propagation—as an independent, interchangeable object adhering to the `BaseLabelingStrategy` contract.

This document details the expected behavior, mathematical foundations, and design considerations behind each implemented strategy.

---

## 🏗️ Core Architectural Principles

1. **The Unix Philosophy:** Each strategy does one thing perfectly. We do not enforce preprocessing (like PCA or scaling) inside the strategies. We trust the computational biologist to prepare their `AnnData` object correctly using `scanpy` standard practices before passing it to our estimators.
2. **Rich Data Transfer Objects (DTO):** Strategies do not mutate `adata` directly during computation. They return a `LabelingResult` object containing the categorical labels, `.obs` metrics (confidence, margins), `.obsm` matrices (continuous scores, posterior probabilities), and `.uns` metadata. The dispatcher (`tl.label`) handles the safe, sequential writing of this DTO back to the main thread.
3. **Concurrency:** Computations are embarrassingly parallel where possible. Batching multiple strategies (or fitting independent per-cell-type distributions in DPMM) is offloaded to a `ThreadPoolExecutor` to minimize latency.

---

## Phase 1: Seed Generation (Weak Labeling)

The goal of Phase 1 is to convert continuous marker gene expression into sparse, ultra-high-confidence categorical labels ("seeds"). The priority here is **precision over recall**; it is better to leave a cell as `"unknown"` than to generate a false positive seed.

### 1. QCQ Adaptive Thresholding (`QCQAdaptiveThresholding`)

**Concept:** Quality-Checked Quantile (QCQ) acts as a robust statistical heuristic. It scores cells based on marker signatures and strictly selects the top  percentile of expressors, provided they meet a minimum absolute quality threshold.

* **Expected Behavior:** Calls `sc.tl.score_genes` to generate robust scores (which correct for technical dropout and sequencing depth by subtracting random reference gene sets). It then calculates a dynamic threshold for *each* cell type based on the user-defined `quantile` (e.g., ).
* **Design Considerations:** * *Adaptive:* Hardcoded thresholds fail across different datasets. Quantiles scale dynamically with the dataset's specific expression distribution.
* *The QC Floor:* The `min_score` parameter prevents the algorithm from labeling the top 5% of a pure noise distribution if a cell type is simply absent from the manifold.



### 2. Otsu's Adaptive Thresholding (`OtsuAdaptiveThresholding`)

**Concept:** A parameter-free variance optimization technique borrowed from computer vision. It automatically finds the optimal threshold to separate a bimodal distribution into "Background" and "Signal".

* **Expected Behavior:** Computes a 1D histogram of marker scores for each cell type. It iteratively searches for a threshold bin  that maximizes the inter-class variance :



\sigma_b^2 = \omega_1(t) \omega_2(t) \left[ \mu_1(t) - \mu_2(t) \right]^2



Where  represents class probabilities and  represents class means.
* **Design Considerations:** * *Parameter-Free Splitting:* Unlike QCQ, which requires guessing a quantile, Otsu dynamically adapts to the size of the cluster. If 40% of the dataset is T-cells, Otsu will naturally set a threshold that captures the whole 40%, whereas a 95th-percentile QCQ would drop millions of valid cells.
* *Speed:* Implemented in pure, vectorized NumPy. It calculates the optimal split in milliseconds.



### 3. Graph Score Propagation (`GraphScorePropagation`)

**Concept:** Manifold diffusion. It leverages the global topology of the cell-cell -NN graph to smooth out local technical noise (dropout) in raw marker scores before gating the seeds.

* **Expected Behavior:** Extracts raw marker scores (), normalizes them to , and diffuses them over the symmetrically normalized adjacency matrix  via Personalized PageRank / Label Spreading:



Y_{t+1} = \alpha \hat{A} Y_t + (1-\alpha) Y_0



After  iterations, a "margin gate" is applied: a cell is only labeled if its highest diffused score beats its second-highest diffused score by a minimum `margin`.
* **Design Considerations:** * *Lightweight Math:* Explicitly avoids heavy deep learning frameworks (PyTorch/GCN) in favor of fast, deterministic `scipy.sparse` matrix multiplication.
* *The Margin Gate:* Crucial for resolving ambiguous cells positioned on the boundaries between closely related lineages in the latent space.



### 4. Dirichlet Process Mixture Models (`DirichletProcessLabeling`)

**Concept:** Probabilistic confidence bounds. It models the multi-dimensional marker expression of each cell type as a mixture of two Gaussian distributions (Signal vs. Noise) and uses Variational Inference to calculate the posterior probability of a cell belonging to the Signal component.

* **Expected Behavior:** Slices the log-normalized data into  matrices (where  is the number of markers for a specific cell type). Fits a Bayesian Gaussian Mixture Model. The Dirichlet Process prior () automatically handles complexity—if the distribution is unimodal (pure noise), it effectively zeroes out the second component.
* **Design Considerations:**
* *True Probabilities:* Yields rigorous mathematical confidence bounds () rather than arbitrary continuous scores, mapped directly to `.obsm['posterior_probabilities']`.
* *Threaded Execution:* Variational Inference is computationally expensive. This strategy internally maps the independent cell-type BGM fits to a `ThreadPoolExecutor`, drastically cutting execution time.



---

## Phase 1.5: The Aggregator

### Consensus Voting (`ConsensusVoting`)

**Concept:** The bridge between Phase 1 and Phase 2. It takes the independent, conflicting opinions of multiple weak labelers and distills them into a single, high-fidelity ground truth vector.

* **Expected Behavior:** Performs a row-wise evaluation across  label columns. It filters out abstentions (`"unknown"`). If the most frequent remaining label meets the `majority_fraction` (e.g.,  for a 2/3 supermajority), it becomes the consensus seed. Otherwise, the cell reverts to `"unknown"`.
* **Design Considerations:** * *Robustness via Ensemble:* Averages out the algorithmic biases of the individual Phase 1 strategies.
* *Polymorphic Use:* This strategy is agnostic to *what* it is ensembling. It is used to aggregate Phase 1 weak labels, but is seamlessly reused at the very end of the pipeline to aggregate Phase 2 ML predictions.



---

## Phase 2: Label Propagation (Generalization)

The goal of Phase 2 is **recall**. Taking the sparse  of cells labeled in Phase 1 (the seeds), these strategies train classical Machine Learning algorithms to classify the remaining unstructured manifold.

**Shared Design Considerations for Phase 2:**

* **The `obsm_key` Paradigm:** Phase 2 strategies do not train on raw genes ( dimensions). They train on compressed feature spaces to avoid the curse of dimensionality. By making `obsm_key` dynamic (defaulting to `"X_pca"`), the library natively supports standard PCA as well as modern deep-learning autoencoders (e.g., passing `"X_scVI"`).
* **`sklearn` Isolation:** `scikit-learn` is treated as a hard dependency, but is isolated entirely to the `strategies.ml` modules.

### 1. k-Nearest Neighbors (`KNNPropagation`)

* **Concept:** Local topological propagation.
* **Expected Behavior:** Fits a -NN classifier on the coordinates of the seed cells. It predicts the labels of unknown cells based on a distance-weighted majority vote of their  closest seed neighbors in the PCA/latent space. Highly effective for resolving non-linear, branching trajectories.

### 2. Random Forest (`RandomForestPropagation`)

* **Concept:** Non-linear decision boundaries.
* **Expected Behavior:** Fits an ensemble of decision trees on the seed cells.
* **Design Considerations:** Highly robust to noisy, unscaled latent spaces. Capable of outputting rigorous `.obsm['probabilities']` via the fraction of trees voting for a specific class, providing an excellent confidence metric for downstream filtering.

### 3. Nearest Centroid (`NearestCentroidPropagation`)

* **Concept:** Distance to the cluster mean.
* **Expected Behavior:** Calculates the exact geometric center (centroid) of the seed cells for each cell type. Assigns unknown cells to the class of the closest centroid.
* **Design Considerations:** The fastest and simplest algorithm. Acts as a baseline. It assumes clusters are convex and spherical in the latent space; while less accurate for complex topologies, its inclusion in a final Consensus Vote anchors the ensemble against wild extrapolations by the Random Forest.
