from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.mixture import BayesianGaussianMixture

from .base import BaseLabelingStrategy, LabelingResult


class DirichletProcessLabeling(BaseLabelingStrategy):
    r"""
    Multivariate Dirichlet Process Mixture Model (DPMM) for Seed Generation.

    This strategy models the joint expression distribution of marker genes for each
    cell type as a multivariate mixture of "Signal" and "Background" components.
    By using a Dirichlet Process prior, it infers the parameters of these
    distributions and assigns a posterior probability (confidence bound) that a
    cell truly expresses the signature.

    Note: This strategy assumes that the input data (`adata.X` or `adata.raw.X`)
    is already log-normalized.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    min_confidence : float, default 0.8
        The minimum posterior probability required to assign a label.
    weight_concentration_prior : float, default 1e-3
        The Dirichlet Process parameter ($\gamma$). Lower values favor fewer
        components (e.g., strongly enforces a binary Signal vs Noise split).
    max_iter : int, default 500
        Maximum number of iterations for the Variational Inference EM algorithm.
    use_raw : bool, default True
        Whether to extract marker expression from `adata.raw`.
    random_state : int, optional
        Seed for the random number generator to ensure reproducible DPMM convergence.
    n_jobs : int, default 4
        Number of threads to use for fitting cell type distributions in parallel.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        min_confidence: float = 0.80,
        weight_concentration_prior: float = 1e-3,
        max_iter: int = 500,
        use_raw: bool = True,
        random_state: Optional[int] = None,
        n_jobs: int = 4,
        **kwargs,
    ):
        self.markers = markers
        self.min_confidence = min_confidence
        self.weight_concentration_prior = weight_concentration_prior
        self.max_iter = max_iter
        self.use_raw = use_raw
        self.random_state = random_state
        self.n_jobs = n_jobs

    @property
    def name(self) -> str:
        return "dpmm"

    def _fit_single_ctype(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fits a Multivariate DPMM for a single cell type."""
        # Edge case: No data or completely zeroed data
        if X.shape[1] == 0 or np.all(X == 0):
            return np.zeros(X.shape[0]), {
                "signal_mean_sum": 0.0,
                "noise_mean_sum": 0.0,
                "converged": True,
            }

        # Fit 2-component BGM with Dirichlet Process Prior
        bgm = BayesianGaussianMixture(
            n_components=2,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=self.weight_concentration_prior,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        bgm.fit(X)

        # Predict posterior probabilities (N_cells x 2)
        probs = bgm.predict_proba(X)

        # Identify "Signal" component
        # bgm.means_ shape is (n_components, n_features)
        # We define the signal as the component with the highest overall marker expression
        component_sums = bgm.means_.sum(axis=1)
        signal_idx = np.argmax(component_sums)
        noise_idx = 1 - signal_idx

        stats = {
            "signal_mean_sum": float(component_sums[signal_idx]),
            "noise_mean_sum": float(component_sums[noise_idx]),
            "converged": bool(bgm.converged_),
        }

        # Return probability of belonging to the signal component
        return probs[:, signal_idx], stats

    def execute_on(self, adata: AnnData) -> LabelingResult:
        cell_types = list(self.markers.keys())

        # Data structures to hold our DPMM outputs
        posteriors_df = pd.DataFrame(0.0, index=adata.obs_names, columns=cell_types)
        dpmm_stats = {}

        # 1. Prepare expression matrices for each cell type
        # We do this sequentially to safely interact with AnnData, then pass numpy arrays to threads
        ctype_matrices = {}
        for ctype, genes in self.markers.items():
            valid_genes = [
                g
                for g in genes
                if g in adata.var_names
                or (self.use_raw and adata.raw is not None and g in adata.raw.var_names)
            ]

            if not valid_genes:
                ctype_matrices[ctype] = np.zeros((adata.n_obs, 0))
                continue

            if self.use_raw and adata.raw is not None:
                X_slice = adata.raw[:, valid_genes].X
            else:
                X_slice = adata[:, valid_genes].X

            # BGM requires dense arrays
            if sp.issparse(X_slice):
                X_slice = X_slice.toarray()

            ctype_matrices[ctype] = X_slice

        # 2. Fit Multivariate DPMM per cell type (IN PARALLEL)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Future -> Cell Type mapping
            future_to_ctype = {
                executor.submit(self._fit_single_ctype, ctype_matrices[ctype]): ctype
                for ctype in cell_types
            }

            for future in future_to_ctype:
                ctype = future_to_ctype[future]
                try:
                    posterior_probs, stats = future.result()
                    posteriors_df[ctype] = posterior_probs
                    dpmm_stats[ctype] = stats

                except Exception as e:
                    print(f"DPMM fitting failed for '{ctype}': {e}")
                    posteriors_df[ctype] = 0.0
                    dpmm_stats[ctype] = {"converged": False, "error": str(e)}

        # 3. Label Assignment
        final_labels = pd.Series("unknown", index=adata.obs_names)

        max_probs = posteriors_df.max(axis=1)
        best_matches = posteriors_df.idxmax(axis=1)

        is_confident = max_probs >= self.min_confidence

        final_labels[is_confident] = best_matches[is_confident]

        # 4. Return Rich DTO
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={"max_confidence": max_probs, "is_confident": is_confident},
            obsm={"posterior_probabilities": posteriors_df},
            uns={
                "fraction_assigned": float(is_confident.mean()),
                "dpmm_convergence_stats": dpmm_stats,
            },
        )
