"""EXAMPLE: Annotating the PBMC3k dataset"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import ssa_scrna as ssa

# Minimum genes detected per cell to keep obvious low-quality droplets.
MIN_GENES_TO_RETAIN = 200
# Minimum number of cells a gene must appear in to retain it.
MIN_CELLS_PER_GENE = 3
# High-end cutoffs are set by percentiles to trim extreme outliers.
QC_PERCENTILES = {
    "umi_hi": 99.5,
    "genes_hi": 99.5,
    "mito_hi": 95.0,
}
# Mitochondrial fraction ceiling; acts as a floor on the dynamic cutoff.
MITO_FLOOR = 20.0  # = 20% mitochondrial reads, a common conservative threshold for human PBMCs.

SEED_PLOT_KEYS = ["seeds_qcq", "seeds_otsu", "seeds_graph", "seeds_dpmm", "seeds_consensus"]
DPMM_PLOT_KEYS = ["seeds_dpmm", "seeds_dpmm_max_confidence", "seeds_dpmm_is_confident"]
PROP_PLOT_KEYS = ["prop_knn", "prop_rf", "prop_centroid", "labels_final"]
BASELINE_PLOT_KEYS = [
    "leiden_res0.2",
    "leiden_res0.4",
    "leiden_res0.6",
    "leiden_res0.8",
    "leiden_res1.0",
]
SAVE_OUTPUTS = True
OUTPUT_DIR = Path("examples/pbmc3k/outputs")
FIGURE_FORMAT = "png"

# Unified color palette for all cell types and labels.
# "unknown" is assigned a neutral gray (appears hollow when edge-colored).
UNIFIED_PALETTE = {
    # Cell type seeds
    "Myeloid": "#1f77b4",
    "Dendritic": "#ff7f0e",
    "Mast": "#2ca02c",
    "Keratinocyte": "#d62728",
    "Epithelial": "#9467bd",
    "Endothelial": "#8c564b",
    "Fibroblast": "#e377c2",
    "SmoothMuscle": "#7f7f7f",
    "Pericyte": "#bcbd22",
    "Melanocyte": "#17becf",
    "Schwann": "#1f77b4",
    "Adipocyte": "#aec7e8",
    "Hepatocyte": "#ffbb78",
    "SweatGland": "#98df8a",
    "Sebaceous": "#c5b0d5",
    # Unknown/unlabeled
    "unknown": "#cccccc",
    # Boolean fields (from is_confident, etc.)
    "True": "#2ca02c",
    "False": "#d62728",
}


def add_qc_gene_sets(adata: sc.AnnData) -> None:
    """Annotate mitochondrial, ribosomal, and hemoglobin gene flags."""
    vn_up = adata.var_names.str.upper()
    # Human and mouse 'mt-' both become 'MT-' after uppercasing.
    adata.var["mt"] = vn_up.str.startswith("MT-")
    adata.var["ribo"] = vn_up.str.startswith(("RPS", "RPL", "MRPS", "MRPL"))
    # Hemoglobin genes mark ambient RNA or erythrocyte contamination.
    adata.var["hb"] = adata.var_names.str.match(r"^(HB[ABEDM][A-Z0-9]*)", case=False)


def compute_qc_metrics(adata: sc.AnnData) -> None:
    """Compute standard QC metrics using the annotated gene sets."""
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )


def derive_thresholds(adata: sc.AnnData) -> tuple[float, float, float]:
    """Return data-driven thresholds for UMI, gene counts, and mito fraction."""
    # Keep the bulk while removing extreme high-count and high-gene outliers.
    umi_hi = np.percentile(adata.obs["total_counts"], QC_PERCENTILES["umi_hi"])
    genes_hi = np.percentile(adata.obs["n_genes_by_counts"], QC_PERCENTILES["genes_hi"])
    # Use a percentile-based mito cutoff, but never below a conservative floor.
    mito_hi = max(
        MITO_FLOOR,
        np.percentile(adata.obs["pct_counts_mt"], QC_PERCENTILES["mito_hi"]),
    )
    return umi_hi, genes_hi, mito_hi


def filter_cells(adata: sc.AnnData, umi_hi: float, genes_hi: float, mito_hi: float) -> sc.AnnData:
    """Filter cells by gene counts, UMI counts, and mitochondrial fraction."""
    # Combine gene/UMI/mito filters to remove low-quality and outlier cells.
    keep_cells = (
        (adata.obs["n_genes_by_counts"] >= MIN_GENES_TO_RETAIN)
        & (adata.obs["n_genes_by_counts"] <= genes_hi)
        & (adata.obs["total_counts"] <= umi_hi)
        & (adata.obs["pct_counts_mt"] <= mito_hi)
    )
    return adata[keep_cells].copy()


def preprocess_pbmc3k() -> sc.AnnData:
    """Load PBMC3k and run QC filtering plus normalization."""
    adata = sc.datasets.pbmc3k()

    print(f"START  | cells={adata.n_obs:,}  genes={adata.n_vars:,}")
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()  # avoids duplicate symbol problems in downstream tools
    adata.obs_names_make_unique()

    add_qc_gene_sets(adata)
    compute_qc_metrics(adata)

    print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())

    umi_hi, genes_hi, mito_hi = derive_thresholds(adata)
    print(
        "Thresholds -> min_genes="
        f"{MIN_GENES_TO_RETAIN}, umi_hi~{umi_hi:.0f}, genes_hi~{genes_hi:.0f}, mito_hi~{mito_hi:.1f}%"
    )

    before = adata.n_obs
    adata = filter_cells(adata, umi_hi, genes_hi, mito_hi)
    print(f"CELL FILTER | kept {adata.n_obs:,}/{before:,} ({adata.n_obs / before:.1%})")

    before_g = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    print(f"GENE FILTER | kept {adata.n_vars:,}/{before_g:,} (>= {MIN_CELLS_PER_GENE} cells)")

    # Library size normalization, then log-transform for downstream analysis.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    print(f"READY | cells={adata.n_obs:,} genes={adata.n_vars:,}")

    return adata


def prepare_features(adata: sc.AnnData, n_pcs: int = 50) -> sc.AnnData:
    """Compute PCA and neighbors for downstream SSA strategies."""
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
    return adata


def run_ssa_pipelines(adata: sc.AnnData) -> sc.AnnData:
    """Run Phase 1 seeding, consensus, Phase 2 propagation, and final consensus."""
    adata = prepare_features(adata)

    seed_strategies = {
        "seeds_qcq": ssa.strategies.QCQAdaptiveThresholding(markers=PBMC_MARKERS),
        "seeds_otsu": ssa.strategies.OtsuAdaptiveThresholding(markers=PBMC_MARKERS),
        "seeds_graph": ssa.strategies.GraphScorePropagation(markers=PBMC_MARKERS),
        "seeds_dpmm": ssa.strategies.DirichletProcessLabeling(markers=PBMC_MARKERS),
    }
    ssa.tl.label(adata, strategies=seed_strategies, n_jobs=4)

    consensus_seed = ssa.strategies.ConsensusVoting(
        keys=list(seed_strategies.keys()), majority_fraction=0.66
    )
    ssa.tl.label(adata, strategies=consensus_seed, key_added="seeds_consensus")

    propagation_strategies = {
        "prop_knn": ssa.strategies.KNNPropagation(seed_key="seeds_consensus"),
        "prop_rf": ssa.strategies.RandomForestPropagation(
            seed_key="seeds_consensus", random_state=0
        ),
        "prop_centroid": ssa.strategies.NearestCentroidPropagation(seed_key="seeds_consensus"),
    }
    ssa.tl.label(adata, strategies=propagation_strategies, n_jobs=3)

    final_consensus = ssa.strategies.ConsensusVoting(
        keys=list(propagation_strategies.keys()), majority_fraction=0.66
    )
    ssa.tl.label(adata, strategies=final_consensus, key_added="labels_final")

    run_baselines(adata)
    compute_ablation_metrics(adata)
    plot_ssa_umaps(
        adata,
        save_plots=SAVE_OUTPUTS,
        output_dir=OUTPUT_DIR,
    )
    return adata


def run_baselines(adata: sc.AnnData) -> None:
    """Compute canonical clustering baselines for comparison."""
    # Leiden with multiple resolutions for sensitivity analysis.
    for res in [0.2, 0.4, 0.6, 0.8, 1.0]:
        sc.tl.leiden(
            adata, key_added=f"leiden_res{res}", resolution=res, flavor="igraph", n_iterations=2
        )

    sc.tl.tsne(adata, n_pcs=50)


def compute_ablation_metrics(adata: sc.AnnData) -> None:
    """Compute clustering metrics (ARI, NMI) separately for seeds and propagated methods.

    Evaluates:
    - Seed methods vs. seed consensus
    - Propagation methods vs. final consensus
    - Baselines vs. final consensus
    """
    if "seeds_consensus" not in adata.obs:
        print("No consensus labels found; skipping ablation metrics.")
        return

    print("\n" + "=" * 70)
    print("ABLATION METRICS")
    print("=" * 70)

    # ========== PHASE 1: SEEDS ==========
    seed_consensus = adata.obs["seeds_consensus"]
    seed_methods = [f"seeds_{s}" for s in ["qcq", "otsu", "graph", "dpmm"]]

    seed_metrics = []
    for col in seed_methods:
        if col not in adata.obs:
            continue

        labels = adata.obs[col]
        if (labels == "unknown").all():
            continue

        valid = (labels != "unknown") & (seed_consensus != "unknown")
        if valid.sum() == 0:
            continue

        labels_valid = labels[valid]
        seed_consensus_valid = seed_consensus[valid]

        ari = adjusted_rand_score(seed_consensus_valid, labels_valid)
        nmi = normalized_mutual_info_score(seed_consensus_valid, labels_valid)

        seed_metrics.append({"Method": col, "ARI": ari, "NMI": nmi, "N_labeled": int(valid.sum())})

    if seed_metrics:
        seed_df = pd.DataFrame(seed_metrics)
        print("\n--- Phase 1: Seed Strategies vs. Seed Consensus ---")
        print(seed_df.to_string(index=False))
        print(f"Seed consensus labeled: {(seed_consensus != 'unknown').sum():,} cells")

    # ========== PHASE 2: PROPAGATION ==========
    if "labels_final" in adata.obs:
        final_consensus = adata.obs["labels_final"]
        prop_methods = [f"prop_{s}" for s in ["knn", "rf", "centroid"]]

        prop_metrics = []
        for col in prop_methods:
            if col not in adata.obs:
                continue

            labels = adata.obs[col]
            if (labels == "unknown").all():
                continue

            valid = (labels != "unknown") & (final_consensus != "unknown")
            if valid.sum() == 0:
                continue

            labels_valid = labels[valid]
            final_consensus_valid = final_consensus[valid]

            ari = adjusted_rand_score(final_consensus_valid, labels_valid)
            nmi = normalized_mutual_info_score(final_consensus_valid, labels_valid)

            prop_metrics.append(
                {"Method": col, "ARI": ari, "NMI": nmi, "N_labeled": int(valid.sum())}
            )

        if prop_metrics:
            prop_df = pd.DataFrame(prop_metrics)
            print("\n--- Phase 2: Propagation Methods vs. Final Consensus ---")
            print(prop_df.to_string(index=False))
            print(f"Final consensus labeled: {(final_consensus != 'unknown').sum():,} cells")

        # ========== BASELINES: LEIDEN ==========
        baseline_methods = [f"leiden_res{r}" for r in [0.2, 0.4, 0.6, 0.8, 1.0]]

        baseline_metrics = []
        for col in baseline_methods:
            if col not in adata.obs:
                continue

            labels = adata.obs[col]
            if (labels == "unknown").all():
                continue

            valid = (labels != "unknown") & (final_consensus != "unknown")
            if valid.sum() == 0:
                continue

            labels_valid = labels[valid]
            final_consensus_valid = final_consensus[valid]

            ari = adjusted_rand_score(final_consensus_valid, labels_valid)
            nmi = normalized_mutual_info_score(final_consensus_valid, labels_valid)

            baseline_metrics.append(
                {"Method": col, "ARI": ari, "NMI": nmi, "N_labeled": int(valid.sum())}
            )

        if baseline_metrics:
            baseline_df = pd.DataFrame(baseline_metrics)
            print("\n--- Baselines: Leiden vs. Final Consensus ---")
            print(baseline_df.to_string(index=False))

        # ========== SAVE COMBINED METRICS ==========
        all_metrics = seed_metrics + prop_metrics + baseline_metrics
        if all_metrics:
            combined_df = pd.DataFrame(all_metrics)
            metrics_output = OUTPUT_DIR / "ablation_metrics.csv"
            if SAVE_OUTPUTS:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                combined_df.to_csv(metrics_output, index=False)
                print(f"\nAll metrics saved to: {metrics_output}")

    print("=" * 70)


def _build_legend_handles(color_keys: list[str], adata: sc.AnnData, palette: dict | None) -> dict:
    """Build legend patches from data categories and colors.

    Args:
        color_keys: List of obs column names to extract categories from.
        adata: AnnData object containing the observations.
        palette: Optional dict mapping category names to colors. If None, uses tab20 colormap.

    Returns:
        Dict mapping category strings to matplotlib Patch handles.

    Note:
        Skips numerical columns to preserve their continuous visualization as colormaps.
    """
    handles_dict = {}
    for color_key in color_keys:
        if color_key not in adata.obs:
            continue

        col_data = adata.obs[color_key]

        # Skip numerical columns; let Scanpy handle their continuous colormaps
        if col_data.dtype in ("float64", "float32", "int64", "int32"):
            continue

        # Only process categorical or string columns
        if hasattr(col_data, "cat"):
            categories = col_data.cat.categories
        else:
            # For string/object columns, only create legend entries
            categories = sorted(col_data.unique())

        for cat in categories:
            cat_str = str(cat)
            if cat_str not in handles_dict:
                if palette is not None:
                    color = palette.get(cat_str, "#cccccc")
                else:
                    # Use categorical cmap if no palette specified
                    cmap = mpl.colormaps.get_cmap("tab20")
                    idx = list(categories).index(cat)
                    color = cmap(idx % 20)

                handles_dict[cat_str] = mpl.patches.Patch(facecolor=color)

    return handles_dict


def plot_ssa_umaps(
    adata: sc.AnnData,
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot UMAPs for seeds, propagation outputs, and final consensus labels.

    Uses a unified palette across all plots for consistent coloring.
    Removes individual subplot legends and adds a single shared legend per figure.
    """
    if save_plots:
        plot_dir = output_dir or OUTPUT_DIR
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Convert boolean columns to categorical strings for plotting
    for col in adata.obs.columns:
        if adata.obs[col].dtype == bool:
            adata.obs[col] = adata.obs[col].astype(str)

    sc.tl.umap(adata)

    # Helper function to save figure with optional consolidated legend
    def plot_and_save(color_keys, title_suffix, palette=UNIFIED_PALETTE, consolidate_legend=True):
        # When consolidating, suppress individual legends; otherwise use default legend rendering
        fig = sc.pl.umap(
            adata,
            color=color_keys,
            ncols=2,
            palette=palette,
            legend_loc=None if consolidate_legend else "right",
            return_fig=True,
        )

        # Only consolidate legend when comparing subplots with same cell types
        if consolidate_legend:
            # Remove all individual legends from subplots
            for ax in fig.axes:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            # Build legend from categories and colors
            handles_dict = _build_legend_handles(color_keys, adata, palette)

            # Add single shared legend to the figure
            if handles_dict:
                fig.legend(
                    handles_dict.values(),
                    handles_dict.keys(),
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=True,
                )

        if save_plots:
            fig.savefig(
                OUTPUT_DIR / f"umap{title_suffix}.{FIGURE_FORMAT}",
                bbox_inches="tight",
                dpi=100,
            )

    plot_and_save(SEED_PLOT_KEYS, "_seeds", consolidate_legend=True)
    plot_and_save(DPMM_PLOT_KEYS, "_dpmm", consolidate_legend=False)
    plot_and_save(PROP_PLOT_KEYS, "_prop", consolidate_legend=True)
    plot_and_save(BASELINE_PLOT_KEYS, "_baselines", palette=None, consolidate_legend=False)

    fig = sc.pl.tsne(
        adata,
        color=BASELINE_PLOT_KEYS,
        ncols=2,
        palette=None,
        legend_loc="right",
        return_fig=True,
    )

    # Default legends are rendered for each subplot

    if save_plots:
        fig.savefig(
            OUTPUT_DIR / f"tsne_baselines.{FIGURE_FORMAT}",
            bbox_inches="tight",
            dpi=100,
        )


PBMC_MARKERS = {
    "Myeloid": ["LYZ", "S100A8", "S100A9", "LST1", "ITGAM"],
    "Dendritic": ["ITGAX", "CD1C", "CLEC10A", "HLA-DRA", "CCR7"],
    "Mast": ["TPSAB1", "CPA3", "KIT", "MS4A2", "HDC"],
    "Keratinocyte": ["KRT1", "KRT5", "KRT14", "KRT10", "DSG1"],  # epithelial/skin
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7"],  # generic epithelial
    "Endothelial": ["PECAM1", "VWF", "KDR", "CDH5", "CLDN5"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "SmoothMuscle": ["ACTA2", "MYH11", "TAGLN", "CNN1", "CALD1"],
    "Pericyte": ["RGS5", "PDGFRB", "MCAM", "KCNJ8", "ABCC9"],
    "Melanocyte": ["PMEL", "MLANA", "TYR", "DCT", "MITF"],
    "Schwann": ["MPZ", "MBP", "PMP22", "SOX10", "PLP1"],
    "Adipocyte": ["ADIPOQ", "PLIN1", "CFD", "FABP4", "LEP"],
    "Hepatocyte": ["ALB", "TF", "APOB", "SERPINA1", "TTR"],
    "SweatGland": ["KRT7", "KRT18", "MUC1", "SCGB2A2", "AQP5"],
    "Sebaceous": ["SOX9", "KRT7", "KRT8", "KRT18", "MUC1"],
}


def main() -> None:
    """Run the PBMC3k preprocessing example."""
    adata = preprocess_pbmc3k()
    run_ssa_pipelines(adata)


if __name__ == "__main__":
    main()
