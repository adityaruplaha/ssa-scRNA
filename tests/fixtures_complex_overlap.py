"""Complex overlap fixture definitions for conftest.py"""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData


@pytest.fixture
def adata_ubiquitous_shared():
    """
    Create AnnData with ubiquitous (house-keeping) genes shared across ALL cell types.

    Structure:
    - Genes 0-4: Ubiquitous housekeeping genes (present in all 8 classes)
    - Genes 5-7: Class A-specific
    - Genes 8-10: Class B-specific
    etc.
    """
    np.random.seed(43)
    n_cells, n_genes = 1000, 80

    X = np.random.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)
    X[:, 0:5] += np.random.poisson(lam=6.0, size=(n_cells, 5)).astype(np.float32)  # High ubiquitous

    # Class A-H with specific genes
    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 5),
            (125, 250, 8),
            (250, 375, 11),
            (375, 500, 14),
            (500, 625, 17),
            (625, 750, 20),
            (750, 875, 23),
            (875, 1000, 26),
        ]
    ):
        gene_end = gene_start + 3
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=5.0, size=(end_cell - start_cell, 3)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=43)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=43)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_ubiquitous():
    """Marker dict with ubiquitous genes shared by all + specific genes."""
    ubiquitous = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"]
    return {
        "Class A": ubiquitous + ["gene_5", "gene_6", "gene_7"],
        "Class B": ubiquitous + ["gene_8", "gene_9", "gene_10"],
        "Class C": ubiquitous + ["gene_11", "gene_12", "gene_13"],
        "Class D": ubiquitous + ["gene_14", "gene_15", "gene_16"],
        "Class E": ubiquitous + ["gene_17", "gene_18", "gene_19"],
        "Class F": ubiquitous + ["gene_20", "gene_21", "gene_22"],
        "Class G": ubiquitous + ["gene_23", "gene_24", "gene_25"],
        "Class H": ubiquitous + ["gene_26", "gene_27", "gene_28"],
    }


@pytest.fixture
def adata_highly_specific():
    """
    Create AnnData with completely non-overlapping marker genes per class.
    """
    np.random.seed(44)
    n_cells, n_genes = 1000, 80
    X = np.random.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)

    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 0),
            (125, 250, 4),
            (250, 375, 8),
            (375, 500, 12),
            (500, 625, 16),
            (625, 750, 20),
            (750, 875, 24),
            (875, 1000, 28),
        ]
    ):
        gene_end = gene_start + 4
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=6.0, size=(end_cell - start_cell, 4)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=44)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=44)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_highly_specific():
    """Marker dict with completely non-overlapping genes."""
    return {
        "Class A": ["gene_0", "gene_1", "gene_2", "gene_3"],
        "Class B": ["gene_4", "gene_5", "gene_6", "gene_7"],
        "Class C": ["gene_8", "gene_9", "gene_10", "gene_11"],
        "Class D": ["gene_12", "gene_13", "gene_14", "gene_15"],
        "Class E": ["gene_16", "gene_17", "gene_18", "gene_19"],
        "Class F": ["gene_20", "gene_21", "gene_22", "gene_23"],
        "Class G": ["gene_24", "gene_25", "gene_26", "gene_27"],
        "Class H": ["gene_28", "gene_29", "gene_30", "gene_31"],
    }


@pytest.fixture
def adata_hierarchical_overlap():
    """
    Create AnnData with hierarchical overlap structure.
    Simulates biological relationships at multiple levels.
    """
    np.random.seed(45)
    n_cells, n_genes = 1000, 80
    X = np.random.poisson(lam=0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Root: shared by all
    X[:, 0:3] += np.random.poisson(lam=3.0, size=(n_cells, 3)).astype(np.float32)
    # Level 1A: shared by A, B, C, D
    X[0:500, 3:6] += np.random.poisson(lam=3.0, size=(500, 3)).astype(np.float32)
    # Level 1B: shared by E, F, G, H
    X[500:1000, 6:9] += np.random.poisson(lam=3.0, size=(500, 3)).astype(np.float32)
    # Level 2 pairs
    X[0:250, 9:12] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)  # A-B
    X[250:500, 12:15] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)  # C-D
    X[500:750, 15:18] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)  # E-F
    X[750:1000, 18:21] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)  # G-H
    # Leaf: specific
    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 21),
            (125, 250, 24),
            (250, 375, 27),
            (375, 500, 30),
            (500, 625, 33),
            (625, 750, 36),
            (750, 875, 39),
            (875, 1000, 42),
        ]
    ):
        gene_end = gene_start + 3
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=5.0, size=(end_cell - start_cell, 3)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=45)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=45)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_hierarchical():
    """Marker dict with hierarchical overlap structure."""
    return {
        "Class A": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_21",
            "gene_22",
            "gene_23",
        ],
        "Class B": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_24",
            "gene_25",
            "gene_26",
        ],
        "Class C": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_27",
            "gene_28",
            "gene_29",
        ],
        "Class D": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_30",
            "gene_31",
            "gene_32",
        ],
        "Class E": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_15",
            "gene_16",
            "gene_17",
            "gene_33",
            "gene_34",
            "gene_35",
        ],
        "Class F": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_15",
            "gene_16",
            "gene_17",
            "gene_36",
            "gene_37",
            "gene_38",
        ],
        "Class G": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_18",
            "gene_19",
            "gene_20",
            "gene_39",
            "gene_40",
            "gene_41",
        ],
        "Class H": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_18",
            "gene_19",
            "gene_20",
            "gene_42",
            "gene_43",
            "gene_44",
        ],
    }


@pytest.fixture
def adata_complex_mixed():
    """
    Create AnnData with complex mixed overlap patterns.
    Simulates realistic biological scenarios.
    """
    np.random.seed(46)
    n_cells, n_genes = 1000, 80
    X = np.random.poisson(lam=0.4, size=(n_cells, n_genes)).astype(np.float32)

    # Core ubiquitous
    X[:, 0:3] += np.random.poisson(lam=4.0, size=(n_cells, 3)).astype(np.float32)
    # Frequent background (partial)
    background_mask = np.random.rand(n_cells) > 0.3
    X[background_mask, 3:6] += np.random.poisson(lam=2.0, size=(background_mask.sum(), 3)).astype(
        np.float32
    )
    # Shared cross pattern
    X[np.concatenate([np.arange(0, 250), np.arange(500, 750)]), 6:9] += np.random.poisson(
        lam=3.0, size=(500, 3)
    ).astype(np.float32)
    # Pairwise shared
    for _, (start, end) in enumerate([(0, 250), (250, 500), (500, 750), (750, 1000)]):
        X[start:end, 9:12] += np.random.poisson(lam=3.0, size=(end - start, 3)).astype(np.float32)
    # Triplet shared (A, B, C)
    X[0:375, 12:15] += np.random.poisson(lam=2.0, size=(375, 3)).astype(np.float32)
    # Class-specific
    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 15),
            (125, 250, 18),
            (250, 375, 21),
            (375, 500, 24),
            (500, 625, 27),
            (625, 750, 30),
            (750, 875, 33),
            (875, 1000, 36),
        ]
    ):
        gene_end = gene_start + 3
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=5.5, size=(end_cell - start_cell, 3)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=46)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=46)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_complex_mixed():
    """Marker dict with complex mixed overlap patterns."""
    return {
        "Class A": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_15",
            "gene_16",
            "gene_17",
        ],
        "Class B": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_18",
            "gene_19",
            "gene_20",
        ],
        "Class C": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_21",
            "gene_22",
            "gene_23",
        ],
        "Class D": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_24",
            "gene_25",
            "gene_26",
        ],
        "Class E": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_27",
            "gene_28",
            "gene_29",
        ],
        "Class F": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_5",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_30",
            "gene_31",
            "gene_32",
        ],
        "Class G": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_33",
            "gene_34",
            "gene_35",
        ],
        "Class H": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_36",
            "gene_37",
            "gene_38",
        ],
    }
