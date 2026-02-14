import scanpy as sc

# Load the PBMC 68k dataset
adata = sc.read_10x_mtx(
    "../pbmc68k_10x/filtered_matrices_mex/hg19/",  # Path to the directory
    var_names="gene_symbols",  # Use symbols (e.g., GAPDH) as names
    cache=True,  # Optimization: Saves a binary .h5ad cache
)

# Mandatory Step: Handle duplicate gene names
adata.var_names_make_unique()
adata.obs_names_make_unique()
adata.var_names = adata.var_names.astype(str)
vn_up = adata.var_names.str.upper()

# Print the AnnData dimensions to verify successful loading
print(f"START | cells={adata.n_obs:,}  genes={adata.n_vars:,}")

# --- Define QC gene sets and SAVE as columns in adata.var ---
# Human and mouse 'mt-' both become 'MT-' after uppercasing
adata.var["mt"] = vn_up.str.startswith("MT-")
adata.var["ribo"] = vn_up.str.startswith(("RPS", "RPL", "MRPS", "MRPL"))
adata.var["hb"] = adata.var_names.str.match(r"^(HB[ABEDM][A-Z0-9]*)", case=False)

# --- Compute QC metrics using those var columns ---
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt", "ribo", "hb"],  # <- names of columns in adata.var
    percent_top=None,
    log1p=False,
    inplace=True,
)
# --- Print summary statistics of some key QC metrics ---
print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())
