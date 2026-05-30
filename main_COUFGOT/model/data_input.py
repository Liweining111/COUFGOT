import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from collections import Counter
import random
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD


def sample_cells_by_celltype(cty_df, rna_hvg_df, target_n_cells=None):
    """
    Sample cells according to cell type proportions while maintaining type distribution.
    Parameters:
    - cty_df: DataFrame, containing cell type information, index as cell ID, 'cell_type' column as type.
    - rna_hvg_df: DataFrame, RNA HVG data, index as cell ID.
    - target_n_cells: int, optional, target number of cells. If None or greater than current cell count, no sampling.

    Returns:
    - tuple: (rna_hvg_df_sampled, adt_df_sampled, cty_df_sampled)
    """
    if target_n_cells is not None and target_n_cells < len(cty_df):
        print(f"Sampling to {target_n_cells} cells while maintaining cell type proportions...")
        type_counts = Counter(cty_df['cell_type'])
        total_cells = len(cty_df)
        selected_cells = []
        for cell_type, count in type_counts.items():
            frac = target_n_cells / total_cells
            n_sample = max(1, int(round(count * frac)))  # Round to nearest integer, at least 1
            type_cells = cty_df[cty_df['cell_type'] == cell_type].index.tolist()
            sampled = np.random.choice(type_cells, n_sample, replace=False).tolist()
            selected_cells.extend(sampled)
        # Ensure selected cell count does not exceed target_n_cells
        if len(selected_cells) > target_n_cells:
            selected_cells = np.random.choice(selected_cells, target_n_cells, replace=False).tolist()
        # Filter all DataFrames
        rna_hvg_df_sampled = rna_hvg_df.loc[selected_cells]
        cty_df_sampled = cty_df.loc[selected_cells]
        print(f"Number of cells after sampling: {len(selected_cells)}")
        # Print type distribution after sampling
        sampled_counts = Counter(cty_df_sampled['cell_type'])
        print("Cell type distribution after sampling:", dict(sampled_counts))
    else:
        print("No cell sampling performed, using all cells.")
        rna_hvg_df_sampled = rna_hvg_df
        cty_df_sampled = cty_df

    return rna_hvg_df_sampled, cty_df_sampled
def load_horizonal_data(base_dir, rna_filename_1, rna_filename_2, cty_filename_1,cty_filename_2,\
                        n_hvg=2000,QC_min_genes=500, QC_min_cells = 50, target_n_cells=2000):
    """
    Read RNA1, RNA2 H5 data and cell type CSV, return DataFrames
    Parameters:
    base_dir: str, common folder path
    rna1_filename: str, RNA h5 filename
    rna2_filename: str, RNA h5 filename
    cty_filename: str, cell type csv filename
    Returns:
    rna1_df, rna2_df, cty_1, cty_2: pandas DataFrame
    """
    # Concatenate full file paths
    rna_file_1 = os.path.join(base_dir, rna_filename_1)
    rna_file_2 = os.path.join(base_dir, rna_filename_2)
    cty_file_1 = os.path.join(base_dir, cty_filename_1)
    cty_file_2 = os.path.join(base_dir, cty_filename_2)
    # Read H5 files
    rna_X = h5py.File(rna_file_1, "r")
    rna_Y = h5py.File(rna_file_2, "r")

    # Read row names (cell barcodes)
    rna_rows_x = [x.decode() for x in rna_X["matrix"]["barcodes"][:]]
    rna_rows_y = [x.decode() for x in rna_Y["matrix"]["barcodes"][:]]
    print("Number of cells: RNA1={}, RNA2={}".format(len(rna_rows_x), len(rna_rows_y)))
    print("Are row names exactly the same:", rna_rows_x == rna_rows_y)
    # Read column names (features)
    rna_cols_x = [x.decode() for x in rna_X["matrix"]["features"][:]]
    rna_cols_y = [x.decode() for x in rna_Y["matrix"]["features"][:]]
    print("Number of columns: RNA1={}, RNA2={}".format(len(rna_cols_x), len(rna_cols_y)))
    print("Are column names exactly the same:", rna_cols_x == rna_cols_y)

    # Read matrix and transpose to (cells x features)
    rna_x = np.array(rna_X["matrix"]["data"])
    rna_x_df = pd.DataFrame(rna_x.T, index=rna_rows_x, columns=rna_cols_x)
    print("RNA1 raw data dimensions:", rna_x_df.shape)

    rna_y = np.array(rna_Y["matrix"]["data"])
    rna_y_df = pd.DataFrame(rna_y.T, index=rna_rows_y, columns=rna_cols_y)
    print("RNA2 raw data dimensions:", rna_y_df.shape)
    # ======================================================
    # QC---Filter low-quality cells
    # Count number of genes expressed per cell
    x_gene_counts = (rna_x_df > 0).sum(axis=1)
    y_gene_counts = (rna_y_df > 0).sum(axis=1)
    rna_x_df = rna_x_df.loc[x_gene_counts >= QC_min_genes, :]
    rna_y_df = rna_y_df.loc[y_gene_counts >= QC_min_genes, :]
    print(f"RNA1 cell count after QC: {rna_x_df.shape[0]}")
    print(f"RNA2 cell count after QC: {rna_y_df.shape[0]}")
    # ======================================================
    # QC---Filter very low expression genes
    all_data = pd.concat([rna_x_df, rna_y_df], axis=0)
    gene_filter = (all_data > 0).sum(axis=0) >= QC_min_cells
    all_data = all_data.loc[:, gene_filter]
    print(f"Remaining genes after filtering: {all_data.shape[1]}")
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data.fillna(0, inplace=True)
    # RNA highly variable gene selection
    adata_rna = sc.AnnData(all_data)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_hvg, flavor='seurat')
    hvg = adata_rna.var[adata_rna.var["highly_variable"]].index

    adata_rna = adata_rna[:, hvg]
    # Split back into two batches (order consistent with concat)
    X_hvg = adata_rna.X[:rna_x_df.shape[0], :]
    Y_hvg = adata_rna.X[rna_x_df.shape[0]:, :]
    # Convert back to DataFrame for subsequent use
    X_hvg = pd.DataFrame(X_hvg, index=rna_x_df.index, columns=hvg)
    Y_hvg = pd.DataFrame(Y_hvg, index=rna_y_df.index, columns=hvg)

    # Read cell type CSV and set index and column names
    cty_1 = pd.read_csv(cty_file_1, header=0)
    cty_1.index = rna_rows_x  # Default order correspondence
    cty_1.columns = ['cell_type']
    common_cells = X_hvg.index.intersection(cty_1.index)
    X_hvg = X_hvg.loc[common_cells]
    cty_1 = cty_1.loc[common_cells]
    print("Number of RNA1 cell samples:", len(cty_1))

    cty_2 = pd.read_csv(cty_file_2, header=0)
    cty_2.index = rna_rows_y  # Default order correspondence
    cty_2.columns = ['cell_type']
    common_cells = Y_hvg.index.intersection(cty_2.index)
    Y_hvg = Y_hvg.loc[common_cells]
    cty_2 = cty_2.loc[common_cells]
    print("Number of RNA2 cell types:", len(cty_2))

    # Sample cells by cell type
    X_hvg, cty_1 = sample_cells_by_celltype(
        cty_df=cty_1,
        rna_hvg_df=X_hvg,
        target_n_cells=target_n_cells
    )
    Y_hvg, cty_2 = sample_cells_by_celltype(
        cty_df=cty_2,
        rna_hvg_df=Y_hvg,
        target_n_cells=target_n_cells
    )
    return X_hvg, Y_hvg, cty_1, cty_2

def load_rna_adt_vertical(base_dir, adt_filename, rna_filename, cty_filename,n_hvg=2000, target_n_cells=None, random_seed=123):
    """
    Read ADT, RNA H5 data and cell type CSV, return DataFrames
    Parameters:
    base_dir: str, common folder path
    adt_filename: str, ADT h5 filename
    rna_filename: str, RNA h5 filename
    cty_filename: str, cell type csv filename
    Returns:
    adt_df, rna_df, cty_df: pandas DataFrame
    """
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Concatenate full file paths
    adt_file = os.path.join(base_dir, adt_filename)
    rna_file = os.path.join(base_dir, rna_filename)
    cty_file = os.path.join(base_dir, cty_filename)

    # Read H5 files
    adt = h5py.File(adt_file, "r")
    rna = h5py.File(rna_file, "r")

    # Read row names (cell barcodes)
    adt_rows = [x.decode() for x in adt["matrix"]["barcodes"][:]]
    rna_rows = [x.decode() for x in rna["matrix"]["barcodes"][:]]
    print("Number of cells: ADT={}, RNA={}".format(len(adt_rows), len(rna_rows)))
    print("Are row names exactly the same:", adt_rows == rna_rows)

    # Read column names (features)
    adt_cols = [x.decode() for x in adt["matrix"]["features"][:]]
    rna_cols = [x.decode() for x in rna["matrix"]["features"][:]]
    print("Number of columns: ADT={}, RNA={}".format(len(adt_cols), len(rna_cols)))
    print("Are column names exactly the same:", adt_cols == rna_cols)

    # Read matrix and transpose to (cells x features)
    adt_X = np.array(adt["matrix"]["data"])
    adt_df = pd.DataFrame(adt_X.T, index=adt_rows, columns=adt_cols)
    print("ADT data dimensions:", adt_df.shape)

    rna_X = np.array(rna["matrix"]["data"])
    rna_df = pd.DataFrame(rna_X.T, index=rna_rows, columns=rna_cols)
    print("RNA raw data dimensions:", rna_df.shape)
    rna_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rna_df.fillna(0, inplace=True)
    # RNA highly variable gene selection
    adata_rna = sc.AnnData(rna_df)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_hvg, flavor='seurat')
    rna_hvg_df = pd.DataFrame(adata_rna[:, adata_rna.var['highly_variable']].X,
                              index=rna_df.index,
                              columns=adata_rna.var_names[adata_rna.var['highly_variable']])
    print("RNA highly variable gene matrix dimensions:", rna_hvg_df.shape)

    # Read cell type CSV and set index and column names
    cty_df = pd.read_csv(cty_file, header=0)
    cty_df.index = adt_rows  # Default order correspondence
    cty_df.columns = ['cell_type']
    print("Number of cell type rows:", len(cty_df))

    # Sample cells according to cell type proportions
    if target_n_cells is not None and target_n_cells < len(cty_df):
        print(f"Sampling to {target_n_cells} cells while maintaining cell type proportions...")
        type_counts = Counter(cty_df['cell_type'])
        total_cells = len(cty_df)
        selected_cells = []
        for cell_type, count in type_counts.items():
            frac = target_n_cells / total_cells
            n_sample = max(1, int(round(count * frac)))  # Round to nearest integer, at least 1
            type_cells = cty_df[cty_df['cell_type'] == cell_type].index.tolist()
            sampled = np.random.choice(type_cells, n_sample, replace=False).tolist()
            selected_cells.extend(sampled)
        # Ensure selected cell count does not exceed target_n_cells
        if len(selected_cells) > target_n_cells:
            selected_cells = np.random.choice(selected_cells, target_n_cells, replace=False).tolist()
        # Filter all DataFrames
        rna_hvg_df = rna_hvg_df.loc[selected_cells]
        adt_df = adt_df.loc[selected_cells]
        cty_df = cty_df.loc[selected_cells]

        print(f"Number of cells after sampling: {len(selected_cells)}")
        # Print type distribution after sampling
        sampled_counts = Counter(cty_df['cell_type'])
        print("Cell type distribution after sampling:", dict(sampled_counts))
    else:
        print("No cell sampling performed, using all cells.")
    return rna_hvg_df, adt_df, cty_df


def load_rna_atac_vertical(base_dir, atac_filename, rna_filename, cty_filename,
                           n_hvg=2000, n_lsi=50, target_n_cells=None, random_seed=123):
    """
    Read RNA, scATAC H5 data and cell type CSV, reduce dimensions for vertical alignment, return integrated DataFrame
    Parameters:
    base_dir: str, data folder
    atac_filename: str, scATAC h5 filename
    rna_filename: str, RNA h5 filename
    cty_filename: str, cell type csv filename
    n_hvg: int, number of RNA highly variable genes
    n_lsi: int, number of ATAC LSI components
    target_n_cells: int, target total number of cells (if specified, randomly sample to this count maintaining proportions; if None, no sampling)
    random_seed: int, random seed for reproducibility
    Returns:
    rna_hvg_df: pandas DataFrame, RNA highly variable gene matrix (filtered)
    atac_lsi_df: pandas DataFrame, ATAC LSI matrix (filtered)
    cty_df: pandas DataFrame, cell types (filtered)
    """
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Concatenate paths
    atac_file = os.path.join(base_dir, atac_filename)
    rna_file = os.path.join(base_dir, rna_filename)
    cty_file = os.path.join(base_dir, cty_filename)

    # Read H5 files
    atac = h5py.File(atac_file, "r")
    rna = h5py.File(rna_file, "r")
    # Read row names (cells)
    atac_rows = [x.decode() for x in atac["matrix"]["barcodes"][:]]
    rna_rows = [x.decode() for x in rna["matrix"]["barcodes"][:]]
    print(f"Number of cells: ATAC={len(atac_rows)}, RNA={len(rna_rows)}")
    print("Are row names consistent:", atac_rows == rna_rows)

    # Read column names (features)
    atac_cols = [x.decode() for x in atac["matrix"]["features"][:]]
    rna_cols = [x.decode() for x in rna["matrix"]["features"][:]]
    print(f"Number of columns: ATAC={len(atac_cols)}, RNA={len(rna_cols)}")

    # Read matrix and transpose to (cells x features)
    atac_X = np.array(atac["matrix"]["data"]).T  # Transpose
    atac_df = pd.DataFrame(atac_X, index=atac_rows, columns=atac_cols)
    print("ATAC raw dimensions:", atac_df.shape)

    rna_X = np.array(rna["matrix"]["data"]).T
    rna_df = pd.DataFrame(rna_X, index=rna_rows, columns=rna_cols)
    rna_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rna_df.fillna(0, inplace=True)
    print("RNA raw dimensions:", rna_df.shape)

    # RNA highly variable gene selection
    adata_rna = sc.AnnData(rna_df)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_hvg, flavor='seurat')
    rna_hvg_df = pd.DataFrame(adata_rna[:, adata_rna.var['highly_variable']].X,
                              index=rna_df.index,
                              columns=adata_rna.var_names[adata_rna.var['highly_variable']])
    print("RNA highly variable gene matrix dimensions:", rna_hvg_df.shape)

    # ATAC feature selection: TF-IDF + LSI
    # Calculate TF
    tf = atac_df.div(atac_df.sum(axis=1), axis=0)
    # Calculate IDF
    idf = np.log(1 + atac_df.shape[0] / (1 + (atac_df > 0).sum(axis=0)))
    tf_idf = tf * idf
    # LSI (SVD)
    svd = TruncatedSVD(n_components=n_lsi, random_state=random_seed)
    atac_lsi = svd.fit_transform(tf_idf)
    atac_lsi_df = pd.DataFrame(atac_lsi, index=atac_df.index,
                               columns=[f'LSI_{i+1}' for i in range(atac_lsi.shape[1])])
    print("ATAC LSI matrix dimensions:", atac_lsi_df.shape)

    # Read cell types
    cty_df = pd.read_csv(cty_file, header=0)
    cty_df.index = atac_rows  # Default order correspondence
    cty_df.columns = ['cell_type']
    print("Number of cell type rows:", len(cty_df))

    # Sample cells according to cell type proportions
    if target_n_cells is not None and target_n_cells < len(cty_df):
        print(f"Sampling to {target_n_cells} cells while maintaining cell type proportions...")
        type_counts = Counter(cty_df['cell_type'])
        total_cells = len(cty_df)
        selected_cells = []
        for cell_type, count in type_counts.items():
            frac = target_n_cells / total_cells
            n_sample = max(1, int(round(count * frac)))  # Round to nearest integer, at least 1
            type_cells = cty_df[cty_df['cell_type'] == cell_type].index.tolist()
            sampled = np.random.choice(type_cells, n_sample, replace=False).tolist()
            selected_cells.extend(sampled)
        # Ensure selected cell count does not exceed target_n_cells
        if len(selected_cells) > target_n_cells:
            selected_cells = np.random.choice(selected_cells, target_n_cells, replace=False).tolist()
        # Filter all DataFrames
        rna_hvg_df = rna_hvg_df.loc[selected_cells]
        atac_lsi_df = atac_lsi_df.loc[selected_cells]
        cty_df = cty_df.loc[selected_cells]
        print(f"Number of cells after sampling: {len(selected_cells)}")
        # Print type distribution after sampling
        sampled_counts = Counter(cty_df['cell_type'])
        print("Cell type distribution after sampling:", dict(sampled_counts))
    else:
        print("No cell sampling performed, using all cells.")
    return rna_hvg_df, atac_lsi_df, cty_df

def load_rna_atac_diag(base_dir, rna_filename, atac_filename, cty_filename_rna,cty_filename_atac,\
                           n_hvg=2000, n_lsi=50, rna_n_cells=2000,atac_n_cells=1000, QC_min_genes=500, random_seed=123):
    """
    Read RNA, scATAC H5 data and cell type CSV, filter for diagonal alignment, return integrated DataFrame
    Parameters:
    base_dir: str, data folder
    atac_filename: str, scATAC h5 filename
    rna_filename: str, RNA h5 filename
    cty_filename_rna: str, cell type csv filename for RNA
    cty_filename_atac: str, cell type csv filename for ATAC
    n_hvg: int, number of RNA highly variable genes
    n_lsi: int, number of ATAC LSI components
    target_n_cells: int, target total number of cells (if specified, randomly sample to this count maintaining proportions; if None, no sampling)
    random_seed: int, random seed for reproducibility
    Returns:
    rna_hvg_df: pandas DataFrame, RNA highly variable gene matrix (filtered)
    atac_lsi_df: pandas DataFrame, ATAC LSI matrix (filtered)
    cty_rna: pandas DataFrame, cell types for RNA (filtered)
    cty_atac: pandas DataFrame, cell types for ATAC (filtered)
    """
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Concatenate paths
    atac_file = os.path.join(base_dir, atac_filename)
    rna_file = os.path.join(base_dir, rna_filename)
    cty_file_rna = os.path.join(base_dir, cty_filename_rna)
    cty_file_atac = os.path.join(base_dir, cty_filename_atac)
    # Read H5 files
    atac = h5py.File(atac_file, "r")
    rna = h5py.File(rna_file, "r")
    # Read row names (cells)
    atac_rows = [x.decode() for x in atac["matrix"]["barcodes"][:]]
    rna_rows = [x.decode() for x in rna["matrix"]["barcodes"][:]]
    print(f"Number of cells: ATAC={len(atac_rows)}, RNA={len(rna_rows)}")
    print("Are row names consistent:", atac_rows == rna_rows)

    # Read column names (features)
    atac_cols = [x.decode() for x in atac["matrix"]["features"][:]]
    rna_cols = [x.decode() for x in rna["matrix"]["features"][:]]
    print(f"Number of columns: ATAC={len(atac_cols)}, RNA={len(rna_cols)}")

    # Read matrix and transpose to (cells x features)
    atac_X = np.array(atac["matrix"]["data"]).T  # Transpose
    atac_df = pd.DataFrame(atac_X, index=atac_rows, columns=atac_cols)
    print("ATAC raw dimensions:", atac_df.shape)

    rna_X = np.array(rna["matrix"]["data"]).T
    rna_df = pd.DataFrame(rna_X, index=rna_rows, columns=rna_cols)
    rna_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rna_df.fillna(0, inplace=True)
    print("RNA raw dimensions:", rna_df.shape)

    # QC---Filter low-quality cells
    # Count number of genes expressed per cell
    rna_gene_counts = (rna_df > 0).sum(axis=1)
    atac_gene_counts = (atac_df > 0).sum(axis=1)
    rna_df = rna_df.loc[rna_gene_counts >= QC_min_genes, :]
    atac_df = atac_df.loc[atac_gene_counts >= QC_min_genes, :]
    print(f"RNA cell count after QC: {rna_df.shape[0]}")
    print(f"ATAC cell count after QC: {atac_df.shape[0]}")
    if rna_df.shape[1] > n_hvg:
        # RNA highly variable gene selection
        adata_rna = sc.AnnData(rna_df)
        sc.pp.normalize_total(adata_rna, target_sum=1e4)
        sc.pp.log1p(adata_rna)
        sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_hvg, flavor='seurat')
        rna_hvg_df = pd.DataFrame(adata_rna[:, adata_rna.var['highly_variable']].X,
                                  index=rna_df.index,
                                  columns=adata_rna.var_names[adata_rna.var['highly_variable']])
    else:
        rna_hvg_df = rna_df
    print("RNA highly variable gene matrix dimensions:", rna_hvg_df.shape)
    if atac_df.shape[1] > n_lsi:
        # ATAC feature selection: TF-IDF + LSI
        # Calculate TF
        tf = atac_df.div(atac_df.sum(axis=1), axis=0)
        # Calculate IDF
        idf = np.log(1 + atac_df.shape[0] / (1 + (atac_df > 0).sum(axis=0)))
        tf_idf = tf * idf
        # LSI (SVD)
        svd = TruncatedSVD(n_components=n_lsi, random_state=random_seed)
        atac_lsi = svd.fit_transform(tf_idf)
        atac_lsi_df = pd.DataFrame(atac_lsi, index=atac_df.index,
                                   columns=[f'LSI_{i+1}' for i in range(atac_lsi.shape[1])])
    else:
        atac_lsi_df = atac_df
    print("ATAC LSI matrix dimensions:", atac_lsi_df.shape)

    # Read RNA cell types
    cty_rna_df = pd.read_csv(cty_file_rna, header=0)
    cty_rna_df.index = rna_rows  # Default order correspondence
    cty_rna_df.columns = ['cell_type']
    print("Number of cell samples:", len(cty_rna_df))
    # Re-match cells
    common_cells = rna_hvg_df.index.intersection(cty_rna_df.index)
    rna_hvg_df = rna_hvg_df.loc[common_cells]
    cty_rna_df = cty_rna_df.loc[common_cells]

    # Read ATAC cell types
    cty_atac_df = pd.read_csv(cty_file_atac, header=0)
    cty_atac_df.index = atac_rows  # Default order correspondence
    cty_atac_df.columns = ['cell_type']
    print("Number of cell samples:", len(cty_atac_df))
    # Re-match cells
    common_cells = atac_lsi_df.index.intersection(cty_atac_df.index)
    atac_lsi_df = atac_lsi_df.loc[common_cells]
    cty_atac_df = cty_atac_df.loc[common_cells]

    # Sample cells by cell type
    rna_hvg_df, cty_rna_df = sample_cells_by_celltype(
        cty_df=cty_rna_df,
        rna_hvg_df=rna_hvg_df,
        target_n_cells=rna_n_cells
    )
    atac_lsi_df, cty_atac_df = sample_cells_by_celltype(
        cty_df=cty_atac_df,
        rna_hvg_df=atac_lsi_df,
        target_n_cells=atac_n_cells
    )
    return rna_hvg_df, atac_lsi_df, cty_rna_df, cty_atac_df
def load_rna_atac_crossmo(base_dir, atac_train_filename,atac_test_filename, rna_train_filename,
                          rna_test_filename, cty_train_filename,cty_test_filename,
                           n_hvg=1000, n_lsi=50, target_n_cells=None, random_seed=123):
    """
    Read RNA, scATAC H5 data and cell type CSV, reduce dimensions for vertical alignment, return integrated DataFrame
    Parameters:
    base_dir: str, data folder
    atac_filename: str, scATAC h5 filename
    rna_filename: str, RNA h5 filename
    cty_filename: str, cell type csv filename
    n_hvg: int, number of RNA highly variable genes
    n_lsi: int, number of ATAC LSI components
    target_n_cells: int, target total number of cells (if specified, randomly sample to this count maintaining proportions; if None, no sampling)
    random_seed: int, random seed for reproducibility
    Returns:
    rna_hvg_df: pandas DataFrame, RNA highly variable gene matrix (filtered)
    atac_lsi_df: pandas DataFrame, ATAC LSI matrix (filtered)
    cty_df: pandas DataFrame, cell types (filtered)
    """
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Concatenate paths
    atac_train_file = os.path.join(base_dir, atac_train_filename)
    rna_train_file = os.path.join(base_dir, rna_train_filename)
    cty_train_file = os.path.join(base_dir, cty_train_filename)

    atac_test_file = os.path.join(base_dir, atac_test_filename)
    rna_test_file = os.path.join(base_dir, rna_test_filename)
    cty_test_file = os.path.join(base_dir, cty_test_filename)
    # Read H5 files
    atac_train = h5py.File(atac_train_file, "r")
    rna_train = h5py.File(rna_train_file, "r")
    atac_test = h5py.File(atac_test_file, "r")
    rna_test = h5py.File(rna_test_file, "r")
    # train---Read row names (cells)
    atac_train_rows = [x.decode() for x in atac_train["matrix"]["barcodes"][:]]
    rna_train_rows = [x.decode() for x in rna_train["matrix"]["barcodes"][:]]
    print(f"train number of cells: ATAC={len(atac_train_rows)}, RNA={len(rna_train_rows)}")
    print("train row names consistent:", atac_train_rows == rna_train_rows)
    # Read column names (features)
    atac_train_cols = [x.decode() for x in atac_train["matrix"]["features"][:]]
    rna_train_cols = [x.decode() for x in rna_train["matrix"]["features"][:]]
    print(f"train number of columns: ATAC={len(atac_train_cols)}, RNA={len(rna_train_cols)}")
    # Read matrix and transpose to (cells x features)
    atac_train_X = np.array(atac_train["matrix"]["data"]).T  # Transpose
    atac_train_df = pd.DataFrame(atac_train_X, index=atac_train_rows, columns=atac_train_cols)
    print("train ATAC raw dimensions:", atac_train_df.shape)
    rna_train_X = np.array(rna_train["matrix"]["data"]).T
    rna_train_df = pd.DataFrame(rna_train_X, index=rna_train_rows, columns=rna_train_cols)
    rna_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rna_train_df.fillna(0, inplace=True)
    print("train RNA raw dimensions:", rna_train_df.shape)

    # test---Read row names (cells)
    atac_test_rows = [x.decode() for x in atac_test["matrix"]["barcodes"][:]]
    rna_test_rows = [x.decode() for x in rna_test["matrix"]["barcodes"][:]]
    print(f"test number of cells: ATAC={len(atac_test_rows)}, RNA={len(rna_test_rows)}")
    print("test row names consistent:", atac_test_rows == rna_test_rows)
    # Read column names (features)
    atac_test_cols = [x.decode() for x in atac_test["matrix"]["features"][:]]
    rna_test_cols = [x.decode() for x in rna_test["matrix"]["features"][:]]
    print(f"test number of columns: ATAC={len(atac_test_cols)}, RNA={len(rna_test_cols)}")
    # Read matrix and transpose to (cells x features)
    atac_test_X = np.array(atac_test["matrix"]["data"]).T  # Transpose
    atac_test_df = pd.DataFrame(atac_test_X, index=atac_test_rows, columns=atac_test_cols)
    print("test ATAC raw dimensions:", atac_test_df.shape)
    rna_test_X = np.array(rna_test["matrix"]["data"]).T
    rna_test_df = pd.DataFrame(rna_test_X, index=rna_test_rows, columns=rna_test_cols)
    rna_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rna_test_df.fillna(0, inplace=True)
    print("test RNA raw dimensions:", rna_test_df.shape)

    print("---- Preprocessing RNA (train + test together) ...")
    # concat
    rna_all_df = pd.concat([rna_train_df, rna_test_df], axis=0)
    rna_all_cells = rna_all_df.index.tolist()
    n_train_rna = rna_train_df.shape[0]
    # DataFrame → AnnData
    adata_rna = sc.AnnData(rna_all_df.values)
    adata_rna.obs_names = rna_all_cells
    adata_rna.var_names = rna_all_df.columns
    n_hvg_eff = min(n_hvg, adata_rna.n_vars)
    print(f"RNA genes: {adata_rna.n_vars}, using n_hvg = {n_hvg_eff}")
    sc.pp.normalize_total(adata_rna, target_sum=1e4)  # Temporary normalization only for HVG
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_hvg_eff)
    hvg_genes = adata_rna.var[adata_rna.var['highly_variable']].index.tolist()  # Only take gene names
    # Filter directly using column names
    rna_train_df = rna_train_df[hvg_genes]
    rna_test_df = rna_test_df[hvg_genes]

    # Read train cell types
    cty_train_df = pd.read_csv(cty_train_file, header=0)
    cty_train_df.index = atac_train_rows  # Default order correspondence
    cty_train_df.columns = ['cell_type']
    print("train number of cell type rows:", len(cty_train_df))
    # Read test cell types
    cty_test_df = pd.read_csv(cty_test_file, header=0)
    cty_test_df.index = atac_test_rows  # Default order correspondence
    cty_test_df.columns = ['cell_type']
    print("test number of cell type rows:", len(cty_test_df))
    # Sample train cells according to cell type proportions
    if target_n_cells is not None and target_n_cells < len(cty_train_df):
        print(f"Sampling to {target_n_cells} cells while maintaining cell type proportions...")
        type_counts = Counter(cty_train_df['cell_type'])
        total_cells = len(cty_train_df)
        selected_cells = []
        for cell_type, count in type_counts.items():
            frac = target_n_cells / total_cells
            n_sample = max(1, int(round(count * frac)))  # Round to nearest integer, at least 1
            type_cells = cty_train_df[cty_train_df['cell_type'] == cell_type].index.tolist()
            sampled = np.random.choice(type_cells, n_sample, replace=False).tolist()
            selected_cells.extend(sampled)
        # Ensure selected cell count does not exceed target_n_cells
        if len(selected_cells) > target_n_cells:
            selected_cells = np.random.choice(selected_cells, target_n_cells, replace=False).tolist()
        # Filter all DataFrames
        rna_train_df = rna_train_df.loc[selected_cells]
        atac_train_df = atac_train_df.loc[selected_cells]
        cty_train_df = cty_train_df.loc[selected_cells]
        print(f"Number of cells after sampling: {len(selected_cells)}")
        # Print type distribution after sampling
        sampled_counts = Counter(cty_train_df['cell_type'])
        print("Cell type distribution after sampling:", dict(sampled_counts))
    else:
        print("No cell sampling performed, using all cells.")

    # ========== Filter all-zero cells (train) ==========
    rna_train_nonzero = (rna_train_df.sum(axis=1) != 0)
    atac_train_nonzero = (atac_train_df.sum(axis=1) != 0)
    keep_train_cells = rna_train_nonzero & atac_train_nonzero
    print(
        f"Train all-zero cell filtering:"
        f" before={len(keep_train_cells)}, "
        f" after={keep_train_cells.sum()}"
    )
    rna_train_df = rna_train_df.loc[keep_train_cells]
    atac_train_df = atac_train_df.loc[keep_train_cells]
    cty_train_rows = rna_train_df.index.tolist()
    # Take intersection with filtered cells
    common_cells = rna_train_df.index.intersection(cty_train_df.index)
    rna_train_df = rna_train_df.loc[common_cells]
    atac_train_df = atac_train_df.loc[common_cells]
    cty_train_df = cty_train_df.loc[common_cells]

    # ========== Filter all-zero cells (test) ==========
    rna_test_nonzero = (rna_test_df.sum(axis=1) != 0)
    atac_test_nonzero = (atac_test_df.sum(axis=1) != 0)
    keep_test_cells = rna_test_nonzero & atac_test_nonzero
    print(
        f"Test all-zero cell filtering:"
        f" before={len(keep_test_cells)}, "
        f" after={keep_test_cells.sum()}"
    )
    rna_test_df = rna_test_df.loc[keep_test_cells]
    atac_test_df = atac_test_df.loc[keep_test_cells]
    cty_test_rows = rna_test_df.index.tolist()
    # Take intersection with filtered cells
    common_cells = rna_test_df.index.intersection(cty_test_df.index)
    rna_test_df = rna_test_df.loc[common_cells]
    atac_test_df = atac_test_df.loc[common_cells]
    cty_test_df = cty_test_df.loc[common_cells]
    return rna_train_df, atac_train_df, rna_test_df, atac_test_df, cty_train_df, cty_test_df

def load_rna_adt_crossmo(base_dir, adt_train_filename,adt_test_filename, rna_train_filename,
                          rna_test_filename, cty_train_filename,cty_test_filename,
                           n_hvg=1000,target_n_cells=None, random_seed=123):
    """
    Read ADT, RNA H5 data and cell type CSV, return DataFrames
    Parameters:
    base_dir: str, common folder path
    adt_filename: str, ADT h5 filename
    rna_filename: str, RNA h5 filename
    cty_filename: str, cell type csv filename
    Returns:
    adt_df, rna_df, cty_df: pandas DataFrame
    """
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Concatenate paths
    adt_train_file = os.path.join(base_dir, adt_train_filename)
    rna_train_file = os.path.join(base_dir, rna_train_filename)
    cty_train_file = os.path.join(base_dir, cty_train_filename)

    adt_test_file = os.path.join(base_dir, adt_test_filename)
    rna_test_file = os.path.join(base_dir, rna_test_filename)
    cty_test_file = os.path.join(base_dir, cty_test_filename)
    # Read H5 files
    adt_train = h5py.File(adt_train_file, "r")
    rna_train = h5py.File(rna_train_file, "r")
    adt_test = h5py.File(adt_test_file, "r")
    rna_test = h5py.File(rna_test_file, "r")
    # train---Read row names (cells)
    adt_train_rows = [x.decode() for x in adt_train["matrix"]["barcodes"][:]]
    rna_train_rows = [x.decode() for x in rna_train["matrix"]["barcodes"][:]]
    print(f"train number of cells: adt={len(adt_train_rows)}, RNA={len(rna_train_rows)}")
    print("train row names consistent:", adt_train_rows == rna_train_rows)
    # Read column names (features)
    adt_train_cols = [x.decode() for x in adt_train["matrix"]["features"][:]]
    rna_train_cols = [x.decode() for x in rna_train["matrix"]["features"][:]]
    print(f"train number of columns: adt={len(adt_train_cols)}, RNA={len(rna_train_cols)}")
    # Read matrix and transpose to (cells x features)
    adt_train_X = np.array(adt_train["matrix"]["data"]).T  # Transpose
    adt_train_df = pd.DataFrame(adt_train_X, index=adt_train_rows, columns=adt_train_cols)
    print("train adt raw dimensions:", adt_train_df.shape)
    rna_train_X = np.array(rna_train["matrix"]["data"]).T
    rna_train_df = pd.DataFrame(rna_train_X, index=rna_train_rows, columns=rna_train_cols)
    rna_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rna_train_df.fillna(0, inplace=True)
    print("train RNA raw dimensions:", rna_train_df.shape)

    # test---Read row names (cells)
    adt_test_rows = [x.decode() for x in adt_test["matrix"]["barcodes"][:]]
    rna_test_rows = [x.decode() for x in rna_test["matrix"]["barcodes"][:]]
    print(f"test number of cells: adt={len(adt_test_rows)}, RNA={len(rna_test_rows)}")
    print("test row names consistent:", adt_test_rows == rna_test_rows)
    # Read column names (features)
    adt_test_cols = [x.decode() for x in adt_test["matrix"]["features"][:]]
    rna_test_cols = [x.decode() for x in rna_test["matrix"]["features"][:]]
    print(f"test number of columns: adt={len(adt_test_cols)}, RNA={len(rna_test_cols)}")
    # Read matrix and transpose to (cells x features)
    adt_test_X = np.array(adt_test["matrix"]["data"]).T  # Transpose
    adt_test_df = pd.DataFrame(adt_test_X, index=adt_test_rows, columns=adt_test_cols)
    print("test adt raw dimensions:", adt_test_df.shape)
    rna_test_X = np.array(rna_test["matrix"]["data"]).T
    rna_test_df = pd.DataFrame(rna_test_X, index=rna_test_rows, columns=rna_test_cols)
    rna_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rna_test_df.fillna(0, inplace=True)
    print("test RNA raw dimensions:", rna_test_df.shape)
    print("---- Preprocessing RNA (train + test together) ...")
    # concat
    rna_all_df = pd.concat([rna_train_df, rna_test_df], axis=0)
    rna_all_cells = rna_all_df.index.tolist()
    n_train_rna = rna_train_df.shape[0]
    # DataFrame → AnnData
    adata_rna = sc.AnnData(rna_all_df.values)
    adata_rna.obs_names = rna_all_cells
    adata_rna.var_names = rna_all_df.columns
    n_hvg_eff = min(n_hvg, adata_rna.n_vars)
    print(f"RNA genes: {adata_rna.n_vars}, using n_hvg = {n_hvg_eff}")
    sc.pp.normalize_total(adata_rna, target_sum=1e4)  # Temporary normalization only for HVG
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_hvg_eff)
    hvg_genes = adata_rna.var[adata_rna.var['highly_variable']].index.tolist()  # Only take gene names
    # Filter directly using column names
    rna_train_df = rna_train_df[hvg_genes]
    rna_test_df = rna_test_df[hvg_genes]
    # Read train cell types
    cty_train_df = pd.read_csv(cty_train_file, header=0)
    cty_train_df.index = adt_train_rows  # Default order correspondence
    cty_train_df.columns = ['cell_type']
    print("train number of cell type rows:", len(cty_train_df))
    # Read test cell types
    cty_test_df = pd.read_csv(cty_test_file, header=0)
    cty_test_df.index = adt_test_rows  # Default order correspondence
    cty_test_df.columns = ['cell_type']
    print("test number of cell type rows:", len(cty_test_df))
    # Sample train cells according to cell type proportions
    if target_n_cells is not None and target_n_cells < len(cty_train_df):
        print(f"Sampling to {target_n_cells} cells while maintaining cell type proportions...")
        type_counts = Counter(cty_train_df['cell_type'])
        total_cells = len(cty_train_df)
        selected_cells = []
        for cell_type, count in type_counts.items():
            frac = target_n_cells / total_cells
            n_sample = max(1, int(round(count * frac)))  # Round to nearest integer, at least 1
            type_cells = cty_train_df[cty_train_df['cell_type'] == cell_type].index.tolist()
            sampled = np.random.choice(type_cells, n_sample, replace=False).tolist()
            selected_cells.extend(sampled)
        # Ensure selected cell count does not exceed target_n_cells
        if len(selected_cells) > target_n_cells:
            selected_cells = np.random.choice(selected_cells, target_n_cells, replace=False).tolist()
        # Filter all DataFrames
        rna_train_df = rna_train_df.loc[selected_cells]
        adt_train_df = adt_train_df.loc[selected_cells]
        cty_train_df = cty_train_df.loc[selected_cells]
        print(f"Number of cells after sampling: {len(selected_cells)}")
        # Print type distribution after sampling
        sampled_counts = Counter(cty_train_df['cell_type'])
        print("Cell type distribution after sampling:", dict(sampled_counts))
    else:
        print("No cell sampling performed, using all cells.")

    return rna_train_df, adt_train_df, rna_test_df, adt_test_df, cty_train_df, cty_test_df
from sklearn.model_selection import train_test_split
def split_train_test(
        rna_df,
        atac_df,
        celltype,
        save_dir,
        test_size=0.3,
        random_state=42):

    """
    atac_df : DataFrame (cells x peaks)
    rna_df  : DataFrame (cells x genes)
    celltype : array-like
    """

    os.makedirs(save_dir, exist_ok=True)

    indices = np.arange(len(celltype))

    # Stratified split
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=celltype
    )

    # ---------- Train ----------

    atac_train = atac_df.iloc[train_idx]
    rna_train = rna_df.iloc[train_idx]

    ct_train = np.array(celltype)[train_idx]

    # ---------- Test ----------

    atac_test = atac_df.iloc[test_idx]
    rna_test = rna_df.iloc[test_idx]

    ct_test = np.array(celltype)[test_idx]
    return rna_train, atac_train, rna_test, atac_test, ct_train, ct_test