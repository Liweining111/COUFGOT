import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from model.model_caller import train_coufgot_diag,train_ufgot_diag_gas
from model.data_input import load_rna_atac_diag
from model.eval import *
import warnings
warnings.filterwarnings("ignore")
import time
begin_time = time.time()
def diag_alignment_pipeline():
    # Load data
    print("Loading data...")
    dir_path = 'D:/CO-UFGOT_code/lwn_COUFGOT/result/diag/img/COUFGOT' #Save results
    #H5 data
    data_path = r"D:\CO-UFGOT_code\lwn_COUFGOT\datasets\D34"
    save_data = r'D:\CO-UFGOT_code\lwn_COUFGOT\result\diag\result_data\COUFGOT\D34'
    X, Y, celltype_X, celltype_Y = load_rna_atac_diag(data_path, "rna1.h5", "atac_gas1.h5",
                                                      "rna_cty1.csv", "atac_cty1.csv",
                                                      n_hvg=7745, n_lsi=7745, rna_n_cells=2000, atac_n_cells=1500,
                                                      QC_min_genes=500, random_seed=123)

    int_filename = 'D34_clustering_celltype.png'
    umap_filename = 'COUFGOT_D34_umap.png'
    align_filename = 'COUFGOT_D34_align.png'
    # Data preprocessing, ensure data type is numeric
    X = X.fillna(0).astype(np.float32)
    Y = Y.fillna(0).astype(np.float32)
    print(f"ATAC data shape: {Y.shape}")
    print(f"RNA data shape: {X.shape}")

    # Call train_coufgot_diag function for diagonal alignment
    # X: RNA data as reference, Y: ATAC data as target
    result = train_ufgot_diag_gas(
        X=X,      # Reference dataset (n1, d1)
        Y=Y,       # Target dataset (n2, d2)
        filter_type='g2',  # Options: 'g1', 'g2', etc. filter types; default 'None'
        p1=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        p2=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        eps=0,
        new_filename=int_filename,
        celltype=celltype_X,
        save_dir=dir_path
    )
    end_time = time.time()
    all_time = end_time - begin_time
    return result


# Execute alignment
if __name__ == "__main__":
    result, X, Y = diag_alignment_pipeline()