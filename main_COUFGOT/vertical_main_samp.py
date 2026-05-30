import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from model.model_caller import train_ufgot_vertical_samp
from model.data_input import load_rna_adt_vertical, load_rna_atac_vertical
from model.eval import *
import pandas as pd
begin_time = time.time()
def vertical_alignment_pipeline():
    r"""Execute the complete data vertical alignment process - for cell*gene format"""
    # Load data
    print("Loading data...")
    dir_path = 'D:/CO-UFGOT_code/lwn_COUFGOT/result/vertical/img/COUFGOT'#保存路径
    # H5 type data
    data_path = 'D:/CO-UFGOT_code/lwn_COUFGOT/datasets/D3'
    save_data = r'D:\CO-UFGOT_code\lwn_COUFGOT\result\vertical\result_data\COUFGOT\D3'
    '''
    #ATAC+RNA
    X, Y, celltype = load_rna_atac_vertical(data_path, "atac.h5", "rna.h5", "cty.csv",\
                                        n_hvg=2000, n_lsi=1000, target_n_cells=2000)
    '''

    #ADT+RNA
    X, Y, celltype = load_rna_adt_vertical(data_path, "adt.h5", "rna.h5", "cty.csv",
                                           n_hvg=2000, target_n_cells=3000)

    clustering_filename_single = 'D3_UFGOT_clustering_celltype_align.png'
    clustering_filename_co = 'D3_UFGOT_clustering_celltype_int.png'
    xy_filename = 'D3_UFGOT_xy.png'
    avFOSCTTM_filename = 'D3_UFGOT_FOSCTTM.png'
    print(f"Data shape: X{X.shape}, Y{Y.shape}")
    print(f"X - Number of cells: {X.shape[0]}, Number of features: {X.shape[1]}")
    print(f"Y - Number of cells: {Y.shape[0]}, Number of features: {Y.shape[1]}")

    # Check if cells are aligned
    if X.shape[0] != Y.shape[0]:
        print("Warning: Number of cells in the two datasets does not match!")
        # Take cell intersection
        common_cell = X.index.intersection(Y.index)
        X = X.loc[common_cell]
        Y = Y.loc[common_cell]
        print(f"Using common cells: {len(common_cell)} cells")

    # Data preprocessing, ensure data type is numeric
    X = X.fillna(0).astype(np.float32)
    Y = Y.fillna(0).astype(np.float32)
    random_state = 123
    # Perform vertical alignment - align feature dimensions
    print("Starting vertical alignment training...")
    result = train_ufgot_vertical_samp(
        X=X.values,  # Shape: (n_cells_X, n_genes)
        Y=Y.values,  # Shape: (n_cells_Y, n_genes)
        filter_type='g2',  # Options: 'identity', 'g1', 'g2', ..., 'g6'
        p1=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        p2=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        eps=0,
        filename=xy_filename,
        new_filename=clustering_filename_single,
        celltype=celltype,
        save_dir=dir_path
    )
    end_time = time.time()
    all_time = end_time - begin_time

    return result


# main
if __name__ == "__main__":
    result, X, Y = vertical_alignment_pipeline()