import time
from model.model_caller import train_ufgot_crossmo
from model.data_input import load_rna_atac_crossmo,load_rna_adt_crossmo,load_rna_atac_vertical,split_train_test
from model.eval import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
begin_time = time.time()
def crossmo_alignment_pipeline():
    r"""Execute the complete process of cross-modal translation - for cell*gene format"""
    # Load data
    print("Loading data...")
    dir_path = 'D:/CO-UFGOT_code/lwn_COUFGOT/result/crossmo/img/COUFGOT'#Save path
    # H5 type data
    data_path = 'D:/CO-UFGOT_code/lwn_COUFGOT/datasets/imputation_data/D56/data1'
    save_data = r'D:\CO-UFGOT_code\lwn_COUFGOT\result\crossmo\result_data\COUFGOT\D561'

    #ATAC+RNA training data and prediction data
    X_train, Y_train, X_test, Y_test, cty_train, cty_test = load_rna_atac_crossmo(data_path, "atac1.h5", "atac2.h5","rna1.h5",
                                                                                "rna2.h5", "cty1.csv","cty2.csv",
                                                                                 n_hvg=2000, n_lsi=1000, target_n_cells=2000)

    '''
    #ADT+RNA training data and prediction data
    X_train, Y_train, X_test, Y_test, cty_train, cty_test = load_rna_adt_crossmo(data_path, "adt1.h5", "adt2.h5","rna1.h5",
                                                                                  "rna2.h5", "cty1.csv", "cty2.csv",
                                                                                  n_hvg=1000, target_n_cells=2000)
    '''
    '''
    # Read data
    X, Y, celltype = load_rna_atac_vertical(
        data_path,
        "atac.h5",
        "rna.h5",
        "cty.csv",
        n_hvg=2000,
        n_lsi=1000,
        target_n_cells=2000
    )
    X_train, Y_train, X_test, Y_test, cty_train, cty_test=split_train_test(
        X,
        Y,
        celltype,
        save_dir=save_data
    )
    '''
    cluster_filename = 'D561_crossmo.png'
    filename = 'D561.png'
    ROC_filename = 'D561_ROC_1.png'
    ROC_filename_2 = 'D561_ROC_2.png'
    print(f"Data shape: X_train{X_train.shape}, Y_train{Y_train.shape}")
    print(f"Data shape: X_test{X_test.shape}, Y_train{Y_test.shape}")

    # Check if cells are aligned
    if X_train.shape[0] != Y_train.shape[0]:
        print("Warning: Number of cells in the two datasets does not match!")
        # Take cell intersection
        common_cell = X_train.index.intersection(Y_train.index)
        X_train = X_train.loc[common_cell]
        Y_train = Y_train.loc[common_cell]
        print(f"Using common cells: {len(common_cell)} cells")

    # Data preprocessing, ensure data type is numeric
    X_train = X_train.fillna(0).astype(np.float32)
    Y_train = Y_train.fillna(0).astype(np.float32)
    X_test = X_test.fillna(0).astype(np.float32)
    Y_test = Y_test.fillna(0).astype(np.float32)
    Y_origin = Y_test.to_numpy()
    random_state = 123
    # Execute cross-modal translation
    print("Starting cross-modal translation training...")
    result = train_ufgot_crossmo(
        X=X_train.values,  # Shape: (n_cells_X, n_genes)
        Y=Y_train.values,  # Shape: (n_cells_Y, n_genes)
        X_test=X_test.values,
        Y_test=Y_test.values,
        filter_type='g2',  # Options: 'identity', 'g1', 'g2', ..., 'g6'
        p1=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        p2=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        eps=0,
        cty_test=cty_test,
        filename=filename,
        save_dir=dir_path
    )
    # Predict unknown modality
    scaler = StandardScaler()
    feat_coupling = result['feat_coupling']
    col_mass = feat_coupling.sum(axis=0, keepdims=True)  # (n_feat_X, 1)
    X_test = scaler.fit_transform(X_test)
    Y_pre = (X_test @ feat_coupling) / (col_mass + 1e-12)
    Y_test = scaler.fit_transform(Y_test)
    end_time = time.time()
    all_time = end_time - begin_time
    return result,Y_pre,Y_test


# main
if __name__ == "__main__":
    result = crossmo_alignment_pipeline()