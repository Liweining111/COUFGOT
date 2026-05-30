import numpy as np
import os
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py
import os
from sklearn.model_selection import train_test_split
def save_10x_like_h5(file_path, X, barcodes, features):
    """
    Save as H5 format compatible with your reading method:
    matrix/
        data
        barcodes
        features
    """

    with h5py.File(file_path, "w") as f:

        grp = f.create_group("matrix")

        grp.create_dataset(
            "data",
            data=X.T,  # Note: Save as (feature x cell)
            compression="gzip"
        )

        grp.create_dataset(
            "barcodes",
            data=np.array(barcodes, dtype="S")
        )

        grp.create_dataset(
            "features",
            data=np.array(features, dtype="S")
        )


def split_and_save(
        atac_df,
        rna_df,
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

    # ---------- Save ATAC ----------

    save_10x_like_h5(
        os.path.join(save_dir, "atac1.h5"),
        atac_train.values,
        atac_train.index,
        atac_train.columns
    )

    save_10x_like_h5(
        os.path.join(save_dir, "atac2.h5"),
        atac_test.values,
        atac_test.index,
        atac_test.columns
    )

    # ---------- Save RNA ----------

    save_10x_like_h5(
        os.path.join(save_dir, "rna1.h5"),
        rna_train.values,
        rna_train.index,
        rna_train.columns
    )

    save_10x_like_h5(
        os.path.join(save_dir, "rna2.h5"),
        rna_test.values,
        rna_test.index,
        rna_test.columns
    )

    # ---------- Save celltype ----------

    pd.DataFrame(
        ct_train,
        columns=["cell_type"]
    ).to_csv(
        os.path.join(save_dir, "cty1.csv"),
        index=False
    )

    pd.DataFrame(
        ct_test,
        columns=["cell_type"]
    ).to_csv(
        os.path.join(save_dir, "cty2.csv"),
        index=False
    )

    print("Train ATAC:", atac_train.shape)
    print("Train RNA:", rna_train.shape)

    print("Test ATAC:", atac_test.shape)
    print("Test RNA:", rna_test.shape)