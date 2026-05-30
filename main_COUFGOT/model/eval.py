import scipy as sp
import torch
import os
import seaborn as sns
import umap
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False     # Fix negative sign display issue
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def auroc_with_curve(pred, raw, dir_path=None, filename="roc_structure.png"):

    # ===== Construct pseudo labels =====
    row_means = np.mean(raw, axis=1)
    col_means = np.mean(raw, axis=0)

    row_passes = (pred.T > row_means).T
    col_passes = pred > col_means

    y_true = np.logical_and(row_passes, col_passes).astype(int).reshape(-1)
    y_score = pred.reshape(-1)

    # ===== AUROC =====
    auc = roc_auc_score(y_true, y_score)
    auc = float(f"{auc:.6f}")

    # ===== ROC curve =====
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # ===== Plot =====
    plt.figure()
    plt.plot(fpr, tpr, label=f"Structure ROC (AUC = {auc:.6f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Structure Preservation)")
    plt.legend()
    plt.tight_layout()

    # ===== Save =====
    if dir_path is not None:
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(os.path.join(dir_path, filename), dpi=300)

    plt.show()

    return auc
def sparsify_pi(pi, k=10):
    """
    Keep only top-k per row
    """
    pi_sparse = np.zeros_like(pi)
    idx = np.argsort(pi, axis=1)[:, -k:]
    for i in range(pi.shape[0]):
        pi_sparse[i, idx[i]] = pi[i, idx[i]]
    # Re-normalize
    pi_sparse /= (pi_sparse.sum(axis=1, keepdims=True) + 1e-12)
    return pi_sparse

def calc_frac_idx(x1_mat,x2_mat):
	"""
	This returns a vector of domain-averaged FOSCTTM per cell
    Each cell has two samples, one in domain X and one in Y (e.g. its chromatin access. and gene exp. data points)
    So we average them to obtain a single FOSCTTM per cell.
	Returns fraction closer than true match for each sample (as an array)
	"""
	fracs = []
	x = []
	nsamp = x1_mat.shape[0]
	rank=0
	for row_idx in range(nsamp):
		euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
		true_nbr = euc_dist[row_idx]
		sort_euc_dist = sorted(euc_dist)
		rank =sort_euc_dist.index(true_nbr)
		frac = float(rank)/(nsamp -1)

		fracs.append(frac)
		x.append(row_idx+1)

	return fracs,x

def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
	"""
	Outputs average FOSCTTM measure (averaged over both domains)
	Get the fraction matched for all data points in both directions
	Averages the fractions in both directions for each data point
	"""
	fracs1,xs = calc_frac_idx(x1_mat, x2_mat)
	fracs2,xs = calc_frac_idx(x2_mat, x1_mat)
	fracs = []
	for i in range(len(fracs1)):
		fracs.append((fracs1[i]+fracs2[i])/2)
	return fracs

def foscttm(x: np.ndarray, y: np.ndarray, **kwargs):
    r"""Measure how many points in another modality are closer than the true matched pair - smaller is better (0 means true match is always the closest)
    Fraction of samples closer than true match (smaller is better)
    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`
    1. Input requirement: The two modality datasets x and y must come from paired samples (same order), x and y are already aligned
    2. Distance matrix: Compute Euclidean distance matrix between all x samples and all y samples
    3. Comparison baseline: Diagonal elements represent distances of "true matched pairs"
    4. Calculate proportion: For each sample, count how many "non-true matched pairs" have distances smaller than the "true matched pair"
    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively
    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    # Convert possible torch.Tensor (including GPU tensors) to CPU numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if x.shape != y.shape: # Require matching dimensions
        raise ValueError("Shapes do not match!")
    d = sp.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return ((foscttm_x + foscttm_y)/2).mean()

def plot_umap_before_after(
    X,
    Y,
    Y_align,
    n_neighbors=15,
    min_dist=0.5,
    random_state=123,
    titles=('Before alignment', 'After UFGOT alignment'),
    filename="umap_before_after.png",
    save_dir=r"D:\CO-UFGOT_code\lwn_COUFGOT\data\Horizontal\human lung"
):
    """
    Plot 1x2 UMAP comparison: before vs after alignment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    datasets = [
        (X, Y, titles[0]),
        (X, Y_align, titles[1])
    ]
    for ax, (A, B, title) in zip(axes, datasets):
        Z = np.vstack([A, B])
        batch = (
            ['target'] * A.shape[0] +
            ['source'] * B.shape[0]
        )
        adata = ad.AnnData(Z)
        adata.obs['batch'] = batch
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep='X'
        )
        sc.tl.umap(
            adata,
            min_dist=min_dist,
            random_state=random_state
        )
        sc.pl.umap(
            adata,
            color='batch',
            ax=ax,
            show=False,
            frameon=False,
            title=title
        )
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_umap_xy_align(X, Y, X_align,
                       n_neighbors=15,
                       min_dist=0.1,
                       n_components=2,
                       random_state=123,
                       titles=('Before alignment', 'After UFGOT alignment'),
                       filename="umap_before_after.png",
                       save_dir=r"D:\CO-UFGOT_code\lwn_COUFGOT\data\vertical\human lung"):
    """
    1×2 plot:
    Left: Original X and Y (visualization only, absolute positions not comparable)
    Right: Y and X_align (truly comparable: same feature space)
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -------- Left: X & Y (independent UMAP each) --------
    reducer_x = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    emb_X = reducer_x.fit_transform(X)

    reducer_y = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    emb_Y = reducer_y.fit_transform(Y)

    axes[0].scatter(emb_X[:, 0], emb_X[:, 1], s=8, alpha=0.7, label="X")
    axes[0].scatter(emb_Y[:, 0], emb_Y[:, 1], s=8, alpha=0.7, label="Y")
    axes[0].set_title("Before mapping: X vs Y")
    axes[0].legend()

    # -------- Right: Y & X_align (same space, truly comparable) --------
    reducer_align = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )

    reducer_align.fit(Y)

    emb_Y2 = reducer_align.transform(Y)
    emb_Xalign = reducer_align.transform(X_align)

    axes[1].scatter(emb_Xalign[:, 0], emb_Xalign[:, 1], s=8, alpha=0.7, label="X_align")
    axes[1].scatter(emb_Y2[:, 0], emb_Y2[:, 1], s=8, alpha=0.7, label="Y")
    axes[1].set_title("After mapping: X_align vs Y")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def row_entropy(P, eps=1e-12):
    """
    P: (n_rows, n_cols) transport matrix
    return: (n_rows,) row-wise entropy
    """
    P = np.asarray(P)
    P = P / (P.sum(axis=1, keepdims=True) + eps)  # Row normalization
    entropy = -np.sum(P * np.log(P + eps), axis=1)
    return entropy

def plot_tSNE_clustering_celltype(
        Y_align,
        celltype,
        random_state=123,
        n_neighbors=15,
        min_dist=0.3,
        point_size=10,
        save_path=None
):
    """
    UMAP visualization for clustering comparison.

    Left: Ground-truth cell types
    Right: KMeans predicted clusters

    Parameters
    ----------
    Y_align : np.ndarray, shape (n_cells, n_features) Aligned ATAC representations.
    celltype : array-like, shape (n_cells,) Ground-truth cell type labels.
    random_state : int Random seed.
    n_neighbors : int UMAP parameter.
    min_dist : float UMAP parameter.
    point_size : int Marker size for scatter plot.
    save_path : str or None If provided, save figure to this path.

    Returns
    -------
    metrics : dict ARI, NMI, and Silhouette coefficient.
    """
    celltype = np.array(celltype)
    # ===== Clustering =====
    K = len(np.unique(celltype))
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=20)
    pred = kmeans.fit_predict(Y_align)
    # Convert celltype from 2D to 1D
    celltype = np.asarray(celltype).ravel()
    ARI = adjusted_rand_score(celltype, pred)
    NMI = normalized_mutual_info_score(celltype, pred)
    SC = silhouette_score(Y_align, pred)
    # First reduce to 10 dimensions
    Y_pca = PCA(n_components=10).fit_transform(Y_align)
    # ===== T-SNE =====
    tsne = TSNE(
        n_components=2,
        perplexity=50,  # Adjustable (between 5~50)
        learning_rate=500,
        random_state=random_state,
        n_iter=1000
    )
    Y_tsne = tsne.fit_transform(Y_pca)

    # ===== Color mapping =====
    unique_ct = np.unique(celltype)
    n_ct = len(unique_ct)
    colors = sns.color_palette("tab20", n_colors=n_ct)
    color_dict = {ct: colors[i] for i, ct in enumerate(unique_ct)}

    # ===== Plot =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Ground truth cell types
    ax = axes[0]
    for ct in unique_ct:
        idx = celltype == ct
        ax.scatter(
            Y_tsne[idx, 0],
            Y_tsne[idx, 1],
            s=point_size,
            label=ct,
            color=color_dict[ct],
            alpha=0.7
        )
    ax.set_title("t-SNE (Ground Truth Cell Types)",fontsize=18,fontweight='bold')
    ax.set_xlabel("t-SNE1",fontsize=16,
    fontweight='bold')
    ax.set_ylabel("t-SNE2",fontsize=16,
    fontweight='bold')
    ax.legend(
        markerscale=3,
        prop={'weight':'bold', 'size':13},
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False
    )
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    # Right: Predicted clusters
    ax = axes[1]
    unique_pred = np.unique(pred)
    pred_colors = sns.color_palette("tab20", n_colors=len(unique_pred))  # Use Set2 for right plot to avoid conflict with left
    for k in np.unique(pred):
        idx = pred == k
        ax.scatter(
            Y_tsne[idx, 0],
            Y_tsne[idx, 1],
            s=point_size,
            label=f"Cluster {k}",
            color=pred_colors[list(unique_pred).index(k)],  # Assign colors
            alpha=0.7
        )
    ax.set_title("t-SNE (Predicted Clusters)",fontsize=18,fontweight='bold')
    ax.set_xlabel("t-SNE1",fontsize=16,
    fontweight='bold')
    ax.set_ylabel("t-SNE2",fontsize=16,
    fontweight='bold')
    ax.legend(
        markerscale=3,
        prop={'weight':'bold', 'size':13},
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False
    )
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return {
        "ARI": ARI,
        "NMI": NMI,
        "SC": SC
    }

def plot_alignment_visualization(
    X_alig,
    Y_scaled,
    celltype_X,
    celltype_Y,
    save_dir,
    filename="alignment_test",
    random_state=123,
    point_size=5
):
    """
    Visualize alignment between X_alig and Y_scaled using t-SNE.

    Parameters
    ----------
    X_alig : np.ndarray (n_X, d)
    Y_scaled : np.ndarray (n_Y, d)
    celltype_X : array-like (n_X,)
    celltype_Y : array-like (n_Y,)
    save_dir : str
    filename : str
    random_state : int
    point_size : int
    """
    os.makedirs(save_dir, exist_ok=True)
    # Concatenate
    Z = np.vstack([X_alig, Y_scaled])
    modality = np.array([0]*X_alig.shape[0] + [1]*Y_scaled.shape[0])
    celltype_all = np.concatenate([celltype_X, celltype_Y])
    # Encode
    le = LabelEncoder()
    celltype_all = le.fit_transform(celltype_all)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=random_state)
    Z_emb = tsne.fit_transform(Z)

    # Plot
    plt.figure(figsize=(12, 5))

    # ① Cell type distribution
    plt.subplot(1, 2, 1)
    sc=plt.scatter(
        Z_emb[:, 0], Z_emb[:, 1],
        c=celltype_all,
        s=point_size,
        cmap='tab20'
    )
    # Generate legend
    handles, _ = sc.legend_elements()
    labels = le.classes_
    plt.legend(handles, labels, title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Cell Type")

    # ② Modality mixing
    plt.subplot(1, 2, 2)
    color_X = "#018B8D"
    color_Y = "#E95D22"
    mask_X = modality == 0
    mask_Y = modality == 1
    plt.scatter(
        Z_emb[mask_X, 0], Z_emb[mask_X, 1],
        c=color_X, s=point_size, alpha=0.6, label="X"
    )
    plt.scatter(
        Z_emb[mask_Y, 0], Z_emb[mask_Y, 1],
        c=color_Y, s=point_size, alpha=0.6, label="Y"
    )
    plt.legend()
    plt.title("Modality")
    plt.tight_layout()
    # Save
    save_path = os.path.join(save_dir, filename + "_tsne.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved to: {save_path}")

def plot_clustering_celltype(
        Y_align,
        celltype,
        random_state=123,
        n_neighbors=15,
        min_dist=0.3,
        point_size=10,
        save_path=None
):
    """
    Parameters
    ----------
    Y_align : np.ndarray, shape (n_cells, n_features) Aligned ATAC representations.
    celltype : array-like, shape (n_cells,) Ground-truth cell type labels.
    random_state : int Random seed.
    n_neighbors : int UMAP parameter.
    min_dist : float UMAP parameter.
    point_size : int Marker size for scatter plot.
    save_path : str or None If provided, save figure to this path.

    Returns
    -------
    metrics : dict ARI, NMI, and Silhouette coefficient.
    """
    celltype = np.array(celltype)
    # ===== Clustering =====
    K = len(np.unique(celltype))
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=20)
    pred = kmeans.fit_predict(Y_align)
    # Convert celltype from 2D to 1D
    celltype = np.asarray(celltype).ravel()
    ARI = adjusted_rand_score(celltype, pred)
    NMI = normalized_mutual_info_score(celltype, pred)
    if len(np.unique(pred)) < 2:
        print("Only one cluster found, skip silhouette score.")
        SC = np.nan
    else:
        SC = silhouette_score(Y_align, pred)
    return {
        "ARI": ARI,
        "NMI": NMI,
        "SC": SC
    }

def eval_pcc_mse_feat(X_true, X_pred):
    """
    X_true, X_pred: shape (n_cells, n_features)
    """
    pcc_list = []
    mse_list = []

    for i in range(X_true.shape[1]):
        true_col = X_true[:, i]
        pred_col = X_pred[:, i]

        # Avoid pearsonr error caused by constant vectors
        if np.std(true_col) == 0 or np.std(pred_col) == 0:
            continue

        pcc, _ = spearmanr(true_col, pred_col)
        mse = mean_squared_error(true_col, pred_col)

        pcc_list.append(pcc)
        mse_list.append(mse)

    return {
        "PCC_mean": np.mean(pcc_list),
        "MSE_mean": np.mean(mse_list)
    }

def eval_pcc_mse_samp(X_true, X_pred):
    pcc_list = []
    mse_list = []

    for i in range(X_true.shape[0]):
        true_row = X_true[i, :]
        pred_row = X_pred[i, :]

        if np.std(true_row) == 0 or np.std(pred_row) == 0:
            continue

        pcc, _ = spearmanr(true_row, pred_row)
        mse = mean_squared_error(true_row, pred_row)

        pcc_list.append(pcc)
        mse_list.append(mse)

    return {
        "PCC_mean": np.mean(pcc_list),
        "MSE_mean": np.mean(mse_list)
    }