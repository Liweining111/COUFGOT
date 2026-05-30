import numpy as np
import torch
import ot
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from COUFGOT.main_COUFGOT.model.eval import *
from COUFGOT.main_COUFGOT.model.filter import apply_filter,compute_covariance_matrix
from COUFGOT.main_COUFGOT.model.initialization import MegaWass
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
scaler = MinMaxScaler()

def train_ufgot_vertical_samp(X, Y, filter_type='None', p1=[0.1, 0.5, 1, 5, 10, 50, 100],
                       p2=[0.1, 0.5, 1, 5, 10, 50, 100], eps=0, filename='umap.png',
                    celltype='', new_filename='cluster.png',save_dir=''):
    r"""
        Vertical alignment training function (using UFGOT)
        Vertical alignment: same samples, different features, using sample transfer matrix
        X, Y: shape (n, d1) and (n, d2), number of samples n is the same
    """
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Data preprocessing
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y)
    X_scaled = scaler.fit_transform(X)
    # X_scaled = normalize(X_scaled, norm='l2', axis=1)
    # Y_scaled = normalize(Y_scaled, norm='l2', axis=1)
    # Check dimensions
    #assert X_scaled.shape[0] == Y_scaled.shape[0], "Vertical alignment requires the same number of samples"
    #print(f"Vertical alignment: X shape{X_scaled.shape} -> Y shape{Y_scaled.shape}")
    n = X_scaled.shape[0]
    pi_init = np.eye(n)
    pi_init = pi_init / pi_init.sum()
    pi_init = torch.tensor(pi_init, device=device, dtype=torch.float32)
    if eps == 0:
        para_eps = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    else:
        para_eps = [eps]

    if filter_type != 'None':
        print(f"Applied {filter_type} filter to covariance matrices")

    megawass = MegaWass(nits_bcd=200, nits_uot=1000, tol_bcd=1e-6, tol_uot=1e-6, eval_bcd=1,eval_uot=20)

    X_tensor = torch.Tensor(X_scaled.astype(float)).float().to(device)
    Y_tensor = torch.Tensor(Y_scaled.astype(float)).float().to(device)
    eval_ufgot_best = float('inf')
    avFOSCTTM_best = float('inf')
    i = 0
    best_params = None

    ### Construct sample graph matrix
    device, dtype = X_tensor.device, X_tensor.dtype
    gL1 = X_tensor @ X_tensor.T  # Sample covariance matrix
    gL2 = Y_tensor @ Y_tensor.T
    # Apply filter
    if filter_type != 'identity':
        gL1 = apply_filter(gL1, filter_type, device=device)
        gL2 = apply_filter(gL2, filter_type, device=device)
    ### Feature graph matrix does not need to be constructed
    nx, dx = X.shape
    ny, dy = Y.shape
    gl1 = np.zeros((dx, dx), dtype=np.float32)
    gl2 = np.zeros((dy, dy), dtype=np.float32)
    gL1 = torch.as_tensor(gL1, dtype=torch.float32, device=device)
    gL2 = torch.as_tensor(gL2, dtype=torch.float32, device=device)

    for rho1 in p1:
        for rho2 in p2:
            for eps1 in para_eps:
                rho = (rho1, rho2)
                eps_param = (eps1, 0)
                print("Initial parameters =", float(rho1), float(rho2), float(eps1))
                # Vertical alignment uses UFGOT, only trains the feature transfer matrix, transpose data to align feature dimensions
                pi_samp = megawass.solver_megawass_ufgot(
                    X=X_tensor,
                    Y=Y_tensor,
                    gL1=gL1,
                    gL2=gL2,
                    gl1=gl1,
                    gl2=gl2,
                    rho=rho,
                    eps=eps_param,
                    init_pi=(pi_init,None),
                    entropic_mode="independent",
                    verbose=True,
                    vertical=True,
                    filter_type=filter_type
                )

                if isinstance(pi_samp, torch.Tensor):
                    pi_samp_np = pi_samp.cpu().numpy()
                else:
                    pi_samp_np = pi_samp
                if (np.isnan(pi_samp_np).any() or np.isinf(pi_samp_np).any()) :
                    print("NaN or Inf detected in pi_samp, skip this iteration.")
                    continue
                # Align data: Y_aligned = π_samp^T @ Y @ π_feat
                pi_samp_np = sparsify_pi(pi_samp_np, k=50)
                #col_mass = pi_samp_np.sum(axis=1)  # (n_samp_X, 1)
                #X_alig= pi_samp_np.T @ X_scaled/(col_mass[:, None] + 1e-12)
                col_mass = pi_samp_np.sum(axis=1,keepdims=True)  # (n_samp_X, 1)
                X_alig = pi_samp_np @ Y_scaled/(col_mass + 1e-12)
                #X_alig = scaler.fit_transform(X_alig)
                #Y_alig = scaler.transform(Y_alig)
                # Evaluate alignment effect
                #eval_ufgot = foscttm(Y_alig, X_scaled)
                fracs = calc_domainAveraged_FOSCTTM(X_alig, Y_scaled)
                avFOSCTTM = np.mean(fracs)
                eval_ufgot = 0
                save_cluster = os.path.join(save_dir, new_filename)
                metrics = plot_clustering_celltype(
                    Y_align=X_alig,
                    celltype=celltype,
                    save_path=save_cluster
                )
                metrics_fmt = {k: round(v, 6) for k, v in metrics.items()}
                print("alig_clustering metrics:", metrics_fmt)
                print(f'Vertical alignment Iteration {i+1} -- FOSCTTM: {eval_ufgot:.6f},avFOSCTTM: {avFOSCTTM:.6f}, params: rho1={rho1}, rho2={rho2}, eps={eps1}')
                i += 1
                if avFOSCTTM < avFOSCTTM_best:
                    eval_ufgot_best = eval_ufgot
                    avFOSCTTM_best = avFOSCTTM
                    best_fracs = fracs
                    data_best_alig = X_alig
                    best_params = (rho1, rho2, eps1)
                    best_pi_samp = pi_samp_np
    print(f"Vertical alignment best FOSCTTM: {eval_ufgot_best:.6f},avFOSCTTM: {avFOSCTTM_best:.6f} with params: {best_params}")
    return {
        'aligned_data': data_best_alig,
        'sample_coupling': best_pi_samp,
        'best_FOSCTTM': avFOSCTTM_best,
        'best_params': best_params,
        'Y_scaled': Y_scaled,
        'X_scaled':X_scaled
    }

def train_coufgot_diag(X, Y, filter_type='None', p1=[0.1, 0.5, 1, 5, 10, 50, 100], p2=[0.1, 0.5, 1, 5, 10, 50, 100],\
                       eps=0, filename='umap.png',celltype='', new_filename='cluster.png', save_dir=''):
    r"""
        Diagonal alignment training function (using CO-UFGOT)
        Diagonal alignment: different samples and features, using sample and feature transfer matrices
        X, Y: shape (n1, d1) and (n2, d2)
    """
    scaler = MinMaxScaler()
    use_cuda = torch.cuda.is_available()
    print('Is CUDA available:')
    print(use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    device = 'cpu'
    print('device:', device)
    # Data preprocessing
    Y_scaled = scaler.fit_transform(Y)
    X_scaled = scaler.fit_transform(X)
    Y_scaled = normalize(Y_scaled, norm='l2', axis=1)
    X_scaled = normalize(X_scaled, norm='l2', axis=1)
    print(f"Diagonal alignment: X shape{X_scaled.shape} -> Y shape{Y_scaled.shape}")
    if eps == 0:
        para_eps = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    else:
        para_eps = [eps]
    if filter_type != 'None':
        print(f"Applied {filter_type} filter to covariance matrices")

    megawass = MegaWass(nits_bcd=100, nits_uot=1000, tol_bcd=1e-7, tol_uot=1e-6, eval_bcd=1,eval_uot=50)

    X_tensor = torch.Tensor(X_scaled.astype(float)).float().to(device)
    Y_tensor = torch.Tensor(Y_scaled.astype(float)).float().to(device)
    eval_ufgot_best = float('inf')
    i = 0
    best_params = None

    ### Construct sample graph matrix
    device, dtype = X_tensor.device, X_tensor.dtype
    gL1 = X_tensor @ X_tensor.T  # Sample covariance matrix
    gL2 = Y_tensor @ Y_tensor.T
    # Apply filter
    if filter_type != 'identity':
        gL1 = apply_filter(gL1, filter_type, device=device)
        gL2 = apply_filter(gL2, filter_type, device=device)
    ### Construct feature graph matrix
    gl1 = X_tensor.T @ X_tensor  # Feature covariance matrix
    gl2 = Y_tensor.T @ Y_tensor
    # Apply filter
    if filter_type != 'identity':
        gl1 = apply_filter(gl1, filter_type, device=device)
        gl2 = apply_filter(gl2, filter_type, device=device)
    # MinMaxScaler
    scale = MinMaxScaler()
    gL1 = scale.fit_transform(gL1)
    gL2 = scale.fit_transform(gL2)
    gl1 = scale.fit_transform(gl1)
    gl2 = scale.fit_transform(gl2)
    # Variable conversion
    gL1 = torch.as_tensor(gL1, dtype=torch.float32, device=device)
    gL2 = torch.as_tensor(gL2, dtype=torch.float32, device=device)
    gl1 = torch.as_tensor(gl1, dtype=torch.float32, device=device)
    gl2 = torch.as_tensor(gl2, dtype=torch.float32, device=device)

    for rho1 in p1:
        for rho2 in p2:
            for eps1 in para_eps:
                rho = (rho1, rho2)
                eps = (eps1, 0)
                print("Initial parameters =", float(rho1), float(rho2), float(eps1))
                # Diagonal alignment uses CO-UFGOT, trains sample and feature transfer matrices
                (pi_samp, pi_feat),_ = megawass.solver_megawass_coufgot(
                    X=X_tensor,
                    Y=Y_tensor,
                    gL1=gL1,
                    gL2=gL2,
                    gl1=gl1,
                    gl2=gl2,
                    rho=rho,
                    eps=eps,
                    log=False,
                    verbose=False,
                    early_stopping_tol=1e-6,
                    filter_type=filter_type,
                    device=device
                )

                # Convert to numpy
                if isinstance(pi_feat, torch.Tensor):
                    pi_feat_np = pi_feat.cpu().numpy()
                else:
                    pi_feat_np = pi_feat
                if isinstance(pi_samp, torch.Tensor):
                    pi_samp_np = pi_samp.cpu().numpy()
                else:
                    pi_samp_np = pi_samp
                if (np.isnan(pi_feat_np).any() or np.isnan(pi_samp_np).any() or
                        np.isinf(pi_feat_np).any() or np.isinf(pi_samp_np).any()):
                    print("NaN or Inf detected in pi_feat or pi_samp, skip this iteration.")
                    continue

                #print("pi_samp_np:", pi_samp_np.shape)
                #print("Y_scaled:", Y_scaled.shape)
                #print("pi_feat_np:", pi_feat_np.shape)

                Y_alig= pi_samp_np @ Y_scaled @ pi_feat_np.T
                Y_alig= scaler.fit_transform(Y_alig)
                # Evaluate alignment effect
                eval_ufgot = foscttm(Y_alig, X_scaled)
                fracs = calc_domainAveraged_FOSCTTM(Y_alig, X_scaled)
                avFOSCTTM = np.mean(fracs)

                save_cluster = os.path.join(save_dir, new_filename)
                metrics = plot_clustering_celltype(
                    Y_align=Y_alig,
                    celltype=celltype,
                    save_path=save_cluster
                )
                metrics_fmt = {k: round(v, 6) for k, v in metrics.items()}
                print("Clustering metrics:", metrics_fmt)

                print(f'Diagonal alignment Iteration {i+1} -- FOSCTTM: {avFOSCTTM:.6f}, params: rho1={rho1}, rho2={rho2}, eps={eps1}')
                i += 1
                if avFOSCTTM < eval_ufgot_best:
                    eval_ufgot_best = avFOSCTTM
                    data_best_alig = Y_alig
                    best_params = (rho1, rho2, eps1)
                    best_pi_samp = pi_samp_np
                    best_pi_feat = pi_feat_np
    print(f"Diagonal alignment best FOSCTTM: {eval_ufgot_best:.6f} with params: {best_params}")

    return {
        'aligned_data': data_best_alig,
        'sample_coupling': best_pi_samp,
        'feature_coupling': best_pi_feat,
        'best_FOSCTTM': eval_ufgot_best,
        'best_params':best_params,
        'X_scaled': X_scaled,
        'Y_scaled':Y_scaled
    }

def train_ufgot_diag_gas(X, Y, filter_type='None', p1=[0.1, 0.5, 1, 5, 10, 50, 100],
                       p2=[0.1, 0.5, 1, 5, 10, 50, 100], eps=0,
                    celltype='', new_filename='cluster.png',save_dir=''):
    r"""
        Diagonal alignment training function (using UFGOT)
        Using atac_gas data
    """
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Data preprocessing
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = scaler.fit_transform(Y)
    # X_scaled = normalize(X_scaled, norm='l2', axis=1)
    # Y_scaled = normalize(Y_scaled, norm='l2', axis=1)

    if eps == 0:
        para_eps = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    else:
        para_eps = [eps]

    if filter_type != 'None':
        print(f"Applied {filter_type} filter to covariance matrices")

    megawass = MegaWass(nits_bcd=200, nits_uot=1000, tol_bcd=1e-6, tol_uot=1e-6, eval_bcd=1,eval_uot=20)

    X_tensor = torch.Tensor(X_scaled.astype(float)).float().to(device)
    Y_tensor = torch.Tensor(Y_scaled.astype(float)).float().to(device)
    eval_ufgot_best = float('inf')
    avFOSCTTM_best = float('inf')
    i = 0
    best_params = None

    ### Construct sample graph matrix
    device, dtype = X_tensor.device, X_tensor.dtype
    gL1 = X_tensor @ X_tensor.T  # Sample covariance matrix
    gL2 = Y_tensor @ Y_tensor.T
    # Apply filter
    if filter_type != 'identity':
        gL1 = apply_filter(gL1, filter_type, device=device)
        gL2 = apply_filter(gL2, filter_type, device=device)
    ### Feature graph matrix does not need to be constructed
    nx, dx = X.shape
    ny, dy = Y.shape
    gl1 = np.zeros((dx, dx), dtype=np.float32)
    gl2 = np.zeros((dy, dy), dtype=np.float32)
    # MinMaxScaler
    scale = MinMaxScaler()
    gL1 = scale.fit_transform(gL1)
    gL2 = scale.fit_transform(gL2)
    gL1 = torch.as_tensor(gL1, dtype=torch.float32, device=device)
    gL2 = torch.as_tensor(gL2, dtype=torch.float32, device=device)

    for rho1 in p1:
        for rho2 in p2:
            for eps1 in para_eps:
                rho = (rho1, rho2)
                eps_param = (eps1, 0)
                print("Initial parameters =", float(rho1), float(rho2), float(eps1))
                # Vertical alignment uses UFGOT, only trains the feature transfer matrix, transpose data to align feature dimensions
                pi_samp = megawass.solver_megawass_ufgot(
                    X=X_tensor,
                    Y=Y_tensor,
                    gL1=gL1,
                    gL2=gL2,
                    gl1=gl1,
                    gl2=gl2,
                    rho=rho,
                    eps=eps_param,
                    verbose=True,
                    vertical=True,
                    filter_type=filter_type
                )

                if isinstance(pi_samp, torch.Tensor):
                    pi_samp_np = pi_samp.cpu().numpy()
                else:
                    pi_samp_np = pi_samp
                if (np.isnan(pi_samp_np).any() or np.isinf(pi_samp_np).any()) :
                    print("NaN or Inf detected in pi_samp, skip this iteration.")
                    continue
                # Align data: Y_aligned = π_samp^T @ Y @ π_feat
                col_mass = pi_samp_np.sum(axis=1, keepdims=True)  # (n_samp_X, 1)
                X_alig= pi_samp_np @ Y_scaled/ (col_mass + 1e-12)
                #X_alig = scaler.fit_transform(X_alig)
                #print(pi_samp_np.shape)
                #pi_samp_np = sparsify_pi(pi_samp_np, k=50)
                #col_mass = pi_samp_np.sum(axis=0, keepdims=True)
                #X_alig = pi_samp_np.T @ X_scaled / (col_mass.T + 1e-12)
                # Evaluate alignment effect
                fracs = 0
                avFOSCTTM = 0
                save_cluster = os.path.join(save_dir, new_filename)
                metrics = plot_clustering_celltype(
                    Y_align=X_alig,
                    celltype=celltype,
                    save_path=save_cluster
                )
                metrics_fmt = {k: round(v, 6) for k, v in metrics.items()}
                print("Clustering metrics:", metrics_fmt)
                print(f'Diagonal alignment Iteration {i+1} -- avFOSCTTM: {avFOSCTTM:.6f}, params: rho1={rho1}, rho2={rho2}, eps={eps1}')
                i += 1
                if avFOSCTTM < avFOSCTTM_best:
                    avFOSCTTM_best = avFOSCTTM
                    best_fracs = fracs
                    data_best_alig = X_alig
                    best_params = (rho1, rho2, eps1)
                    best_pi_samp = pi_samp_np
    print(f"Diagonal alignment best avFOSCTTM: {avFOSCTTM_best:.6f} with params: {best_params}")

    return {
        'aligned_data': data_best_alig,
        'sample_coupling': best_pi_samp,
        'X_scaled': X_scaled,
        'Y_scaled': Y_scaled,
        'best_params': best_params
    }

def train_ufgot_crossmo(X, Y, X_test, Y_test, filter_type='None', p1=[0.1, 0.5, 1, 5, 10, 50, 100],
                       p2=[0.1, 0.5, 1, 5, 10, 50, 100], eps=0, filename='umap.png',save_dir='',cty_test=''):
    r"""
        Cross-modal translation training function (using UFGOT)
        Cross-modal translation: same samples, different features, using feature transfer matrix
        X, Y: shape (n, d1) and (n, d2), number of samples n is the same
    """
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Data preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = scaler.fit_transform(Y)
    # X_scaled = normalize(X_scaled, norm='l2', axis=1)
    # Y_scaled = normalize(Y_scaled, norm='l2', axis=1)
    # Check dimensions
    assert X_scaled.shape[0] == Y_scaled.shape[0], "Cross-modal translation requires the same number of samples"
    print(f"Cross-modal translation: X shape{X_scaled.shape} -> Y shape{Y_scaled.shape}")

    if eps == 0:
        para_eps = [1e-3, 1e-2, 1e-1, 1]
    else:
        para_eps = [eps]

    if filter_type != 'None':
        print(f"Applied {filter_type} filter to covariance matrices")

    megawass = MegaWass(nits_bcd=200, nits_uot=1000, tol_bcd=1e-6, tol_uot=1e-6, eval_bcd=1,eval_uot=20)

    X_tensor = torch.Tensor(X_scaled.astype(float)).float().to(device)
    Y_tensor = torch.Tensor(Y_scaled.astype(float)).float().to(device)
    eval_mse_best = float('inf')
    mse_best = float('inf')
    i = 0
    best_params = None

    ### Construct feature graph matrix
    device, dtype = X_tensor.device, X_tensor.dtype
    gl1 = X_tensor.T @ X_tensor  # Feature covariance matrix
    gl2 = Y_tensor.T @ Y_tensor
    # Apply filter
    if filter_type != 'identity':
        gl1 = apply_filter(gl1, filter_type, device=device)
        gl2 = apply_filter(gl2, filter_type, device=device)
    ### Sample graph matrix does not need to be constructed
    nx, dx = X.shape
    ny, dy = Y.shape
    gL1 = np.zeros((nx, nx), dtype=np.float32)
    gL2 = np.zeros((ny, ny), dtype=np.float32)
    # MinMaxScaler
    scale = MinMaxScaler()
    gl1 = scale.fit_transform(gl1)
    gl2 = scale.fit_transform(gl2)
    gl1 = torch.as_tensor(gl1, dtype=torch.float32, device=device)
    gl2 = torch.as_tensor(gl2, dtype=torch.float32, device=device)

    Y_origin = Y_test
    X_origin = Y
    zscore = StandardScaler()
    X_test = zscore.fit_transform(X_test)
    Y_test = zscore.fit_transform(Y_test)
    for rho1 in p1:
        for rho2 in p2:
            for eps1 in para_eps:
                rho = (rho1, rho2)
                eps_param = (eps1, 0)
                print("Initial parameters =", float(rho1), float(rho2), float(eps1))
                # Cross-modal translation uses UFGOT, only trains the feature transfer matrix
                pi_feat = megawass.solver_megawass_ufgot(
                    X=X_tensor,
                    Y=Y_tensor,
                    gL1=gL1,
                    gL2=gL2,
                    gl1=gl1,
                    gl2=gl2,
                    rho=rho,
                    eps=eps_param,
                    verbose=True,
                    horizonal=True,
                    filter_type=filter_type
                )
                if isinstance(pi_feat, torch.Tensor):
                    pi_feat_np = pi_feat.cpu().numpy()
                else:
                    pi_feat_np = pi_feat
                if (np.isnan(pi_feat_np).any() or np.isinf(pi_feat_np).any()) :
                    print("NaN or Inf detected in pi_feat, skip this iteration.")
                    continue
                # Align data: Y_aligned = π_samp @ Y @ π_feat^T, map Y to X's space
                pi_feat_np = sparsify_pi(pi_feat_np, k=50)
                col_mass = pi_feat_np.sum(axis=0, keepdims=True)  # (n_feat_X, 1)
                Y_alig = (X_scaled @ pi_feat_np) / (col_mass + 1e-12)
                #Y_alig = scaler.transform(Y_alig)
                #X_alig = normalize(X_alig, norm='l2', axis=0)
                # Evaluate alignment effect
                mse = mean_squared_error(Y_alig, Y_scaled)
                print(f'Cross-modal translation Iteration {i+1} -- MSE: {mse:.6f}, params: rho1={rho1}, rho2={rho2}, eps={eps1}')

                Y_pre = (X_test @ pi_feat_np) / (col_mass + 1e-12)
                #Y_pre = zscore.transform(Y_pre)
                ###++++++++++ Evaluation metrics ++++++++++###
                # Calculate clustering metrics
                save_path = os.path.join(save_dir, 'clustering_crossmo')
                metrics = plot_clustering_celltype(
                    Y_align=Y_pre,
                    celltype=cty_test,
                    random_state=123,
                    save_path=save_path
                )
                MSE_test = mean_squared_error(Y_pre, Y_test)
                ARI, NMI, SC = metrics["ARI"], metrics["NMI"], metrics["SC"]
                # Output results
                print(f"MSE: {MSE_test:.4f}")
                print(f"ARI: {ARI:.4f}")
                print(f"NMI: {NMI:.4f}")
                print(f"SC: {SC:.4f}")

                i += 1
                if mse < mse_best:
                    eval_mse_best = mse
                    data_best_alig = Y_alig
                    best_params = (rho1, rho2, eps1)
                    best_pi_feat = pi_feat_np
    print(f"Cross-modal translation best MSE: {eval_mse_best:.6f} with params: {best_params}")

    return {
        'aligned_data': data_best_alig,
        'feat_coupling': best_pi_feat,
        'best_params': best_params
    }