import os
import json
import argparse
import numpy as np
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture
import matplotlib.cm as cm
from matplotlib.colors import to_hex


class ClusterByGaussian:
    def __init__(self, input_npz: str = "./latent_param_dataset_16.npz",
                 output_csv: str = "./dataset_with_colors.csv",
                 frontend_js_path: str = None,
                 max_components: int = 12):
        self.input_npz = input_npz
        self.output_csv = output_csv
        if frontend_js_path is None:
            # Go up two levels: clustering/ -> python_server/ -> project_root/
            frontend_js_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dataset3D_withcolors.js')
            )
        self.frontend_js_path = frontend_js_path
        self.max_components = int(max_components)

    def train_dataset_with_colors(self):
        # 1) Load points (expecting 3D)
        ds = np.load(self.input_npz)
        pts = np.squeeze(ds['reduced_latent_matrix'])
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[1] < 3:
            raise ValueError("Dataset has fewer than 3 dimensions; cannot export x,y,z.")
        points = pts[:, :3]

        # 2) Cluster with Gaussian Mixture (select number of components via BIC)
        max_k = max(2, min(self.max_components, points.shape[0]))
        print(f"Clustering with GaussianMixture (model selection via BIC over k=2..{max_k}) ...")
        best_gmm = None
        best_bic = np.inf
        best_k = None
        for k in range(2, max_k + 1):
            try:
                gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=2, random_state=42)
                gmm.fit(points)
                bic = gmm.bic(points)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_k = k
            except Exception as e:
                print(f"GMM(k={k}) failed: {e}")
                continue
        if best_gmm is None:
            raise RuntimeError("GaussianMixture model selection failed for all k")
        labels = best_gmm.predict(points)

        # 3) Log cluster counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Selected k={best_k} with BIC={best_bic:.2f}")
        print(f"Found {len(unique_labels)} clusters")
        for lbl, cnt in zip(unique_labels, counts):
            print(f" - Cluster {lbl}: {cnt} points")

        # 4) Compute per-cluster closeness using mean Mahalanobis distance to centroid
        cluster_closeness = {}
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            pts_c = points[idx]
            mean = best_gmm.means_[lbl]
            cov = best_gmm.covariances_[lbl]  # (3x3) for covariance_type='full'
            # Regularize covariance to avoid singularities
            cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
            try:
                inv_cov = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_reg)
            diffs = pts_c - mean
            # Mahalanobis distances
            dists = np.sqrt(np.einsum('ij,jk,ik->i', diffs, inv_cov, diffs))
            mean_md = float(np.mean(dists)) if dists.size else 0.0
            cluster_closeness[lbl] = mean_md

        # Normalize closeness: lower mean_md (tighter) -> higher closeness in [0,1]
        if len(cluster_closeness) > 0:
            vals = np.array(list(cluster_closeness.values()))
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax > vmin:
                for lbl, v in cluster_closeness.items():
                    closeness = (vmax - v) / (vmax - vmin)
                    cluster_closeness[lbl] = float(closeness)
            else:
                for lbl in cluster_closeness.keys():
                    cluster_closeness[lbl] = 1.0

        # 4) Map closeness to colors (RGB in [0,1])
        cmap = cm.get_cmap('viridis')
        label_to_rgb = {}
        for lbl in unique_labels:
            closeness = cluster_closeness.get(lbl, 0.0)
            r, g, b, _ = cmap(closeness)
            label_to_rgb[lbl] = (float(r), float(g), float(b))
        # Log per-cluster RGB and meaning
        for lbl, cnt in zip(unique_labels, counts):
            closeness = cluster_closeness.get(lbl, 0.0)
            rgb = label_to_rgb.get(lbl, (0.5, 0.5, 0.5))
            rgb_short = tuple(round(v, 3) for v in rgb)
            hex_color = to_hex(rgb)
            print(f"Cluster {int(lbl)}: closeness={closeness:.3f}, color rgb={rgb_short}, hex={hex_color}, members={int(cnt)}")

        # 5) Assign RGB per point
        colors = [label_to_rgb.get(lbl, (0.5, 0.5, 0.5)) for lbl in labels]
        colors = np.array(colors)

        # 6) Write CSV x,y,z,r,g,b
        with open(self.output_csv, 'w') as f:
            f.write("x,y,z,r,g,b\n")
            for (xv, yv, zv), (rv, gv, bv) in zip(points, colors):
                f.write(f"{xv},{yv},{zv},{rv},{gv},{bv}\n")

        # 7) Write frontend JS dataset
        js_obj = {
            "x": points[:, 0].tolist(),
            "y": points[:, 1].tolist(),
            "z": points[:, 2].tolist(),
            "r": colors[:, 0].tolist(),
            "g": colors[:, 1].tolist(),
            "b": colors[:, 2].tolist(),
        }
        js_content = "var dataset3D_withcolors = " + json.dumps(js_obj) + "\n"
        with open(self.frontend_js_path, 'w') as jf:
            jf.write(js_content)

        return {
            "output_csv": self.output_csv,
            "output_js": self.frontend_js_path,
            "n_clusters": int(len(unique_labels)),
            "cluster_counts": {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)},
            "selected_k": int(best_k),
            "bic": float(best_bic),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dataset colors and export JS/CSV")
    parser.add_argument("--input", dest="input_npz", default="./latent_param_dataset_16.npz", help="Path to input NPZ")
    parser.add_argument("--csv", dest="output_csv", default="./dataset_with_colors.csv", help="Path to output CSV")
    parser.add_argument(
        "--frontend-js",
        dest="frontend_js_path",
        # Go up two levels: clustering/ -> python_server/ -> project_root/
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dataset3D_withcolors.js')),
        help="Path to output dataset3D_withcolors.js",
    )
    parser.add_argument("--max-components", dest="max_components", type=int, default=12, help="Max components to try for GMM BIC selection")
    args = parser.parse_args()

    trainer = ClusterByGaussian(
        input_npz=args.input_npz,
        output_csv=args.output_csv,
        frontend_js_path=args.frontend_js_path,
        max_components=args.max_components,
    )
    info = trainer.train_dataset_with_colors()
    print("Export complete:")
    for k, v in info.items():
        print(f" - {k}: {v}")
