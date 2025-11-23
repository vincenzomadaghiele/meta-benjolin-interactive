import os
import json
import argparse
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm


class DataTrainerDBScan:
    def __init__(self, input_npz: str = "./latent_param_dataset_16.npz",
                 output_csv: str = "./dataset_with_colors.csv",
                 frontend_js_path: str = None):
        self.input_npz = input_npz
        self.output_csv = output_csv
        if frontend_js_path is None:
            frontend_js_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dataset3D_withcolors.js')
            )
        self.frontend_js_path = frontend_js_path

    def train_dataset_with_colors(self):
        # 1) Load points (expecting 3D)
        ds = np.load(self.input_npz)
        pts = np.squeeze(ds['reduced_latent_matrix'])
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[1] < 3:
            raise ValueError("Dataset has fewer than 3 dimensions; cannot export x,y,z.")
        points = pts[:, :3]

        # 2) Cluster with DBSCAN (heuristic parameters)
        kdt = KDTree(points)
        dists, _ = kdt.query(points, k=2)  # k=2 to skip self-distance
        nn = dists[:, 1]
        median_nn = float(np.median(nn))
        eps = 1.5 * median_nn if median_nn > 0 else 0.05
        min_samples = max(5, int(round(points.shape[0] * 0.005)))  # ~0.5% or at least 5

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(points)

        # 3) Compute per-cluster closeness
        unique_labels = np.unique(labels[labels >= 0])  # exclude noise (-1)
        cluster_closeness = {}
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if idx.size <= 1:
                cluster_closeness[lbl] = 0.0
                continue
            pts_c = points[idx]
            kdt_c = KDTree(pts_c)
            d_c, _ = kdt_c.query(pts_c, k=2)
            nn_c = d_c[:, 1]
            mean_nn = float(np.mean(nn_c))
            cluster_closeness[lbl] = mean_nn

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
        noise_rgb = (0.5, 0.5, 0.5)

        # 5) Assign RGB per point
        colors = []
        for lbl in labels:
            if lbl == -1:
                colors.append(noise_rgb)
            else:
                colors.append(label_to_rgb.get(lbl, noise_rgb))
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
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": int(len(unique_labels)),
            "n_noise": int(np.sum(labels == -1)),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dataset colors and export JS/CSV")
    parser.add_argument("--input", dest="input_npz", default="./latent_param_dataset_16.npz", help="Path to input NPZ")
    parser.add_argument("--csv", dest="output_csv", default="./dataset_with_colors.csv", help="Path to output CSV")
    parser.add_argument(
        "--frontend-js",
        dest="frontend_js_path",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dataset3D_withcolors.js')),
        help="Path to output dataset3D_withcolors.js",
    )
    args = parser.parse_args()

    trainer = DataTrainer(
        input_npz=args.input_npz,
        output_csv=args.output_csv,
        frontend_js_path=args.frontend_js_path,
    )
    info = trainer.train_dataset_with_colors()
    print("Export complete:")
    for k, v in info.items():
        print(f" - {k}: {v}")
