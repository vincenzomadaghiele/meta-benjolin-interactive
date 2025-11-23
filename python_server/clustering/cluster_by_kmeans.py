import os
import json
import argparse
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from matplotlib.colors import to_hex


class ClusterByKMeans:
    def __init__(self, input_npz: str = "./latent_param_dataset_16.npz",
                 output_csv: str = "./dataset_with_colors.csv",
                 frontend_js_path: str = None,
                 n_clusters: int = 10):
        self.input_npz = input_npz
        self.output_csv = output_csv
        if frontend_js_path is None:
            frontend_js_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dataset3D_withcolors.js')
            )
        self.frontend_js_path = frontend_js_path
        self.n_clusters = int(n_clusters)

    def train_dataset_with_colors(self):
        # 1) Load points (expecting 3D)
        ds = np.load(self.input_npz)
        pts = np.squeeze(ds['reduced_latent_matrix'])
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[1] < 3:
            raise ValueError("Dataset has fewer than 3 dimensions; cannot export x,y,z.")
        points = pts[:, :3]

        # 2) Cluster with KMeans
        print(f"Clustering with KMeans (k={self.n_clusters}) ...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(points)

        # Log cluster counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Found {len(unique_labels)} clusters")
        for lbl, cnt in zip(unique_labels, counts):
            print(f" - Cluster {lbl}: {cnt} points")

        # 3) Compute per-cluster compactness (mean distance to centroid)
        cluster_compactness = {}
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if idx.size == 0:
                cluster_compactness[lbl] = 0.0
                continue
            pts_c = points[idx]
            centroid = kmeans.cluster_centers_[lbl]
            dists = np.linalg.norm(pts_c - centroid, axis=1)
            mean_dist = float(np.mean(dists))
            cluster_compactness[lbl] = mean_dist

        # Map compactness (lower is tighter) to closeness in [0,1]
        if len(cluster_compactness) > 0:
            vals = np.array(list(cluster_compactness.values()))
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax > vmin:
                cluster_closeness = {lbl: float((vmax - v) / (vmax - vmin)) for lbl, v in cluster_compactness.items()}
            else:
                cluster_closeness = {lbl: 1.0 for lbl in cluster_compactness.keys()}
        else:
            cluster_closeness = {lbl: 0.0 for lbl in unique_labels}

        # 4) Map closeness to colors (RGB in [0,1])
        cmap = cm.get_cmap('viridis')
        label_to_rgb = {}
        for lbl in unique_labels:
            closeness = cluster_closeness.get(lbl, 0.0)
            r, g, b, _ = cmap(closeness)
            label_to_rgb[lbl] = (float(r), float(g), float(b))
        # Log RGB and color meaning (closeness) per cluster
        for lbl, cnt in zip(unique_labels, counts):
            closeness = cluster_closeness.get(lbl, 0.0)
            rgb = label_to_rgb.get(lbl, (0.5, 0.5, 0.5))
            rgb_short = tuple(round(v, 3) for v in rgb)
            hex_color = to_hex(rgb)
            print(f"Cluster {int(lbl)}: closeness={closeness:.3f}, color rgb={rgb_short}, hex={hex_color}, members={int(cnt)}")
        # 5) Assign RGB per point (no noise in kmeans)
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
    parser.add_argument("--k", dest="n_clusters", type=int, default=10, help="Number of clusters for KMeans")
    args = parser.parse_args()

    trainer = DataTrainer(
        input_npz=args.input_npz,
        output_csv=args.output_csv,
        frontend_js_path=args.frontend_js_path,
        n_clusters=args.n_clusters,
    )
    info = trainer.train_dataset_with_colors()
    print("Export complete:")
    for k, v in info.items():
        print(f" - {k}: {v}")
