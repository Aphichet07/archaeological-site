import numpy as np
import plyfile
import torch
from torch.utils.data import Dataset


class HeritagePointCloudDataset(Dataset):
    """
    Features :
        0-2  : X, Y, Z
        3-5  : R, G, B  
        6    : Roughness
        7    : Mean Curvature
        8    : Normal Change Rate
        9    : Anisotropy
        10   : Planarity
        11   : Linearity
        12   : Surface Variation
    """
    FEATURE_COLUMNS = [
        'x', 'y', 'z',
        'red', 'green', 'blue',
        'scalar_Roughness_(0.05)',
        'scalar_Mean_curvature_(0.05)',
        'scalar_Normal_change_rate_(0.05)',
        'scalar_Anisotropy_(0.05)',
        'scalar_Planarity_(0.05)',
        'scalar_Linearity_(0.05)',
        'scalar_Surface_variation_(0.05)',
    ]

    LABEL_COLUMNS = ['scalar_class', 'Class_Label']

    def __init__(
        self,
        file_path: str,
        partition_method: str = 'block',
        block_size: tuple = (1.0, 1.0),
        angle_step: int = 30,
        normalize: bool = True,       
        center_to_zero: bool = True,
        min_points_per_chunk: int = 512,  
        transform=None,
    ):
        self.transform = transform
        self.angle_step = angle_step
        self.min_points_per_chunk = min_points_per_chunk

        ply_data = plyfile.PlyData.read(file_path)  

        vertex_props = {p.name for p in ply_data['vertex'].properties}
        missing = [c for c in self.FEATURE_COLUMNS if c not in vertex_props]
        if missing:
            raise ValueError(f"ไม่พบ column ใน .ply: {missing}")

        self.points = np.stack(
            [ply_data['vertex'][col] for col in self.FEATURE_COLUMNS],
            axis=1
        ).astype(np.float32)

        self.labels, self.class_map, self.num_classes = self._load_labels(ply_data, vertex_props)

        del ply_data

        if center_to_zero:
            self.points = self._center_data(self.points)

        if normalize:
            self.points = self._normalize_features(self.points)

        self.chunks = self._partition_data(partition_method, block_size)
        print(f"partition={partition_method} | chunks={len(self.chunks)} "
              f"| min_pts={min_points_per_chunk}")


    def _load_labels(self, ply_data, vertex_props):
        raw_labels = None
        for col in self.LABEL_COLUMNS:
            if col in vertex_props:
                raw_labels = np.array(ply_data['vertex'][col], dtype=np.int64)
                print(f"โหลด label จาก '{col}'")
                break

        if raw_labels is None:
            print("ไม่พบ label column ทุกจุดเป็น class 0")
            raw_labels = np.zeros(len(self.points), dtype=np.int64)

        unique_classes = np.unique(raw_labels)
        class_map = {int(old): int(new) for new, old in enumerate(unique_classes)}
        remapped = np.vectorize(class_map.get)(raw_labels).astype(np.int64)
        num_classes = len(unique_classes)

        print(f"[Dataset] class_map = {class_map} | num_classes = {num_classes}")
        return remapped, class_map, num_classes

    def _center_data(self, points: np.ndarray) -> np.ndarray:
        """Shift XYZ ให้ min bound = (0,0,0)"""
        min_bound = np.min(points[:, :3], axis=0)
        points[:, :3] -= min_bound
        return points

    def _normalize_features(self, points: np.ndarray) -> np.ndarray:
        """
        XYZ  → unit sphere  ให้ทำหลัง center แล้ว
        RGB  → [0, 1]
        Geo  → z-score per feature 
        """
        points = points.copy()

        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] -= centroid
        furthest = np.max(np.linalg.norm(points[:, :3], axis=1))
        if furthest > 1e-6:
            points[:, :3] /= furthest

        # RGB: 0-255 → 0-1
        points[:, 3:6] = np.clip(points[:, 3:6] / 255.0, 0.0, 1.0)

        # Geometric features (col 6-12): z-score normalization
        for i in range(6, points.shape[1]):
            col = points[:, i]
            mean, std = col.mean(), col.std()
            if std > 1e-6:
                points[:, i] = (col - mean) / std
            else:
                points[:, i] = col - mean  

        return points

    def _partition_data(self, method: str, block_size: tuple) -> list:
        """แบ่ง point cloud เป็น chunks"""
        chunks = []

        if method == 'block':
            x_min, y_min = np.min(self.points[:, :2], axis=0)
            x_max, y_max = np.max(self.points[:, :2], axis=0)
            x_steps = np.arange(x_min, x_max, block_size[0])
            y_steps = np.arange(y_min, y_max, block_size[1])

            for x in x_steps:
                for y in y_steps:
                    mask = (
                        (self.points[:, 0] >= x) & (self.points[:, 0] < x + block_size[0]) &
                        (self.points[:, 1] >= y) & (self.points[:, 1] < y + block_size[1])
                    )
                    if mask.sum() >= self.min_points_per_chunk:
                        chunks.append({
                            'points': self.points[mask],
                            'labels': self.labels[mask],
                        })

        elif method == 'angular':
            angles_deg = np.degrees(np.arctan2(self.points[:, 1], self.points[:, 0]))
            for start_angle in range(-180, 180, self.angle_step):
                end_angle = start_angle + self.angle_step
                mask = (angles_deg >= start_angle) & (angles_deg < end_angle)
                if mask.sum() >= self.min_points_per_chunk:
                    chunks.append({
                        'points': self.points[mask],
                        'labels': self.labels[mask],
                    })

        else:
            raise ValueError(f"partition_method ไม่รู้จัก: '{method}'. เลือก 'block' หรือ 'angular'")

        if len(chunks) == 0:
            raise RuntimeError(
                f"ไม่มี chunk ที่มีจุด >= {self.min_points_per_chunk} "
               
            )

        return chunks


    def get_raw(self, idx: int):
        """
        คืน (points, labels) 
        """
        return (
            self.chunks[idx]['points'].copy(),
            self.chunks[idx]['labels'].copy(),
        )

    def get_statistics(self) -> np.ndarray:
        total_pts = sum(c['points'].shape[0] for c in self.chunks)
        all_labels = np.concatenate([c['labels'] for c in self.chunks])
        class_counts = np.bincount(all_labels, minlength=self.num_classes)

        print("=" * 50)
        print(f"  Dataset Statistics")
        print("=" * 50)
        print(f"  Total chunks     : {len(self.chunks):,}")
        print(f"  Total points     : {total_pts:,}")
        print(f"  Avg pts/chunk    : {total_pts / len(self.chunks):,.0f}")
        print(f"  Num classes      : {self.num_classes}")
        print(f"  Class distribution:")
        for cls_id, count in enumerate(class_counts):
            bar = '█' * int(30 * count / total_pts)
            print(f"    Class {cls_id:2d}: {count:>10,} pts ({100*count/total_pts:5.1f}%)  {bar}")

        imbalance_ratio = class_counts.max() / (class_counts.min() + 1e-6)
        if imbalance_ratio > 10:
            print(f"\n Imbalance ratio = {imbalance_ratio:.1f}x ")
        print("=" * 50)

        return class_counts

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        points, labels = self.get_raw(idx)

        if self.transform is not None:
            points, labels = self.transform(points, labels)

        return torch.from_numpy(points), torch.from_numpy(labels)

    def __repr__(self) -> str:
        return (
            f"HeritagePointCloudDataset("
            f"chunks={len(self.chunks)}, "
            f"features={self.points.shape[1]}, "
            f"num_classes={self.num_classes})"
        )