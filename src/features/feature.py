import numpy as np
import plyfile
import torch
from torch.utils.data import Dataset

class HeritagePointCloudDataset(Dataset):
    def __init__(self, file_path, partition_method='block', block_size=(1.0, 1.0), 
                 angle_step=30, normalize=False, center_to_zero=True,
                 transform=None): 
        
        self.ply_data = plyfile.PlyData.read(file_path)
        self.points = np.stack([
            self.ply_data['vertex']['x'],
            self.ply_data['vertex']['y'],
            self.ply_data['vertex']['z'],
            self.ply_data['vertex']['red'],
            self.ply_data['vertex']['green'],
            self.ply_data['vertex']['blue'],
            self.ply_data['vertex']['scalar_Roughness_(0.05)'],
            self.ply_data['vertex']['scalar_Mean_curvature_(0.05)'],
            self.ply_data['vertex']['scalar_Normal_change_rate_(0.05)'],
            self.ply_data['vertex']['scalar_Anisotropy_(0.05)'],
            self.ply_data['vertex']['scalar_Planarity_(0.05)'],
            self.ply_data['vertex']['scalar_Linearity_(0.05)'],
            self.ply_data['vertex']['scalar_Surface_variation_(0.05)']
        ], axis=1).astype(np.float32)
        
        if 'scalar_class' in self.ply_data['vertex']:
            self.labels = np.array(self.ply_data['vertex']['scalar_class']).astype(np.int64)
            print(f"โหลด Label จาก 'scalar_class' เจอ {len(np.unique(self.labels))} คลาส)")
        elif 'Class_Label' in self.ply_data['vertex']:
            self.labels = np.array(self.ply_data['vertex']['Class_Label']).astype(np.int64)
            print(f"โหลด Label จาก 'Class_Label' สำเร็จ!")
        else:
            print("คำเตือน: ไม่พบคอลัมน์ Label ในไฟล์ .ply เซ็ตทุกจุดเป็น Class 0")
            self.labels = np.zeros(len(self.points), dtype=np.int64)

        if center_to_zero:
            self.points = self._center_data(self.points)

        if normalize:
            self.points = self._normalize_data(self.points)

        self.angle_step = angle_step
        self.transform = transform 
        self.chunks = self._partition_data(partition_method, block_size)

    def _center_data(self, points):
        min_bound = np.min(points[:, :3], axis=0)
        points[:, :3] = points[:, :3] - min_bound
        return points

    def _normalize_data(self, points):
        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points[:, :3])**2, axis=-1)))
        points[:, :3] /= furthest_distance
        return points

    def _partition_data(self, method, block_size):
        chunks = []
        if method == 'block':
            x_min, y_min = np.min(self.points[:, :2], axis=0)
            x_max, y_max = np.max(self.points[:, :2], axis=0)
            x_steps = np.arange(x_min, x_max, block_size[0])
            y_steps = np.arange(y_min, y_max, block_size[1])
            for x in x_steps:
                for y in y_steps:
                    mask = (self.points[:, 0] >= x) & (self.points[:, 0] < x + block_size[0]) & \
                           (self.points[:, 1] >= y) & (self.points[:, 1] < y + block_size[1])
                    if np.any(mask):
                        chunks.append({'points': self.points[mask], 'labels': self.labels[mask]})
        elif method == 'angular':
            angles = np.arctan2(self.points[:, 1], self.points[:, 0])
            angles_deg = np.degrees(angles) 
            for start_angle in range(-180, 180, self.angle_step):
                end_angle = start_angle + self.angle_step
                mask = (angles_deg >= start_angle) & (angles_deg < end_angle)
                if np.any(mask):
                    chunks.append({'points': self.points[mask], 'labels': self.labels[mask]})
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        points = self.chunks[idx]['points'].copy()
        labels = self.chunks[idx]['labels'].copy()
        
        if self.transform is not None:
            points, labels = self.transform(points, labels)
        
        return torch.from_numpy(points), torch.from_numpy(labels)