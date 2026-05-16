import numpy as np

class RandomRotateZ:
    """สุ่มหมุนพอยต์คลาวด์รอบแกน Z (เฉพาะพิกัด X, Y, Z)"""
    def __call__(self, points, labels):
        theta = np.random.uniform(0, 2 * np.pi)
        cos_val, sin_val = np.cos(theta), np.sin(theta)
        
        rotation_matrix = np.array([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0],
            [0, 0, 1]
        ])
        
        # หมุนเฉพาะพิกัดแกน X, Y, Z (Channels 0, 1, 2)
        points[:, :3] = points[:, :3] @ rotation_matrix.T
        return points, labels

class FarthestPointSampling:
    """สุ่มจุดแบบกระจายตัว (FPS) เพื่อให้ได้จำนวนจุดที่คงที่"""
    def __init__(self, num_points=4096):
        self.num_points = num_points
        
    def __call__(self, points, labels):
        num_samples = self.num_points
        
        if len(points) <= num_samples:
            idx = np.random.choice(len(points), num_samples, replace=True)
            return points[idx], labels[idx]
        
        selected_idx = np.zeros(num_samples, dtype=np.int32)
        distances = np.ones(len(points)) * 1e10
        farthest_node = np.random.randint(0, len(points))
        
        xyz = points[:, :3]
        
        for i in range(num_samples):
            selected_idx[i] = farthest_node
            centroid = xyz[farthest_node, :]
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest_node = np.argmax(distances)
            
        return points[selected_idx], labels[selected_idx]

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, labels):
        for t in self.transforms:
            points, labels = t(points, labels)
        return points, labels