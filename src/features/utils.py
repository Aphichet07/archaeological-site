import numpy as np

class RandomRotateZ:
    """
    สุ่มหมุน point cloud รอบแกน Z 
    """
    def __call__(self, points: np.ndarray, labels: np.ndarray):
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        R = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)

        points = points.copy()
        points[:, :3] = points[:, :3] @ R.T
        return points, labels


class RandomFlipHorizontal:
    """
    สุ่ม flip รอบแกน X หรือ Y
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        points = points.copy()
        if np.random.random() < self.p:
            points[:, 0] = -points[:, 0]   # flip X
        if np.random.random() < self.p:
            points[:, 1] = -points[:, 1]   # flip Y
        return points, labels


class RandomScale:
    """
    สุ่ม scale XYZ ในช่วง [lo, hi]
    """
    def __init__(self, lo: float = 0.9, hi: float = 1.1):
        self.lo = lo
        self.hi = hi

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        scale = np.random.uniform(self.lo, self.hi)
        points = points.copy()
        points[:, :3] *= scale
        return points, labels



class RandomJitter:
    """
    เพิ่ม Gaussian noise ให้ XYZ
    จำลอง measurement noise จาก TLS / photogrammetry

    Args:
        sigma : ค่าเบี่ยงเบนมาตรฐานของ noise 
        clip  : clipping range = ±clip  

    """
    def __init__(self, sigma: float = 0.002, clip: float = 0.005):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        noise = np.clip(
            np.random.normal(0.0, self.sigma, size=(len(points), 3)),
            -self.clip, self.clip
        ).astype(np.float32)
        points = points.copy()
        points[:, :3] += noise
        return points, labels


class RandomDropout:
    """
    สุ่มลบ points และแทนที่ด้วยจุดแรกของ chunk
    จำลองพื้นผิวชำรุด / ส่วนที่หลุดออกไปของโบราณสถาน
    """
    def __init__(self, max_dropout_ratio: float = 0.2):
        assert 0.0 <= max_dropout_ratio < 1.0
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        dropout_ratio = np.random.uniform(0.0, self.max_dropout_ratio)
        drop_mask = np.random.random(len(points)) <= dropout_ratio

        if drop_mask.all():
            drop_mask[0] = False

        points = points.copy()
        labels = labels.copy()
        points[drop_mask] = points[0]
        labels[drop_mask] = labels[0]
        return points, labels


class RandomColorJitter:
    """
    สุ่มปรับ RGB 
    จำลองความแตกต่างของแสง / สภาพอากาศระหว่างการ scan คนละครั้ง
    """
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        points = points.copy()
        b_offset = np.random.uniform(-self.brightness, self.brightness)
        c_scale = 1.0 + np.random.uniform(-self.contrast, self.contrast)

        rgb = points[:, 3:6]
        mean_rgb = rgb.mean(axis=0)
        points[:, 3:6] = np.clip((rgb - mean_rgb) * c_scale + mean_rgb + b_offset, 0.0, 1.0)
        return points, labels



class FarthestPointSampling:
    """
    Farthest Point Sampling 

    Args:
        num_points : จำนวนจุดที่ต้องการหลัง sampling
    """
    def __init__(self, num_points: int = 4096):
        self.num_points = num_points

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        n = len(points)
        k = self.num_points

        if n <= k:
            idx = np.random.choice(n, k, replace=True)
            return points[idx], labels[idx]

        xyz = points[:, :3].astype(np.float64)   
        selected = np.zeros(k, dtype=np.int64)
        distances = np.full(n, np.inf)

        current = np.random.randint(0, n)

        for i in range(k):
            selected[i] = current

            diff = xyz - xyz[current]                   # (N, 3)
            dist = np.einsum('ij,ij->i', diff, diff)    # (N,)  = ||diff||²
            distances = np.minimum(distances, dist)    

            current = int(np.argmax(distances))

        return points[selected], labels[selected]


class RandomSampling:
    """
    Random Sampling แบบ uniform 
    """
    def __init__(self, num_points: int = 4096):
        self.num_points = num_points

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        n = len(points)
        k = self.num_points
        replace = n < k
        idx = np.random.choice(n, k, replace=replace)
        return points[idx], labels[idx]



class Compose:
    """
    Example:
        train_transform = Compose([
            RandomRotateZ(),
            RandomJitter(sigma=0.002),
            RandomDropout(max_dropout_ratio=0.2),
            RandomColorJitter(),
            FarthestPointSampling(num_points=4096),
        ])
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, points: np.ndarray, labels: np.ndarray):
        for t in self.transforms:
            points, labels = t(points, labels)
        return points, labels

    def __repr__(self) -> str:
        lines = ["Compose("]
        for t in self.transforms:
            lines.append(f"    {t.__class__.__name__},")
        lines.append(")")
        return "\n".join(lines)



def get_train_transform(num_points: int = 4096) -> Compose:
    """
    Transform preset สำหรับ training
    rotation, noise, dropout, color variation, sampling
    """
    return Compose([
        RandomRotateZ(),
        RandomFlipHorizontal(p=0.5),
        RandomScale(lo=0.9, hi=1.1),
        RandomJitter(sigma=0.002, clip=0.005),
        RandomDropout(max_dropout_ratio=0.2),
        RandomColorJitter(brightness=0.1, contrast=0.1),
        RandomSampling(num_points=num_points),   
    ])


def get_test_transform(num_points: int = 4096) -> Compose:
    """
    Transform preset สำหรับ validation / test
    ไม่มี random augmentation — มีแค่ deterministic sampling
    """
    return Compose([
        FarthestPointSampling(num_points=num_points),  
    ])