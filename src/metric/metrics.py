import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class Evaluator:
    def __init__(self, model, test_loader, num_classes=5, class_names=None):
        """
        Args:
            model: โมเดล PointNet 
            test_loader: DataLoader สำหรับชุด Test
            num_classes: จำนวน Class ทั้งหมด
            class_names: List ชื่อของแต่ละ Class 
        """
        self.model = model
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            self.class_names = class_names
            
    def evaluate(self):
        self.model.eval() 
        
        total_correct = 0
        total_seen = 0
        
        tps = torch.zeros(self.num_classes).to(self.device)
        fps = torch.zeros(self.num_classes).to(self.device)
        fns = torch.zeros(self.num_classes).to(self.device)
        
        with torch.no_grad(): 
            for points, labels in self.test_loader:
                points = points.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)
                
                points = points.permute(0, 2, 1)
                
                outputs, _ = self.model(points)
                
                preds = torch.max(outputs, 1)[1] # Shape: [Batch, Num_Points]
                
                total_correct += (preds == labels).sum().item()
                total_seen += labels.numel()
                
                for c in range(self.num_classes):
                    pred_c = (preds == c)
                    label_c = (labels == c)
                    
                    tps[c] += (pred_c & label_c).sum()
                    fps[c] += (pred_c & ~label_c).sum()
                    fns[c] += (~pred_c & label_c).sum()
        
        overall_accuracy = (total_correct / total_seen) * 100
        
        ious = tps / (tps + fps + fns + 1e-6)
        ious = ious.cpu().numpy() * 100 
        
        mean_iou = np.mean(ious)

        self._print_report(overall_accuracy, ious, mean_iou)
        
        return overall_accuracy, mean_iou, ious
    
    def _print_report(self, oa, ious, miou):
        print("\n" + "="*50)
        print("Evaluation Report")
        print("="*50)
        print(f"Overall Accuracy: {oa:.2f}%")
        print(f"Mean IoU :                  {miou:.2f}%")
        print("-" * 50)
        print("Class-wise IoU:")
        for i, name in enumerate(self.class_names):
            print(f"   - {name:<15}: {ious[i]:.2f}%")
        print("="*50)
        
class PlusEvaluator:
    def __init__(self, model, test_loader, num_classes=5, class_names=None):
        """
        Args:
            model: โมเดล PointNet++ ที่ผ่านการเทรนมาแล้ว
            test_loader: DataLoader สำหรับชุด Test
            num_classes: จำนวน Class ทั้งหมด 
            class_names: List ชื่อของแต่ละ Class 
        """
        self.model = model
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            self.class_names = class_names
            
    def evaluate(self):
        self.model.eval() 
        
        total_correct = 0
        total_seen = 0
        
        tps = torch.zeros(self.num_classes).to(self.device)
        fps = torch.zeros(self.num_classes).to(self.device)
        fns = torch.zeros(self.num_classes).to(self.device)
        
        with torch.no_grad(): 
            for points, labels in self.test_loader:
                points = points.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)
                
                
                outputs = self.model(points)
                
                preds = torch.max(outputs, 1)[1] # Shape: [Batch, Num_Points]
                
                total_correct += (preds == labels).sum().item()
                total_seen += labels.numel()
                
                for c in range(self.num_classes):
                    pred_c = (preds == c)
                    label_c = (labels == c)
                    
                    tps[c] += (pred_c & label_c).sum()
                    fps[c] += (pred_c & ~label_c).sum()
                    fns[c] += (~pred_c & label_c).sum()
        
        overall_accuracy = (total_correct / total_seen) * 100
        
        ious = tps / (tps + fps + fns + 1e-6)
        ious = ious.cpu().numpy() * 100 
        
        mean_iou = np.mean(ious)

        self._print_report(overall_accuracy, ious, mean_iou)
        
        return overall_accuracy, mean_iou, ious

    def _print_report(self, overall_accuracy, ious, mean_iou):
        print("="*50)
        print("Evaluation Report".center(50))
        print("="*50)
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"Mean IoU :        {mean_iou:.2f}%")
        print("-" * 50)
        print("Class-wise IoU:")
        for i, name in enumerate(self.class_names):
            print(f"   - {name:<15}: {ious[i]:.2f}%")
        print("="*50)