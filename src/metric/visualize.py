import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []

    def update(self, epoch, train_loss, test_loss=None, train_acc=None, test_acc=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        
        if test_loss is not None: self.test_losses.append(test_loss)
        if train_acc is not None: self.train_accs.append(train_acc)
        if test_acc is not None: self.test_accs.append(test_acc)

    def plot(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- กราฟ Loss ---
        axes[0].plot(self.epochs, self.train_losses, label='Train Loss', color='#1f77b4', marker='o', markersize=4)
        if self.test_losses:
            axes[0].plot(self.epochs, self.test_losses, label='Test Loss', color='#ff7f0e', linestyle='--', marker='x', markersize=4)
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.7)

        # --- กราฟ Accuracy ---
        if self.train_accs:
            axes[1].plot(self.epochs, self.train_accs, label='Train Acc', color='#2ca02c', marker='o', markersize=4)
        if self.test_accs:
            axes[1].plot(self.epochs, self.test_accs, label='Test Acc', color='#d62728', linestyle='--', marker='x', markersize=4)
        axes[1].set_title('Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()