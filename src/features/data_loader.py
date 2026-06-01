from .feature import HeritagePointCloudDataset
from torch.utils.data import DataLoader, random_split , Subset
import torch
import copy


class ArchaeologicalLoader:
    def __init__(self, dataset, train_transform=None, test_transform=None, 
                 batch_size=4, shuffle=True, test_split=0.2, 
                 val_enabled=False, val_split=0.1, num_workers=0):
        
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_transform = train_transform
        self.test_transform = test_transform  
        self.shuffle = shuffle
        self.test_split = test_split
        self.val_enabled = val_enabled
        self.val_split = val_split
        self.num_workers = num_workers

    def get_loaders(self):
        full_size = len(self.dataset)
        test_size = int(full_size * self.test_split)
        
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(full_size, generator=generator).tolist()
        
        train_base_ds = copy.deepcopy(self.dataset)
        test_base_ds = copy.deepcopy(self.dataset)
        
        train_base_ds.transform = self.train_transform
        test_base_ds.transform = self.test_transform
        
        if self.val_enabled:
            val_size = int(full_size * self.val_split)
            train_size = full_size - test_size - val_size
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
            
            val_base_ds = copy.deepcopy(self.dataset)
            val_base_ds.transform = self.test_transform
            
            train_ds = Subset(train_base_ds, train_idx)
            val_ds = Subset(val_base_ds, val_idx)
            test_ds = Subset(test_base_ds, test_idx)
            
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=self.shuffle, 
                num_workers=self.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False, 
                num_workers=self.num_workers, pin_memory=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False, 
                num_workers=self.num_workers, pin_memory=True
            )
            
            return train_loader, val_loader, test_loader
            
        else:
            train_size = full_size - test_size
            
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            
            train_ds = Subset(train_base_ds, train_idx)
            test_ds = Subset(test_base_ds, test_idx)
            
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=self.shuffle, 
                num_workers=self.num_workers, pin_memory=True, drop_last=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False, 
                num_workers=self.num_workers, pin_memory=True
            )
            
            return train_loader, test_loader