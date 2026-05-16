import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, requires_grad=True).view(1, self.k * self.k).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=4, in_channels=13):
        super(PointNetSegmentation, self).__init__()
        
        # T-Net 3x3 
        self.tnet3 = TNet(k=3)
        
        # Local Feature MLP 1
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        
        # T-Net 64x64
        self.tnet64 = TNet(k=64)
        
        # Global Feature MLP 2
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Segmentation Head (Local + Global = 64 + 1024 = 1088)
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        
        self.sbn1 = nn.BatchNorm1d(512)
        self.sbn2 = nn.BatchNorm1d(256)
        self.sbn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        B, C, N = x.size()
        
        # Input Transform
        xyz = x[:, :3, :] # [B, 3, N]
        trans3 = self.tnet3(xyz) # [B, 3, 3]
        
        xyz_trans = torch.bmm(xyz.transpose(2, 1), trans3).transpose(2, 1)
        
        if C > 3:
            features = x[:, 3:, :] 
            x = torch.cat([xyz_trans, features], dim=1) #  [B, 13, N]
        else:
            x = xyz_trans

        # Local Features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature Transform (64 มิติ)
        trans64 = self.tnet64(x) #  [B, 64, 64]
        local_feat = torch.bmm(x.transpose(2, 1), trans64).transpose(2, 1) # [B, 64, N]
        
        # Global Feature
        x = F.relu(self.bn3(self.conv3(local_feat)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        global_feat = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        
        global_feat_expanded = global_feat.repeat(1, 1, N) # [B, 1024, N]
        concat_feat = torch.cat([local_feat, global_feat_expanded], dim=1) # [B, 1088, N]
        
        x = F.relu(self.sbn1(self.seg_conv1(concat_feat)))
        x = F.relu(self.sbn2(self.seg_conv2(x)))
        x = F.relu(self.sbn3(self.seg_conv3(x)))
        out = self.seg_conv4(x) # [B, 4, N]
        
        return out, trans64