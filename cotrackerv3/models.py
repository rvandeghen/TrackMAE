import torch
import torch.nn as nn


class CoTrackerFF(nn.Module):
    def __init__(self, grid_size = 14):
        super().__init__()
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.grid_size = grid_size
    

    def forward(self, video):
        pred_tracks, pred_visibility = self.cotracker(video, grid_size=self.grid_size)
        return pred_tracks, pred_visibility