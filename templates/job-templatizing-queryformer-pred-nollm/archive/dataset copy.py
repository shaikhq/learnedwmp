import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
import logging
import ipdb

class PlanTreeDataset(Dataset):
    def __init__(self, length, x, attn_bias, rel_pos, heights, cost_labels, raw_costs):
        logging.info('Initializing PlanTreeDataset')

        # Validate inputs
        if not all(len(data) == length for data in [x, attn_bias, rel_pos, heights, cost_labels, raw_costs]):
            raise ValueError("All input data must have the same length as the 'length' parameter.")

        # Initialize attributes
        self.length = length
        self.to_predict = 'cost'
        # self.costs = torch.from_numpy(np.array(raw_costs))  # Convert raw_costs to tensor
        self.costs = raw_costs
        self.cost_labels = cost_labels  # Cost labels
        
        # Create collated dictionaries for each item
        self.collated_dicts = [
            {'x': x[i], 'attn_bias': attn_bias[i], 'rel_pos': rel_pos[i], 'heights': heights[i]}
            for i in range(length)
        ]

        logging.info('PlanTreeDataset initialized')

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.collated_dicts[idx], (self.cost_labels[idx], self.costs[idx])
