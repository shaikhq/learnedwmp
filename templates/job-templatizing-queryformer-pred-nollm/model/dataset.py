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
        self.costs = raw_costs
        self.cost_labels = cost_labels

        # Create collated dictionaries for each item
        self.collated_dicts = [
            {'x': x[i], 'attn_bias': attn_bias[i], 'rel_pos': rel_pos[i], 'heights': heights[i]}
            for i in range(length)
        ]

        logging.info('PlanTreeDataset initialized')

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):  # Handle slicing
            sliced_data = {
                'x': [item['x'] for item in self.collated_dicts[idx]],
                'attn_bias': [item['attn_bias'] for item in self.collated_dicts[idx]],
                'rel_pos': [item['rel_pos'] for item in self.collated_dicts[idx]],
                'heights': [item['heights'] for item in self.collated_dicts[idx]],
                'cost_labels': self.cost_labels[idx],
                'costs': self.costs[idx],
            }
            sliced_length = len(range(*idx.indices(self.length)))
            return PlanTreeDataset(
                sliced_length,
                sliced_data['x'],
                sliced_data['attn_bias'],
                sliced_data['rel_pos'],
                sliced_data['heights'],
                sliced_data['cost_labels'],
                sliced_data['costs'],
            )
        return self.collated_dicts[idx], (self.cost_labels[idx], self.costs[idx])

