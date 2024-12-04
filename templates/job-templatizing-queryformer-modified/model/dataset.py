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
        #ipdb.set_trace()

        logging.info('Initializing PlanTreeDataset')
            
        # length is the number of queries in the training set
        self.length = length
            
        self.to_predict = 'cost'
        self.costs = torch.from_numpy(np.array(raw_costs))
        self.gts = self.costs
        self.cost_labels = cost_labels
        
        my_dict1 = {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }
        
        self.collated_dicts = []
        self.collated_dicts.append(my_dict1)

        logging.info('PlanTreeDataset initialized')



    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.collated_dicts[idx], (self.cost_labels[idx], self.costs[idx])
    
    def old_getitem(self, idx):
        return self.dicts[idx], self.labels[idx]