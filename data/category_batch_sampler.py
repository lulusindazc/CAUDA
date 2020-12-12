# coding=utf-8
import numpy as np
import torch
# from torch.utils.data import Sampler


class CategoryBatchSampler(object):
    '''
    CategoryBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self,num_classes, classes_per_it,iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(CategoryBatchSampler, self).__init__()
        self.num_classes =num_classes
        self.classes_per_it = classes_per_it
        self.iterations = iterations


    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        # spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):

            c_idxs = torch.randperm(self.num_classes)[:cpi].tolist()

            yield c_idxs

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
