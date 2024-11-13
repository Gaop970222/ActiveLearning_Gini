import numpy as np
import torch
from .strategy import Strategy
import os
import pickle
import pdb
import matplotlib.pyplot as plt
from random import *
import random
import copy


class Nearest(Strategy):
    def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
        super(Nearest, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)
    
    def query(self,n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]#没有被选中的标
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        data_labeled = self.X[idxs_labeled]#被选中的数据 数据大小为(5000,32,32,3)
        data_unlabeled = self.X[idxs_unlabeled]#未被选中的数据 数据大小为(24750,32,32,3)
        embedding_labeled = self.get_embedding_resnet(self.X[idxs_labeled], self.Y[idxs_labeled])#数据大小为(5000,512)
        embedding_unlabeled = self.get_embedding_resnet(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])#数据大小为(24750,512)
        dist = self.calculate_distance(embedding_labeled,embedding_unlabeled).numpy()
        dist = np.column_stack([idxs_unlabeled,dist])
        result = np.array(sorted(dist,key = lambda x:x[1])).astype(int)
        return result[:n,0]
        
    