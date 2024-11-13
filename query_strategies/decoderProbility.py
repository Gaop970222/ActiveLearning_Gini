import numpy as np
import torch
from .strategy import Strategy
import os
import pickle
import pdb
import matplotlib.pyplot as plt
from random import *
import random
import cvxpy as cp
import copy
from collections import Counter
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

class DecoderProbility(Strategy):
    def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
        super(DecoderProbility, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)
        self.pca = PCA(n_components=2)
        self.tsne = TSNE(n_components=2)
        #self.umap = umap.UMAP()

    def query(self,n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]#没有被选中的标
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        X_labeled = self.X[idxs_labeled]#被选中的数据 数据大小为(5000,32,32,3)
        Y_labeled = self.Y[idxs_labeled]#被选中的数据标签
        X_unlabeled = self.X[idxs_unlabeled]#未被选中的数据 数据大小为(24750,32,32,3)
        Y_unlabeled = self.Y[idxs_unlabeled]
        probs,P = self.predict_prob(X_unlabeled,Y_unlabeled)
        unlable_embeddings = self.get_embeddings()
        labeled_probs, labeled_P = self.predict_prob(X_labeled, Y_labeled)
        label_embeddings = self.get_embeddings()

        P = P.long()
        max_probs = [probs[i,P[i]].item() for i in range(len(P))]
        # log_probs = torch.log(probs)
        # U = (probs*log_probs).sum(1)
        
        if self.dataset == 'cifar10':
            num_classes=10
        elif self.dataset == 'cifar100':
            num_classes=100

        if "density" in self.method:
            # label_embeddings = self.tsne.fit_transform(label_embeddings)
            # unlable_embeddings = self.tsne.fit_transform(unlable_embeddings)
            # 将之前的数据拿出来预测后归纳 然后放到一个字典里 看谁的数量最少 接下来将unlabled的归纳 按照p的顺序排序然后进行选择
            kde = KernelDensity(kernel='exponential', bandwidth='silverman').fit(label_embeddings)
            log_density = kde.score_samples(unlable_embeddings)
            normalized_density = (log_density - min(log_density)) / (max(log_density) - min(log_density))
            density = np.exp(log_density).astype(np.longdouble)
            top_n = sorted(enumerate(density), key=lambda x: x[1], reverse=False)[:n]
            top_nor = sorted(enumerate(normalized_density), key=lambda x: x[1], reverse=False)[:n]
            indexes = [i for i, v in top_n]
            indexes_nor = [i for i, v in top_nor]
            return np.array(indexes_nor)

        elif "GMM" in self.method:
            # label_embeddings = self.tsne.fit_transform(label_embeddings)
            # unlable_embeddings = self.tsne.fit_transform(unlable_embeddings)
            # 将之前的数据拿出来预测后归纳 然后放到一个字典里 看谁的数量最少 接下来将unlabled的归纳 按照p的顺序排序然后进行选择
            gmm = GaussianMixture(n_components=num_classes)
            gmm.fit(label_embeddings)
            log_density = gmm.score_samples(unlable_embeddings)
            normalized_density = (log_density - min(log_density)) / (max(log_density) - min(log_density))
            density = np.exp(log_density).astype(np.longdouble)
            top_n = sorted(enumerate(density), key=lambda x: x[1], reverse=False)[:n]
            top_nor = sorted(enumerate(normalized_density), key=lambda x: x[1], reverse=True)[:n]
            indexes = [i for i, v in top_n]
            indexes_nor = [i for i, v in top_nor]
            return np.array(indexes_nor)


        elif "both" in self.method:
            max_probs = np.column_stack([idxs_unlabeled, max_probs])
            result = np.array(sorted(max_probs, key=lambda x: x[1]))
            result_d = {item[0]: item[1] for item in result}
            alpha = 0.75
            beta = 1 - alpha
            label_embeddings = self.tsne.fit_transform(label_embeddings)
            unlable_embeddings = self.tsne.fit_transform(unlable_embeddings)
            kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(label_embeddings)
            log_density = kde.score_samples(unlable_embeddings)
            normalized_density = (log_density - min(log_density)) / (max(log_density) - min(log_density))
            max_normalized_density = np.column_stack([idxs_unlabeled, normalized_density])
            density_d = {item[0]: item[1] for item in max_normalized_density}
            result = {key: beta * density_d.get(key, 0) + alpha * result_d.get(key, 0) for key in set(density_d) | set(result_d)}
            result = sorted(result.items(), key=lambda x: x[1])
            result = np.array(result)
            return result[:n, 0]

        elif 'imbalance' in self.method:
            max_probs = np.column_stack([idxs_unlabeled,max_probs])
            result = np.array(sorted(max_probs,key = lambda x:x[1]))
            return result[:n,0]
            
        elif 'optimal' in self.method:
            b=n
            N=len(idxs_unlabeled)
            L1_DISTANCE=[]
            L1_Loss=[]
            ENT_Loss=[]
            probs = probs.numpy()
            U = U.numpy()
            # Adaptive counts of samples per cycle
            labeled_classes=self.Y[self.idxs_lb]
            _, counts = np.unique(labeled_classes, return_counts=True)
            class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
            class_share=class_threshold-counts
            samples_share= np.array([0 if c<0 else c for c in class_share]).reshape(num_classes,1)
            if self.dataset == 'cifar10':
                lamda=0.6
            elif self.dataset == 'cifar100':
                lamda=2

            for lam in [lamda]:

                z=cp.Variable((N,1),boolean=True)
                constraints = [sum(z) == b]
                cost = z.T @ max_probs + lam * cp.norm1(probs.T @ z - samples_share)
                objective = cp.Minimize(cost)
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.GUROBI, verbose=True, TimeLimit=1000)
                print('Optimal value with gurobi : ', problem.value)
                print(problem.status)
                print("A solution z is")
                print(z.value.T)
                lb_flag = np.array(z.value.reshape(1, N)[0], dtype=bool)
                # -----------------Stats of optimization---------------------------------
                ENT_Loss.append(np.matmul(z.value.T, U))
                print('ENT LOSS= ', ENT_Loss)
                threshold = (2 * n / num_classes) + (self.cycle + 1) * n / num_classes
                round=self.cycle+1
                freq = torch.histc(torch.FloatTensor(self.Y[idxs_unlabeled[lb_flag]]), bins=num_classes)+torch.histc(torch.FloatTensor(self.Y[self.idxs_lb]), bins=num_classes)
                L1_distance = (sum(abs(freq - threshold)) * num_classes / (2 * (2 * n + round * n) * (num_classes - 1))).item()
                print('Lambda = ',lam)
                L1_DISTANCE.append(L1_distance)
                L1_Loss_term=np.linalg.norm(np.matmul(probs.T,z.value) - samples_share, ord=1)
                L1_Loss.append(L1_Loss_term)

            print('L1 Loss = ')
            for i in L1_Loss:
                print('%.3f' %i)
            print('L1_distance = ')
            for j in L1_DISTANCE:
                print('%.3f' % j)
            print('ENT LOSS = ')
            for k in ENT_Loss:
                print('%.3f' % k)
            return idxs_unlabeled[lb_flag]

    
    