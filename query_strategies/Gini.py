import copy

import numpy as np
import torch
from .strategy import Strategy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
class Gini(Strategy):
    def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
        super(Gini, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)
        self.pca = PCA(n_components=2)
        self.tsne = TSNE(n_components=2)
        #self.umap = umap.UMAP()

    def gini(self,probs):
        # 计算Gini系数
        return 1 - np.sum(probs ** 2)

    def calculate_probs(self,idxs):
        # 基于给定的索引计算类别分布
        return np.bincount(idxs) / len(idxs)

    def query(self,n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]#没有被选中的标
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        X_labeled = self.X[idxs_labeled]#被选中的数据 数据大小为(5000,32,32,3)
        Y_labeled = self.Y[idxs_labeled]#被选中的数据标签
        X_unlabeled = self.X[idxs_unlabeled]#未被选中的数据 数据大小为(24750,32,32,3)
        Y_unlabeled = self.Y[idxs_unlabeled]
        probs_unlabled,P = self.predict_prob(X_unlabeled,Y_unlabeled)
        labeled_quantity = len(Y_labeled)
        unlabled_quantity = len(Y_unlabeled)
        probs_labeled = self.calculate_probs(Y_labeled)

        probs_unlabeled = probs_unlabled.numpy()
        selected_idxs = set()
        idxs_unlabeled_set = set(idxs_unlabeled)
        pbar = tqdm(total=n, desc="Selecting samples")
        while len(selected_idxs)<n:
            gini_max = -1
            max_idx = 0
            max_i = 0
            for i in range(len(idxs_unlabeled)):
                if idxs_unlabeled[i] not in selected_idxs:
                    probs_new = (probs_labeled * len(Y_labeled) + probs_unlabeled[i]) / (len(Y_labeled) + 1)
                    current_gini = self.gini(probs_new)
                    if current_gini>gini_max:
                        max_idx = idxs_unlabeled[i]
                        gini_max = current_gini
                        max_i = i
            selected_idxs.add(max_idx)
            probs_labeled = (probs_labeled * len(Y_labeled) + probs_unlabeled[max_i]) / (len(Y_labeled) + 1)
            pbar.update(1)

        if "imbalance" in self.method:
            return np.array(list(selected_idxs))


        elif "both" in self.method:
            probs, P = self.predict_prob(X_unlabeled, Y_unlabeled)
            max_probs = [probs[i, P[i]].item() for i in range(len(P))]
            max_probs = np.column_stack([idxs_unlabeled, max_probs])
            result = np.array(sorted(max_probs, key=lambda x: x[1]))
            score_probs = copy.deepcopy(result[:n])
            alpha = 0.5
            beta = 1 - alpha
            idxs_gini = np.array(list(selected_idxs))
            idxs_probability = result[:n, 0]
            X_gini = self.X[idxs_gini]#被选中的数据 数据大小为(5000,32,32,3)
            Y_gini = self.Y[idxs_gini]
            probs_gini, P_gini = self.predict_prob(X_gini, Y_gini)
            score_gini = [probs_gini[i, P_gini[i]].item() for i in range(len(P_gini))]
            score_gini = np.column_stack([idxs_gini, score_gini])
            result_dict = {}

            # 遍历 list1 和 list2
            index = 1
            for l in [score_probs, score_gini]:
                for item in l:
                    # 如果 key 已经在字典中，将 value 加到已有的 value 上
                    if index == 1:
                        if item[0] in result_dict:
                            result_dict[item[0]] += alpha * item[1]
                        # 否则，将该 key-value 对添加到字典中
                        else:
                            result_dict[item[0]] = alpha * item[1]
                    else:
                        if item[0] in result_dict:
                            result_dict[item[0]] += beta * item[1]
                        # 否则，将该 key-value 对添加到字典中
                        else:
                            result_dict[item[0]] = beta * item[1]

                index +=1
            result_dict = sorted(result_dict.items(), key=lambda item: item[1])
            # print(result_dict)
            sorted_items = [item[0] for item in result_dict[:n]]
            return np.array(sorted_items)