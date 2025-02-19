import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from torch.optim.lr_scheduler import MultiStepLR
import os
import pickle
from Cutout.model.resnet import ResNet18
from torchvision import transforms
from sklearn.neighbors import KernelDensity


#class Strategy:
class Strategy(object):
    def __init__(self, X, Y, X_te, Y_te, cycle , dataset, method,idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.cycle=cycle
        self.dataset=dataset
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.method=method
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n):
        pass

    def query_pseudoLabel(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, loader_tr, optimizer):
        self.clf.train()
        total_loss = 0
        total_samples = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x) #resnet
            loss = F.cross_entropy(out, y.long())
            total_loss += loss.item() * y.size(0)  # accumulate batch loss
            total_samples += y.size(0)
            sys.stdout.write('\r')
            sys.stdout.write('Loss : %3f ' % loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = total_loss / total_samples
        return epoch_loss

    def train(self):
        ## resenet from Cutout train from scratch
        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'cifar100':
            num_classes = 100
        net = ResNet18(num_classes=num_classes)#狗日的这里重新构建了
        self.net = net
        self.clf = self.net.to(self.device)
        n_epoch = self.args['n_epoch']
        optimizer = optim.SGD(
            self.clf.parameters(),
            lr=0.001,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[60, 80], gamma=0.5)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        print('number of training samples : ', len(idxs_train))

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
                            shuffle=True, **self.args['loader_tr_args'])

        epoch_losses = []
        for epoch in range(1, n_epoch+1):
            sys.stdout.write('Epoch %3d' % epoch)
            epoch_loss = self._train(loader_tr, optimizer)
            epoch_losses.append(epoch_loss)
            scheduler.step(epoch)
            for pg in optimizer.param_groups:
                print('lr = ',pg['lr'])

        if 'Random' in self.method:
            save_point = os.path.join(self.dataset+ '_checkpoints_active_set')
            if not os.path.exists(save_point):
              os.makedirs(save_point)
            torch.save({
              'epoch': epoch,
              'model_state_dict': self.clf.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()},
              os.path.join(save_point, 'checkpoint_cycle_' + str(self.cycle) + '.t7'))
            print('checkpoint saved ...')
        return epoch_losses

    def predict(self, X, Y):

        print('inference on ', len(X), ' samples')
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                #out, e1 = self.clf(x) #vgg
                out, e1= self.clf(x) # resnet
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P

    def test(self, X, Y):
        if self.cycle > 0 :
            print('test the current model ...')
        elif self.cycle == 0: # only for cycle 0
            self.clf = self.net.to(self.device)
            print('loading checkpoint of cycle 0 for testing...')
            net = torch.load(
                os.path.join(self.dataset+ '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
            self.clf.load_state_dict(net['model_state_dict'])

        print('inference on ', len(X), ' samples')
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        loader_te = DataLoader(self.handler(X,Y, transform= transforms.Compose([transforms.ToTensor(), normalize])))
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x) #resnet
                pred = out.max(1)[1]
                P[idxs] = pred.int().cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])
        embeddings = np.empty((0, 512))
        self.clf.eval()
        if self.dataset=='cifar100':
            num_classes=100
        elif self.dataset=='cifar10':
            num_classes=10
        probs = torch.zeros([len(Y), num_classes])
        P = torch.zeros(len(Y), dtype=torch.int32)
        logits= torch.zeros([len(Y), num_classes])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x) #Resnet
                pred = out.max(1)[1].int()
                P[idxs] = pred.cpu()
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
                logits[idxs]=out.cpu()
                embeddings = np.vstack((embeddings,e1.cpu().numpy()))
        self.set_embedding(embeddings)
        return probs, P

    def set_embedding(self,embeddings):
        self.embeddings = embeddings
    def get_embeddings(self):
        return self.embeddings

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()#训练一遍
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        #probs 的大小为10*24750*100
        for i in range(n_drop):#运行十次
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():#因为数据会做随机裁剪和翻转 所以最后NN的值数出的会不一样
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x) # resnet
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()#将十次的结果相加
                
        return probs#返回这个probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        return embedding

    def get_embedding_resnet(self,X,Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), 512])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        return embedding
    
    def calculate_distance(self,x,y):
        m = x.size(0)
        n = y.size(0)
        e = x.size(1)
        a1 = torch.sum((x ** 2), 1, keepdim=True).expand(m, n)
        b2 = (y ** 2).sum(1).expand(m, n)
        dist = (a1 + b2 - 2 * (x @ y.T)).float().sqrt()
        dist = dist.transpose(0,1)
        result_max = torch.max(dist,dim = 1)
        dist_max = result_max[0]
        return dist_max
        
        
        
        
        
    def test_prior_balanced(self, X_te,Y_te,Y_tr ):

        if self.cycle > 0 :
            print('test the current model ...')
        elif self.cycle == 0: # only for cycle 0
            self.clf = self.net.to(self.device)
            print('loading checkpoint of cycle 0 for testing...')
            net = torch.load(
                os.path.join(self.dataset+ '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
            self.clf.load_state_dict(net['model_state_dict'])
        print('inference on ', len(Y_te), ' samples')

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        loader_te = DataLoader(self.handler(X_te,Y_te, transform= transforms.Compose([transforms.ToTensor(), normalize])))
        self.clf.eval()
        P = torch.zeros(len(Y_te), dtype=Y_te.dtype)

        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'cifar100':
            num_classes = 100

        freq_np,_ = np.histogram(Y_tr[self.idxs_lb], bins=num_classes)
        freq=torch.from_numpy(freq_np).cuda()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                balanced_prob = prob / freq
                _, predicted = torch.max(balanced_prob, 1)
                P[idxs] = predicted.cpu()

        return P
