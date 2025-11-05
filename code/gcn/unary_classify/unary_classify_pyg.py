import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import random
from data_preprocessing_1 import train_test_idx
from utils_5 import _select_threshold, find_rules_ut, metrics, find_same_etype
from itertools import combinations
import warnings
import os

warnings.filterwarnings('ignore')

class UnaryClassifyPyG(torch.nn.Module):
    def __init__(self, in_feat, h_dim, out_dim, num_rels, num_bases=-1, num_hidden_layers=1, dropout=0.5, use_cuda=False):
        super(UnaryClassifyPyG, self).__init__()
        self.in_feat = in_feat
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        
        # Create RGCN layers
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(RGCNConv(in_feat, h_dim, num_rels, num_bases=num_bases))
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(RGCNConv(h_dim, h_dim, num_rels, num_bases=num_bases))
        
        # Output layer
        self.layers.append(RGCNConv(h_dim, out_dim, num_rels, num_bases=num_bases))
        
        self.dropout_layer = torch.nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i < len(self.layers) - 1:  # Don't apply activation to output layer
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x

def similar_loss(data, node_idx, logits, direction='in'):
    all_loss = 0
    for i in node_idx:
        if direction == 'in':
            # Find incoming edges
            mask = data.edge_index[1] == i
            neighbors = data.edge_index[0][mask]
            edge_types = data.edge_type[mask]
        else:
            # Find outgoing edges
            mask = data.edge_index[0] == i
            neighbors = data.edge_index[1][mask]
            edge_types = data.edge_type[mask]
        
        if len(neighbors) == 0:
            continue
            
        # Group by edge type
        unique_types = torch.unique(edge_types)
        loss = 0
        for etype in unique_types:
            type_mask = edge_types == etype
            sm_nodes = neighbors[type_mask]
            
            if len(sm_nodes) < 2:
                continue
                
            # Calculate similarity loss for nodes of same edge type
            cnt = 0
            l = 0
            for n1, n2 in combinations(sm_nodes.tolist(), 2):
                cnt += 1
                l += torch.norm(logits[n1] - logits[n2], 1)
            if cnt > 0:
                loss += l / cnt
        all_loss += loss
    return all_loss / len(node_idx) if len(node_idx) > 0 else 0

def main(args):
    if not os.path.exists('result/'):
        os.mkdir('result/')

    if not os.path.exists('rules/'):
        os.mkdir('rules/')

    fold = 10
    result = []
    sigma = args.sigma
    beta = args.beta
    output = str(args) + '\n'
    true_rules_out = ''
    pred_rules_out = ''
    
    for i in range(fold):
        path = 'dataset/' + args.dataset + '/10_fold/'
        num_node, edge_list, edge_src, edge_dst, edge_type, edge_norm, num_rel, train_idx, test_idx, train_label, test_label, node_features = train_test_idx(path, i, 'analogy', args.n_hidden)

        tmp_label = train_label
        train_idx = list(train_idx)
        test_template_id = np.where(test_label.sum(axis=0) > 0)[1]

        if test_label.shape[0] == 0:
            continue
            
        edge_type = torch.from_numpy(edge_type).long()
        edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
        node_features = torch.from_numpy(np.array(node_features)).float()

        # check cuda
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.gpu)
            edge_type = edge_type.cuda()
            edge_norm = edge_norm.cuda()
            node_features = node_features.cuda()

        # create PyTorch Geometric data object
        edge_index = torch.stack([torch.from_numpy(edge_src), torch.from_numpy(edge_dst)], dim=0).long()
        if use_cuda:
            edge_index = edge_index.cuda()
            
        data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type)

        in_feat = node_features.shape[1]
        num_classes = train_label.shape[1]

        train_label = torch.from_numpy(np.array(train_label)).float()
        test_label = torch.from_numpy(test_label.toarray()).float()
        
        # create model
        model = UnaryClassifyPyG(in_feat,
                                 args.n_hidden,
                                 num_classes,
                                 num_rel,
                                 num_bases=args.n_bases,
                                 num_hidden_layers=args.n_layers - 2,
                                 dropout=args.dropout,
                                 use_cuda=use_cuda)
        if use_cuda:
            model.cuda()
            train_label.cuda()
            test_label.cuda()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        if args.validation:
            val_idx = sorted(random.sample(list(train_idx), num_node // 9))  # 10% for validation
            train_idx = sorted(list(set(train_idx) - set(val_idx)))
        else:
            val_idx = train_idx
        val_labels = train_label[val_idx]
        train_label = train_label[train_idx]
        
        print("start training...")
        model.train()
        for epoch in range(args.n_epochs):
            optimizer.zero_grad()
            logits = model.forward(data)

            # similar loss
            sim_loss_in = similar_loss(data, train_idx, logits, 'in')
            sim_loss_out = similar_loss(data, train_idx, logits, 'out')
            classify_loss = criterion(logits[train_idx], train_label)  # for multi-label classification
            loss = classify_loss + sigma * sim_loss_in + beta * sim_loss_out
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

        model.eval()
        logits = model.forward(data).cpu()
        logits = F.sigmoid(logits)
        best_thred, _ = _select_threshold(val_labels.cpu(), logits[val_idx].detach().numpy())

        labels_test_pre = np.zeros(test_label.shape)
        labels_test_pre[np.where(logits[test_idx].detach().numpy() > best_thred)] = 1

        test_rules, str_terule = find_rules_ut(path, i, test_idx, tmp_label, test_label, test_template_id)
        pred_rules, str_prerule = find_rules_ut(path, i, test_idx, tmp_label, labels_test_pre, test_template_id, pred=True)

        true_rules_out += 'fold ' + str(i) + '\n'
        true_rules_out += str_terule
        pred_rules_out += 'fold ' + str(i) + '\n'
        pred_rules_out += str_prerule

        test_precision, test_recall, test_f1 = metrics(set(test_rules), set(pred_rules))
        result.append([test_precision, test_recall, test_f1])
        print("Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1: {:.4f}".format(test_precision, test_recall, test_f1))
        output += "Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1: {:.4f} \n".format(test_precision, test_recall, test_f1)

    mean_p, mean_r, mean_f1 = np.mean(np.array(result), axis=0)

    print("Mean values over " + str(fold) + " fold: Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(mean_p, mean_r, mean_f1))
    output += "Mean values over " + str(fold) + " fold: Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}\n".format(mean_p, mean_r, mean_f1)

    file_name = 'pyg_' + args.dataset + '.txt'

    f = open('./result/' + file_name, 'w', encoding='utf-8')
    f.write(output)
    f.close()

    f = open('./rules/true_' + file_name, 'w', encoding='utf-8')
    f.write(true_rules_out)
    f.close()

    f = open('./rules/pred_' + file_name, 'w', encoding='utf-8')
    f.write(pred_rules_out)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN with PyTorch Geometric')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='transport',
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="similar loss in coef")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="similar loss out coef")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = 0
    main(args)
