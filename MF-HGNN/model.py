from torch_geometric.nn import ChebConv,TransformerConv
from dataload import dataloader
from dataload import get_multi_atten
from opt import *
import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGPooling
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
opt = OptInit().initialize()

#############
from torch_geometric.nn import TopKPooling
from brainmsgpassing import MyMessagePassing
from torch_geometric.typing import (OptTensor)
from torch.nn import Parameter
from torch_geometric.utils import add_remaining_self_loops,softmax

from einops import rearrange
import leidenalg
import igraph as ig
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from scipy.stats import pearsonr

class MyGINConvWithMean(MyMessagePassing):
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=False, eps=0.0,
                 **kwargs):
        super(MyGINConvWithMean, self).__init__(aggr='add', **kwargs)  # Use 'add' for GIN

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.epsilon = Parameter(torch.Tensor(1).fill_(eps))  # Learnable epsilon

        # Existing nn for initial weight transformation
        self.nn = nn

        mlp_update = torch.nn.Sequential(torch.nn.Linear(self.out_channels, self.out_channels * 2),
                                         # torch.nn.BatchNorm1d(self.out_channels * 2),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(self.out_channels * 2, self.out_channels))
        self.mlp_update = mlp_update

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()
        self.epsilon.data.fill_(0.0)  # Reset epsilon to initial value

    def forward(self, x, edge_index, edge_weight=None, pseudo=None, size=None, attn = None):
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(0))

        # Apply the nn to transform input features
        weight = self.nn(pseudo.to(opt.device)).view(-1, self.in_channels, self.out_channels)

        # Transform node features with the weights
        if torch.is_tensor(x):
            x_transformed = torch.matmul(x.unsqueeze(1), weight).squeeze(1)  # W_i * h_i
        else:
            x_transformed = (
                None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
                None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1)
            )

        # Apply message passing
        if attn is None:
            aggr_out = self.propagate(edge_index, size=size, x=x_transformed, edge_weight=edge_weight)
        else:
            new_x_transformed = x_transformed + torch.matmul(attn, x_transformed)
            aggr_out = self.propagate(edge_index, size=size, x= new_x_transformed , edge_weight=edge_weight)

        # Combine with transformed node features and apply MLP
        updated_features = (1 + self.epsilon) * x_transformed + aggr_out
        out = self.mlp_update(updated_features)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
        # Apply edge weight normalization if provided
        if edge_weight is not None:
            edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        # Apply the learnable bias
        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        # MLP is applied in the forward function
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, epsilon={:.4f})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.epsilon.item()
        )

def biased_random_walk(adj_matrix, node_scores, x, node_features_all3, perm, max_path_length=10):
    sorted_nodes = torch.argsort(node_scores, descending=True)
    sorted_nodes = sorted_nodes[torch.isin(sorted_nodes, perm)]
    sorted_nodes = sorted_nodes[:5]
    all_visited = torch.zeros(111, dtype=torch.bool, device=opt.device)

    perm_index_map = {node.item(): idx for idx, node in enumerate(perm)}
    all_softmax_scores = F.softmax(node_scores, dim=-1)

    for start_node in sorted_nodes:
        if all_visited[start_node]:
            continue
        path = [start_node]
        current_node = start_node
        visited = all_visited.clone()
        visited[start_node] = True

        while len(path) < max_path_length:
            neighbors = adj_matrix[current_node].nonzero(as_tuple=True)[0]
            valid_neighbors = neighbors[~visited[neighbors]]
            if valid_neighbors.numel() == 0:
                break

            probabilities = all_softmax_scores[valid_neighbors]
            next_node = valid_neighbors[torch.argmax(probabilities).item()]
            path.append(next_node)
            visited[next_node] = True
            current_node = next_node

        if path:
            start_node_index = perm_index_map.get(start_node.item())
            if start_node_index is not None:
                path = torch.tensor(path, device=opt.device)
                path_features = node_features_all3[path]
                path_scores = node_scores[path]
                final_weights = F.softmax(path_scores, dim=0)
                x[start_node_index] = x[start_node_index] + torch.sum(final_weights.unsqueeze(1) * path_features, dim=0)

        all_visited = all_visited | visited

    return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(attn_output)
        return output

class Brain_connectomic_graph(torch.nn.Module):

    def __init__(self):
        super(Brain_connectomic_graph, self).__init__()
        self._setup()
    def _setup(self):
        self.pooling_1 = SAGPooling(20, 0.9)
        self.socre_gcn = ChebConv(111, 9, K=3, normalization='sym')
        self.pooling_2= TopKPooling(20, 0.9)

        self.weight = nn.Parameter(torch.FloatTensor(64, 20)).to(opt.device)
        self.bns=nn.BatchNorm1d(20).to(opt.device)
        nn.init.xavier_normal_(self.weight)

        self.graphnn_whole_brain_time = nn.Sequential(nn.Linear(111, 7, bias=False), nn.ReLU(), nn.Linear(7, 20 * 111))
        self.myginconv_whole_brain_time = MyGINConvWithMean(111, 20, self.graphnn_whole_brain_time, normalize=False)

        self.graphnn_whole_brain_space = nn.Sequential(nn.Linear(111, 7, bias=False), nn.ReLU(), nn.Linear(7, 20 * 111))
        self.myginconv_whole_brain_space = MyGINConvWithMean(111, 20, self.graphnn_whole_brain_space, normalize=False)

        self.graphnn_all = nn.Sequential(nn.Linear(111, 7, bias=False), nn.ReLU(), nn.Linear(7, 20 * 64))
        self.myginconv_all = MyGINConvWithMean(64, 20, self.graphnn_all, normalize=False)

        self.graphnn_all2 = nn.Sequential(nn.Linear(111, 7, bias=False), nn.ReLU(), nn.Linear(7, 20 * 20))
        self.myginconv_all2 = MyGINConvWithMean(20, 20, self.graphnn_all2, normalize=False)

        self.graphnn_all3 = nn.Sequential(nn.Linear(111, 7, bias=False), nn.ReLU(), nn.Linear(7, 20 * 20))
        self.myginconv_all3 = MyGINConvWithMean(20, 20, self.graphnn_all3, normalize=False)

        self.conv_whole_brain_k = nn.Linear(40, 40, bias=False)
        self.conv_whole_brain_q = nn.Linear(40, 40, bias=False)
        self.conv_whole_brain_v = nn.Linear(40, 40, bias=False)

        self.conv_whole_brain_logit_k = nn.Linear(24, 24, bias=False)
        self.conv_whole_brain_logit_q = nn.Linear(40, 24, bias=False)
        self.conv_whole_brain_logit_v = nn.Linear(40, 40, bias=False)

        self.conv_whole_brain_logit_k2 = nn.Linear(40, 24, bias=False)
        self.conv_whole_brain_logit_q2 = nn.Linear(24, 24, bias=False)
        self.conv_whole_brain_logit_v2 = nn.Linear(24, 24, bias=False)

        self.STAGIN = ModelSTAGIN(
            input_dim=111,
            hidden_dim=20,
            num_classes=2,
            num_heads=1,
            num_layers=1,
            sparsity=30,
            dropout=0.5,
            cls_token='sum',
            readout='sero')  # shared branch

        self.mha = MultiHeadAttention(2220, 1)
        self.whole_score = torch.nn.Linear(20, 1)

    def forward(self, data, time_serie, dynamic_fc):
        with np.errstate(divide='ignore', invalid='ignore'):
            dynamic_fc = np.arctanh(dynamic_fc)
        dynamic_fc[dynamic_fc == float('inf')] = 0
        dynamic_fc = torch.from_numpy(dynamic_fc).to(opt.device).float().unsqueeze(0)
        logit, h_bridge = self.STAGIN(dynamic_fc)

        edges, features = data.edge_index, data.x
        edges, features = edges.to(opt.device), features.to(opt.device)

        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(opt.device).to(torch.float32)

        adj=data.adj
        adj=torch.tensor(adj)
        adj=adj.float()
        adj = adj.to(opt.device)

        pos = torch.eye(features.shape[0])

        myfeatures = F.dropout(features, p=opt.dropout, training=self.training)
        node_features_whole_brain_space = torch.nn.functional.leaky_relu(self.myginconv_whole_brain_space(myfeatures, edges, edge_attr, pos))  # (111,20)

        myfeatures = F.dropout(features, p=opt.dropout, training=self.training)
        node_features_whole_brain_time = torch.nn.functional.leaky_relu(self.myginconv_whole_brain_time(myfeatures, edges, edge_attr, pos))  # (111,20)

        node_features_whole_brain_time_space = torch.cat((node_features_whole_brain_space, node_features_whole_brain_time), -1)#(111,40)

        # k2 = self.conv_whole_brain_k(node_features_whole_brain_time_space)
        # q2 = self.conv_whole_brain_q(node_features_whole_brain_time_space)
        # v2 = self.conv_whole_brain_v(node_features_whole_brain_time_space)
        # attn2 = torch.matmul(k2, q2.T)  # (111,111)
        # attn2 = attn2 / abs(attn2.min())  # 2 * 2
        # attn2 = F.normalize(attn2, dim=-1)
        # node_features_whole_brain_time_space = node_features_whole_brain_time_space + torch.matmul(attn2, v2)#(111,40)

        k2 = self.conv_whole_brain_logit_k(logit)#(111,24)
        q2 = self.conv_whole_brain_logit_q(node_features_whole_brain_time_space)#(111,24)
        v2 = self.conv_whole_brain_logit_v(node_features_whole_brain_time_space)
        attn2 = torch.matmul(k2, q2.T)  # (111,111)
        attn2 = attn2 / abs(attn2.min())  # 2 * 2
        attn2 = F.normalize(attn2, dim=-1)
        node_features_whole_brain_time_space = node_features_whole_brain_time_space + torch.matmul(attn2, v2)#(111,40)

        k2 = self.conv_whole_brain_logit_k2(node_features_whole_brain_time_space)
        q2 = self.conv_whole_brain_logit_q2(logit)
        v2 = self.conv_whole_brain_logit_v2(logit)
        attn2 = torch.matmul(k2, q2.T)  # (111,111)
        attn2 = attn2 / abs(attn2.min())  # 2 * 2
        attn2 = F.normalize(attn2, dim=-1)
        logit = logit + torch.matmul(attn2, v2)#(111,40)

        node_features_all = torch.cat((node_features_whole_brain_time_space, logit), -1)

        myfeatures = F.dropout(node_features_all, p=opt.dropout, training=self.training)
        node_features_all = torch.nn.functional.leaky_relu(self.myginconv_all(myfeatures, edges, edge_attr, pos))
        
        myfeatures = F.dropout(node_features_all, p=opt.dropout, training=self.training)
        node_features_all2 = torch.nn.functional.leaky_relu(self.myginconv_all2(myfeatures, edges, edge_attr, pos))

        # node_features_all3 = torch.nn.functional.leaky_relu(self.myginconv_all3(node_features_all2, edges, edge_attr, pos))
        node_features_all3 = node_features_all2

        adj_matrix = torch.sparse_coo_tensor(edges, edge_attr, (111, 111))
        x, edge_index, edge_attr_pooled, batch, perm, score = self.pooling_1(node_features_all3, edges, edge_attr)
        node_scores = self.whole_score(node_features_all3).squeeze()
        new_node_features = biased_random_walk(adj_matrix.to_dense(), node_scores, x, node_features_all3, perm,10)

        graph_embedding = new_node_features.view(1, -1)

        return graph_embedding

class HPG(nn.Module):
    def __init__(self):
        super(HPG, self).__init__()
        self.num_layers = 4
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        self.convs4 = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bns2 = nn.ModuleList()
        self.bns3 = nn.ModuleList()
        self.convs1.append(TransformerConv(in_channels=2000,out_channels=20,heads=1))
        self.convs2.append(TransformerConv(in_channels=2000, out_channels=20,heads=1))
        self.convs3.append(TransformerConv(in_channels=2000,out_channels=20,heads=1))
        self.convs4.append(TransformerConv(in_channels=2000, out_channels=20,heads=1))
        # self.convs2.append(GCNConv(2220, 20))
        self.bns.append(nn.BatchNorm1d(20))
        self.bns2.append(nn.BatchNorm1d(20))
        self.bns3.append(nn.BatchNorm1d(20))

        self.convs1.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs3.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs4.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        # self.convs2.append(GCNConv(20, 20))
        self.bns.append(nn.BatchNorm1d(20))
        self.bns2.append(nn.BatchNorm1d(20))
        self.bns3.append(nn.BatchNorm1d(20))

        self.convs1.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs3.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs4.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        # self.convs2.append(GCNConv(20, 20))
        self.bns.append(nn.BatchNorm1d(20))
        self.bns2.append(nn.BatchNorm1d(20))
        self.bns3.append(nn.BatchNorm1d(20))

        self.convs1.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs3.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs4.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        # self.convs2.append(GCNConv(20, 20))
        self.bns.append(nn.BatchNorm1d(20))
        self.bns2.append(nn.BatchNorm1d(20))
        self.bns3.append(nn.BatchNorm1d(20))

        self.convs1.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs3.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        self.convs4.append(TransformerConv(in_channels=40, out_channels=20, heads=1))
        # self.convs2.append(GCNConv(20, 20))
        self.bns.append(nn.BatchNorm1d(20))
        self.bns2.append(nn.BatchNorm1d(20))
        self.bns3.append(nn.BatchNorm1d(20))
        self.out_fc = nn.Linear(100, 2)

        self.a = torch.nn.Parameter(torch.Tensor(20, 1))

        self.conv13 = TransformerConv(in_channels=40, out_channels=20, heads=1)
        self.conv14 = TransformerConv(in_channels=40, out_channels=20, heads=1)
        self.conv15 = TransformerConv(in_channels=40, out_channels=20, heads=1)
        self.conv16 = TransformerConv(in_channels=40, out_channels=20, heads=1)
        self.conv17 = TransformerConv(in_channels=40, out_channels=20, heads=1)

    def reset_parameters(self):
        for conv in self.convs1:
            conv.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        self.a.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, same_index,diff_index, edge_index, same_index2, diff_index2):
        x = features

        # Graph transformer and information aggregation layers.
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[0](x, same_index2)
        x2 = self.convs2[0](x, diff_index2)
        x1 = self.bns[0](x1)
        x2 = self.bns2[0](x2)
        x1 = F.leaky_relu(x1, inplace=True)
        x2 = F.leaky_relu(x2, inplace=True)

        x = torch.cat((x1, x2), dim=-1)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x = self.conv13(x, edge_index)
        x = self.bns3[0](x)
        x = F.leaky_relu(x, inplace=True)
        fc = x

        x3 = torch.cat((x1, x2), dim=-1)
        x4 = torch.cat((x2, x1), dim=-1)
        x1 = F.dropout(x3, p=opt.dropout, training=self.training)
        x2 = F.dropout(x4, p=opt.dropout, training=self.training)
        x1 = self.convs1[1](x1, same_index2)
        x2 = self.convs2[1](x2, diff_index2)
        x1 = self.bns[1](x1)
        x2 = self.bns2[1](x2)
        x1 = F.leaky_relu(x1, inplace=True)
        x2 = F.leaky_relu(x2, inplace=True)
        x = torch.cat((x1, x2), dim=-1)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x = self.conv14(x, edge_index)
        x = self.bns3[1](x)
        x = F.leaky_relu(x, inplace=True)

        fc = torch.cat((fc, x), dim=-1)

        x3 = torch.cat((x1, x2), dim=-1)
        x4 = torch.cat((x2, x1), dim=-1)
        x1 = F.dropout(x3, p=opt.dropout, training=self.training)
        x2 = F.dropout(x4, p=opt.dropout, training=self.training)
        x1 = self.convs1[2](x1, same_index2)
        x2 = self.convs2[2](x2, diff_index2)
        x1 = self.bns[2](x1)
        x2 = self.bns2[2](x2)
        x1 = F.leaky_relu(x1, inplace=True)
        x2 = F.leaky_relu(x2, inplace=True)
        x = torch.cat((x1, x2), dim=-1)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x = self.conv15(x, edge_index)
        x = self.bns3[2](x)
        x = F.leaky_relu(x, inplace=True)

        fc = torch.cat((fc, x), dim=-1)

        x3 = torch.cat((x1, x2), dim=-1)
        x4 = torch.cat((x2, x1), dim=-1)
        x1 = F.dropout(x3, p=opt.dropout, training=self.training)
        x2 = F.dropout(x4, p=opt.dropout, training=self.training)
        x1 = self.convs1[3](x1, same_index2)
        x2 = self.convs2[3](x2, diff_index2)
        x1 = self.bns[3](x1)
        x2 = self.bns2[3](x2)
        x1 = F.leaky_relu(x1, inplace=True)
        x2 = F.leaky_relu(x2, inplace=True)
        x = torch.cat((x1, x2), dim=-1)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x = self.conv16(x, edge_index)
        x = self.bns3[3](x)
        x = F.leaky_relu(x, inplace=True)

        fc = torch.cat((fc, x), dim=-1)

        x3 = torch.cat((x1, x2), dim=-1)
        x4 = torch.cat((x2, x1), dim=-1)
        x1 = F.dropout(x3, p=opt.dropout, training=self.training)
        x2 = F.dropout(x4, p=opt.dropout, training=self.training)
        x1 = self.convs1[4](x1, same_index2)
        x2 = self.convs2[4](x2, diff_index2)
        x1 = self.bns[4](x1)
        x2 = self.bns2[4](x2)
        x1 = F.leaky_relu(x1, inplace=True)
        x2 = F.leaky_relu(x2, inplace=True)
        x = torch.cat((x1, x2), dim=-1)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x = self.conv17(x, edge_index)
        x = self.bns3[4](x)
        x = F.leaky_relu(x, inplace=True)

        fc = torch.cat((fc, x), dim=-1)

        x = self.out_fc(fc)

        return x

class fc_hgnn(torch.nn.Module):

    def __init__(self,nonimg, phonetic_score):
        super(fc_hgnn, self).__init__()
        self.nonimg = nonimg
        self.phonetic_score = phonetic_score
        self._setup()

    def _setup(self):
        self.individual_graph_model = Brain_connectomic_graph()
        self.population_graph_model = HPG()

    def forward(self, graphs, time_series, dynamic_fc):
        dl = dataloader()
        embeddings = []

        # Brain connectomic graph
        for graph, time_serie, dyna_fc in zip(graphs, time_series, dynamic_fc):
            embedding= self.individual_graph_model(graph, time_serie, dyna_fc)
            embeddings.append(embedding)
        embeddings = torch.cat(tuple(embeddings))

        # Heterogeneous population graph (HPG)
        same_index, diff_index, edge_index, same_index2, diff_index2 = dl.get_inputs(self.nonimg, embeddings, self.phonetic_score, graphs)
        same_index = torch.tensor(same_index, dtype=torch.long).to(opt.device)
        diff_index = torch.tensor(diff_index, dtype=torch.long).to(opt.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        same_index2 = torch.tensor(same_index2, dtype=torch.long).to(opt.device)
        diff_index2 = torch.tensor(diff_index2, dtype=torch.long).to(opt.device)

        predictions = self.population_graph_model(embeddings, same_index, diff_index, edge_index, same_index2, diff_index2)

        return predictions

class Graph_Transformer(nn.Module):
    def __init__(self, input_dim, output_num,head_num, hidden_dim):
        super(Graph_Transformer, self).__init__()
        #  multi-head self-attention
        self.graph_conv = TransformerConv(input_dim, output_num, head_num)
        self.lin_out = nn.Linear(input_dim, input_dim)

        # feed forward network
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        #  multi-head self-attention
        out1 = self.lin_out(self.graph_conv(x, edge_index))

        # feed forward network
        out2 = self.ln1(out1 + x)
        out3 = self.lin2(self.act(self.lin1(out2)))
        out4 = self.ln2(out3 + out2)

        return out4

class ModelSTAGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5, cls_token='sum', readout='sero', garo_upscale=1.0):
        # input_dim: node, hidden_dim: dimension of hidden layer, num_classes: class number
        # num_heads: head number of Transformer
        # num_layers: number of GIN layers
        # sparsity: threshold of adjacency matrix sparsity
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        self.percentile = Percentile()
        self.initial_linear = nn.Linear(111, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            # self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(ModuleTransformer(111, 2*hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    def _collate_adjacency(self, a, sparsity, sparse=True):
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse_coo_tensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))

    def forward(self, a):
        # a: dynamic FCN
        # Size of a: batch x window x node x node
        logit = 0.0
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]
        h = a
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)
        adj = self._collate_adjacency(a, self.sparsity)
        for layer, (G, R, T, L) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h, adj)
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h_bridge, node_axis=2)
            if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])
            h_attend, time_attn = T(h_readout)

            # latent = self.cls_token(h_attend)
            logit = torch.squeeze(h_attend, 1).T
            # logit = torch.squeeze(h_attend, 1).view(1, -1)

        return logit, torch.squeeze(h_bridge, 1)

class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)

class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(3), x_graphattention.permute(1,0,2)

class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)

class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input)
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input

class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v
        v_combine = self.mlp(v_aggregate)
        return v_combine

class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend)
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix
