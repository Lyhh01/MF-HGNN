import torch
from sklearn.model_selection import StratifiedKFold
from graph import get_node_feature
import csv
import opt

from torch import nn
import torch.nn.functional as F
import sys
from opt import *
import os
import scipy.io as sio
from sklearn.decomposition import PCA

opt = OptInit().initialize()


def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std = dataset.std()

    return ((dataset - mean) / std).astype(dtype)


def intensityNormalisationFeatureScaling(dataset, dtype):
    max = dataset.max()
    min = dataset.min()

    return ((dataset - min) / (max - min)).astype(dtype)


class dataloader():
    def __init__(self):
        self.pd_dict = {}
        self.num_classes = opt.num_classes

    def load_data(self):

        subject_IDs = get_ids()
        # Read data, including phenotypic data and labels.
        # It is recommended to adjust the group in the phe file to 0 and 1.
        labels = get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)
        sites = get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        dsms = get_subject_score(subject_IDs,
                                 score='DSM_IV_TR')  # This indicator contains label information and should not be used.
        fiq = get_subject_score(subject_IDs, score='FIQ')
        genders = get_subject_score(subject_IDs, score='SEX')

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])

        age = np.zeros([num_nodes], dtype=np.float32)
        dsm = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int)
        site = np.zeros([num_nodes], dtype=np.int)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]]) - 1] = 1
            y[i] = int(labels[subject_IDs[i]]) - 1
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            dsm[i] = float(dsms[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]

        # Get the labels and features of the subjects.
        self.y = y
        self.raw_features = get_node_feature()
        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:, 0] = site
        phonetic_data[:, 1] = gender
        phonetic_data[:, 2] = age
        self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['SEX'] = np.copy(phonetic_data[:, 1])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:, 2])

        phonetic_score = self.pd_dict

        time_series = []
        for subject in subject_IDs:
            fl = os.path.join(opt.data_path, subject,
                          subject + "_" + "ho" + "_" + "time_series" + ".mat")
            matrix = sio.loadmat(fl)["time_series"]
            time_series.append(matrix)

        dynamic_fc = []
        for subject in subject_IDs:
            fl = os.path.join(opt.data_path, subject,
                          subject + "_" + "ho" + "_" + "dynamic_fc" + ".mat")
            matrix = sio.loadmat(fl)["dynamic_fc"]
            dynamic_fc.append(matrix)
        return self.raw_features, self.y, phonetic_data, phonetic_score, time_series, dynamic_fc

    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=666)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_inputs(self, nonimg, embeddings, phonetic_score, graphs):
        # Compute the edges of the HPG.

        # for graph in graphs:
        #     feature = graph.x
        #     for i in range(feature.shape[0]):
        #         for j in range(i+1, feature.shape[1]):
        peoples_feature = [item.x for item in graphs]
        peoples_feature = [item.reshape(1, -1) for item in peoples_feature]
        peoples_feature = torch.stack(peoples_feature)

        S = self.create_type_mask(embeddings, phonetic_score)  # Gender mask matrix
        S2 = self.create_age_similarity_mask()  # Age mask matrix
        self.node_ftr = np.array(embeddings.detach().cpu().numpy())
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # pd_attn_affinity = get_multi_atten(phonetic_score)

        aff_adj = get_static_affinity_adj(phonetic_score, peoples_feature)  # The Adjacency matrix of the HPG.
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1
        assert flatten_ind == num_edge, "Error in computing edge input"

        # Set the threshold, which is beta in the paper.
        keep_ind = np.where(aff_score > opt.beta)[0]
        edge_index = edge_index[:, keep_ind]
        same_row = []
        same_col = []
        diff_row = []
        diff_col = []

        same_row2 = []
        same_col2 = []
        diff_row2 = []
        diff_col2 = []
        for i in range(edge_index.shape[1]):
            if S[edge_index[0, i], edge_index[1, i]] == 1:
                same_row.append(edge_index[0, i])
                same_col.append(edge_index[1, i])
            else:
                diff_row.append(edge_index[0, i])
                diff_col.append(edge_index[1, i])

            if S2[edge_index[0, i], edge_index[1, i]] == 1:
                same_row2.append(edge_index[0, i])
                same_col2.append(edge_index[1, i])
            else:
                diff_row2.append(edge_index[0, i])
                diff_col2.append(edge_index[1, i])

        same_index = np.stack((same_row, same_col)).astype(np.int64)
        diff_index = np.stack((diff_row, diff_col)).astype(np.int64)

        same_index2 = np.stack((same_row2, same_col2)).astype(np.int64)
        diff_index2 = np.stack((diff_row2, diff_col2)).astype(np.int64)
        return same_index, diff_index, edge_index, same_index2, diff_index2

    def create_type_mask(self, embeddings, phonetic_score):

        # subject_IDs = get_ids()
        # site_id = phonetic_score['SITE_ID']
        # unique_site_id = np.unique(list(site_id)).tolist()
        # # site_embeddings_torch = torch.zeros((len(unique_site_id), embeddings.shape[1]), dtype=torch.float32)
        # site_embeddings_torch = None
        # for i in unique_site_id:
        #     mean_embedding = embeddings[site_id == i].mean(dim=0).unsqueeze(0)
        #     if site_embeddings_torch is None:
        #         site_embeddings_torch = mean_embedding
        #     else:
        #         site_embeddings_torch = torch.cat((site_embeddings_torch, mean_embedding), dim=0)
        #
        # print('site_embeddings_torch:',site_embeddings_torch,' site_embeddings_torch.shape', site_embeddings_torch.shape)

        subject_list = get_ids()
        num_nodes = len(subject_list)
        type_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        type = get_subject_score(subject_list, score='SEX')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if type[subject_list[i]] == type[subject_list[j]]:
                    type_matrix[i, j] = 1
                    type_matrix[j, i] = 1

        type_matrix = torch.from_numpy(type_matrix)
        device = 'cuda:0'
        return type_matrix.to(device)

    def create_age_similarity_mask(self):
        subject_list = get_ids()
        num_nodes = len(subject_list)
        type_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        type = get_subject_score(subject_list, score='SITE_ID')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if type[subject_list[i]] == type[subject_list[j]]:
                    type_matrix[i, j] = 1
                    type_matrix[j, i] = 1

        type_matrix = torch.from_numpy(type_matrix)
        device = 'cuda:0'
        return type_matrix.to(device)


def get_subject_score(subject_list, score):
    scores_dict = {}
    phenotype = opt.phenotype_path
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    return scores_dict


def get_ids(num_subjects=None):
    subject_IDs = np.genfromtxt(opt.subject_IDs_path, dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph


def get_static_affinity_adj(pd_dict, peoples_feature):
    # Phenotypic data similarity scores of HPG based on several phenotypic data were calculated.
    pd_affinity = create_affinity_graph_from_scores(['SEX', 'SITE_ID'], pd_dict)
    # pd_affinity = pd_affinity + pd_affinity2

    # pd_affinity = create_affinity_graph_from_scores(['AGE_AT_SCAN', 'SEX', 'EDU'], pd_dict)
    pd_affinity = (pd_affinity - pd_affinity.mean(axis=0)) / pd_affinity.std(axis=0)
    return pd_affinity


def get_multi_atten(pd_dict):
    conv1 = nn.Sequential(nn.Linear(871, 20, bias=False)).to(opt.device)
    conv2 = nn.Sequential(nn.Linear(871, 20, bias=False)).to(opt.device)
    conv3 = nn.Sequential(nn.Linear(871, 20, bias=False)).to(opt.device)
    conv4 = nn.Sequential(nn.Linear(871 * 20, 100, bias=False)).to(opt.device)
    conv5 = nn.Sequential(nn.Linear(871 * 20, 100, bias=False)).to(opt.device)
    conv6 = nn.Sequential(nn.Linear(871 * 20, 100, bias=False)).to(opt.device)
    conv7 = nn.Sequential(nn.Linear(100, 20, bias=False)).to(opt.device)
    conv8 = nn.Sequential(nn.Linear(100, 20, bias=False)).to(opt.device)
    conv9 = nn.Sequential(nn.Linear(100, 20, bias=False)).to(opt.device)
    conv10 = nn.Sequential(nn.Linear(20, 1, bias=False)).to(opt.device)

    pd_affinity_sex = torch.from_numpy(create_affinity_graph_from_scores(['SEX'], pd_dict)).float().to(opt.device)
    pd_affinity_site = torch.from_numpy(create_affinity_graph_from_scores(['SITE_ID'], pd_dict)).float().to(opt.device)
    pd_affinity_age = torch.from_numpy(create_affinity_graph_from_scores(['AGE_AT_SCAN'], pd_dict)).float().to(
        opt.device)  # (871,871)

    pd_affinity_sex_mapping = torch.nn.functional.leaky_relu(conv1(pd_affinity_sex))
    pd_affinity_site_mapping = torch.nn.functional.leaky_relu(conv2(pd_affinity_site))
    pd_affinity_age_mapping = torch.nn.functional.leaky_relu(conv3(pd_affinity_age))  # (871,20)

    pd_affinity_sex2 = pd_affinity_sex_mapping.view(1, -1)
    pd_affinity_site2 = pd_affinity_site_mapping.view(1, -1)
    pd_affinity_age2 = pd_affinity_age_mapping.view(1, -1)  # (1,871*20)

    pd_affinity_sex_mapping2 = torch.nn.functional.leaky_relu(conv4(pd_affinity_sex2))
    pd_affinity_site_mapping2 = torch.nn.functional.leaky_relu(conv5(pd_affinity_site2))
    pd_affinity_age_mapping2 = torch.nn.functional.leaky_relu(conv6(pd_affinity_age2))  # (1,100)

    edge_info = torch.cat((pd_affinity_sex_mapping2, pd_affinity_site_mapping2, pd_affinity_age_mapping2), 0)  # (3,100)

    edge_info_k = torch.nn.functional.leaky_relu(conv7(edge_info))
    edge_info_q = torch.nn.functional.leaky_relu(conv8(edge_info))
    edge_info_v = torch.nn.functional.leaky_relu(conv9(edge_info))  # (3,20)

    k = edge_info_k
    q = edge_info_q.T
    attn = torch.matmul(k, q)
    attn = attn / abs(attn.min())  # 3 * 3
    attn = F.normalize(attn, dim=-1)
    attn = F.softmax(attn, dim=-1)
    node_features_1 = torch.matmul(attn, edge_info_v) + edge_info_v  # (3,20)

    mul_attn = F.softmax(torch.nn.functional.leaky_relu(conv10(node_features_1)), 1)  # (3,1)

    attn_sex = mul_attn[0] * pd_affinity_sex
    attn_site = mul_attn[1] * pd_affinity_site
    attn_age = mul_attn[2] * pd_affinity_age

    pd_attn_affinity = attn_sex + attn_site + attn_age
    # pd_affinity = create_affinity_graph_from_scores(['AGE_AT_SCAN', 'SEX', 'EDU'], pd_dict)
    pd_attn_affinity = (pd_attn_affinity - pd_attn_affinity.mean(axis=0)) / pd_attn_affinity.std(axis=0)
    return pd_attn_affinity.float()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, output, target):
        n_classes = output.size(1)
        target_one_hot = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
        target_smooth = target_one_hot * (1 - self.smoothing) + (1 - target_one_hot) * self.smoothing / (n_classes - 1)
        log_probs = nn.functional.log_softmax(output, dim=1)
        loss = nn.functional.kl_div(log_probs, target_smooth, reduction='none').sum(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass