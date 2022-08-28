import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import numpy as np
import torch
from tqdm import tqdm
import math, random
#from sklearn.cluster import KMeans, kmeans_plusplus, MeanShift, estimate_bandwidth


def tensor_kmeans_sklearn(data_vecs, n_clusters=7, metric='euclidean', need_layer_masks=False, max_iters=20):
    N,C,H,W = data_vecs.shape
    assert N == 1, 'only support singe image tensor'
    ## (1,C,H,W) -> (HW,C)
    data_vecs = data_vecs.permute(0,2,3,1).view(-1,C)
    ## convert tensor to array
    data_vecs_np = data_vecs.squeeze().detach().to("cpu").numpy()
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300)
    pred = km.fit_predict(data_vecs_np)
    cluster_ids_x = torch.from_numpy(km.labels_).to(data_vecs.device)
    id_maps = cluster_ids_x.reshape(1,1,H,W).long()
    if need_layer_masks:
        one_hot_labels = F.one_hot(id_maps.squeeze(1), num_classes=n_clusters).float()
        cluster_mask = one_hot_labels.permute(0,3,1,2)
        return cluster_mask
    return id_maps


def tensor_kmeans_pytorch(data_vecs, n_clusters=7, metric='euclidean', need_layer_masks=False, max_iters=20):
    N,C,H,W = data_vecs.shape
    assert N == 1, 'only support singe image tensor'

    ## (1,C,H,W) -> (HW,C)
    data_vecs = data_vecs.permute(0,2,3,1).view(-1,C)
    ## cosine | euclidean
    #cluster_ids_x, cluster_centers = kmeans(X=data_vecs, num_clusters=n_clusters, distance=metric, device=data_vecs.device)
    cluster_ids_x, cluster_centers = kmeans(X=data_vecs, num_clusters=n_clusters, distance=metric,\
                                                    tqdm_flag=False, iter_limit=max_iters, device=data_vecs.device)
    id_maps = cluster_ids_x.reshape(1,1,H,W)
    if need_layer_masks:
        one_hot_labels = F.one_hot(id_maps.squeeze(1), num_classes=n_clusters).float()
        cluster_mask = one_hot_labels.permute(0,3,1,2)
        return cluster_mask
    return id_maps


def batch_kmeans_pytorch(data_vecs, n_clusters=7, metric='euclidean', use_sklearn_kmeans=False):
    N,C,H,W = data_vecs.shape
    sample_list = []
    for idx in range(N):
        if use_sklearn_kmeans:
            cluster_mask = tensor_kmeans_sklearn(data_vecs[idx:idx+1,:,:,:], n_clusters, metric, True)
        else:
            cluster_mask = tensor_kmeans_pytorch(data_vecs[idx:idx+1,:,:,:], n_clusters, metric, True)
        sample_list.append(cluster_mask)
    return torch.cat(sample_list, dim=0)


def get_centroid_candidates(data_vecs, n_clusters=7, metric='euclidean', max_iters=20):
    N,C,H,W = data_vecs.shape
    data_vecs = data_vecs.permute(0,2,3,1).view(-1,C)
    cluster_ids_x, cluster_centers = kmeans(X=data_vecs, num_clusters=n_clusters, distance=metric,\
                                                    tqdm_flag=False, iter_limit=max_iters, device=data_vecs.device)
    return cluster_centers


def find_distinctive_elements(data_tensor, n_clusters=7, topk=3, metric='euclidean'):
    N,C,H,W = data_tensor.shape
    centroid_list = []
    for idx in range(N):
        cluster_centers = get_centroid_candidates(data_tensor[idx:idx+1,:,:,:], n_clusters, metric)
        centroid_list.append(cluster_centers)
    
    batch_centroids = torch.stack(centroid_list, dim=0)
    data_vecs = data_tensor.flatten(2)
    ## distance matrix: (N,K,HW) = (N,K,C) x (N,C,HW)
    AtB = torch.matmul(batch_centroids, data_vecs)
    AtA = torch.matmul(batch_centroids, batch_centroids.permute(0,2,1))
    BtB = torch.matmul(data_vecs.permute(0,2,1), data_vecs)
    diag_A = torch.diagonal(AtA, dim1=-2, dim2=-1)
    diag_B = torch.diagonal(BtB, dim1=-2, dim2=-1)
    A2 = diag_A.unsqueeze(2).repeat(1,1,H*W)
    B2 = diag_B.unsqueeze(1).repeat(1,n_clusters,1)
    distance_map = A2 - 2*AtB + B2
    values, indices = distance_map.topk(topk, dim=2, largest=False, sorted=True)
    cluster_mask = torch.where(distance_map <= values[:,:,topk-1:], torch.ones_like(distance_map), torch.zeros_like(distance_map))
    cluster_mask = cluster_mask.view(N,n_clusters,H,W)
    return cluster_mask


##---------------------------------------------------------------------------------
'''
    resource from github: https://github.com/subhadarship/kmeans_pytorch
'''
##---------------------------------------------------------------------------------

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=True,
        iter_limit=0,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            #print('hello, there!')
            break

    return choice_cluster.to(device), initial_state.to(device)


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        tqdm_flag=True
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    if tqdm_flag:
        print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    elif distance == 'soft_dtw':
        sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
        pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu'), tqdm_flag=True):
    if tqdm_flag:
        print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis