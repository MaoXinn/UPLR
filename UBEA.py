
import multiprocessing

import gc
import os

import numpy as np
import time

from scipy.spatial.distance import cdist



def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def cal_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values


def Neighbor(sim_mat1):
    
    tasks = div_list(np.array(range(sim_mat1.shape[0])), 10)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_sim, (sim_mat1[task, :], 1)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values


def rank_sim_mat(task, sim):
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        rank = (sim[i, :]).argsort()
        prec_set.add((ref, rank[0]))
    return prec_set


def GID_Pseudo_label(embed1, embed2):
    hy = 0.5
    sim_mat_X = np.matmul(embed1, embed2.T)
    sim_mat_Y = np.matmul(embed2, embed1.T)
    es = Neighbor(sim_mat_X)
    et = Neighbor(sim_mat_Y)
    es1= np.array(es)
    es2 = es1.reshape(-1,1)
    div = hy * (es2 + et)
    sim_mat = -(sim_mat_X - div)
    ref_num = sim_mat.shape[0]
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), 10)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(rank_sim_mat, (task, sim_mat[task, :])))
    pool.close()
    pool.join()

    for res in reses:
        prec_set = res.get()
        t_prec_set |= prec_set
    return t_prec_set



def GID(embed1, embed2):
    sim_mat_X = np.matmul(embed1, embed2.T)
    sim_mat_Y = np.matmul(embed2, embed1.T)
    es = Neighbor(sim_mat_X)
    et = Neighbor(sim_mat_Y)
    es1= np.array(es)
    es2 = es1.reshape(-1,1)
    div = 0.5 * (es2 + et)
    sim = (sim_mat_X - div)
    return sim


def UBEA(entity_s,entity_t,SI):
    new_pair = []
    np.random.shuffle(entity_s)
    np.random.shuffle(entity_t)
    Lvec = np.array([SI[e] for e in entity_s])
    Rvec = np.array([SI[e] for e in entity_t])
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    A = GID_Pseudo_label(Lvec, Rvec)
    B = GID_Pseudo_label(Rvec, Lvec)
    A = sorted(list(A))
    B = sorted(list(B))
    for es,et in A:
        if  B[et][1] == es:
            new_pair.append([entity_s[es],entity_t[et]])
    print("generate new semi-pairs: %d." % len(new_pair))
    train_pair = np.array(new_pair)
    for e1,e2 in train_pair:
        if e1 in entity_s:
            entity_s.remove(e1)

    for e1,e2 in train_pair:
        if e2 in entity_t:
            entity_t.remove(e2)
    return train_pair


def eval_entity_alignment(entity_s,entity_t,SI,number):
    new_pair = []
    Lvec = np.array([SI[e] for e in entity_s])
    Rvec = np.array([SI[e] for e in entity_t])
    Lvec1 = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec1 = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    sim = -GID(Lvec1,Rvec1)
    top_k=(1, 10)
    mr = 0
    mrr = 0
    top_lr = [0] * len(top_k)
    for i in range(Lvec1.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    mr /=  number
    mrr /= number
    print('mr =%f'% mr)
    print('mrr =%f '% mrr)
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / number * 100))
















