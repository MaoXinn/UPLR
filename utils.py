import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing
import json
from UBEA import *
from load_lm import *

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head,r+1,tail))
    return entity,rel,triples

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel):
        ent_size = max(entity)+1
        rel_size = (max(rel) + 1)
        print(ent_size,rel_size)
        adj_matrix = sp.lil_matrix((ent_size,ent_size))
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []
        rel_in = np.zeros((ent_size,rel_size))
        rel_out = np.zeros((ent_size,rel_size))
        
        for i in range(max(entity)+1):
            adj_features[i,i] = 1

        for h,r,t in triples:        
            adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
            adj_features[h,t] = 1; adj_features[t,h] = 1;
            radj.append([h,t,r]); radj.append([t,h,r+rel_size]); 
            rel_out[h][r] += 1; rel_in[t][r] += 1
            
        count = -1
        s = set()
        d = {}
        r_index,r_val = [],[]
        for h,t,r in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))
                r_index.append([count,r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]
        
        rel_features = np.concatenate([rel_in,rel_out],axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))    
        return adj_matrix,r_index,r_val,adj_features,rel_features      
    
def load_data(lang,train_ratio = 0.3):             
    entity1,rel1,triples1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2')

    alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    np.random.shuffle(alignment_pair)
    train_pair,dev_pair = alignment_pair[0:int(len(alignment_pair)*train_ratio)],alignment_pair[int(len(alignment_pair)*train_ratio):]
    #SI = get_word_embedding(lang)
    SI = load_lm(lang)
    entity_s = [e1 for e1, e2 in alignment_pair]
    entity_t = [e2 for e1, e2 in alignment_pair]
#    number = len(alignment_pair)
    psedu_label = UBEA(entity_s,entity_t,SI)
#    eval_entity = eval_entity_alignment(entity_s,entity_t,SI,number)
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))
#    psedu_label = train_pair
    return np.array(alignment_pair),np.array(psedu_label),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features


def get_word_embedding(lang):
    print('adding the primal input layer...')
    with open(file= lang +  'initial-node.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    return embedding_list

#load_data("data/zh_en/")
