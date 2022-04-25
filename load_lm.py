# _*_ coding:utf-8 _*_

from tqdm import tqdm
import json
import numpy as np


def load_lm(ent_names1):
    word_vecs = {}
    with open("./glove.6B.300d.txt",encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])
    ent_names = json.load(open(ent_names1+"name.json","r"))
    file_path = ent_names1
    all_triples,node_size,rel_size = load_triples(file_path,True)
    d = {}
    count = 0
    for _,name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word)-1):
                if word[idx:idx+2] not in d:
                    d[word[idx:idx+2]] = count
                    count += 1

    ent_vec = np.zeros((node_size,300))

    for i,name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
        if k:
            ent_vec[i]/=k
        else:
            ent_vec[i] = np.random.random(300)-0.5
        ent_vec[i] = ent_vec[i]/ np.linalg.norm(ent_vec[i])
    return ent_vec


def load_triples(file_path,reverse = True):
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            if reverse:
                reversed_triples[i,1] = triples[i,1] + rel_size
            else:
                reversed_triples[i,1] = triples[i,1]
        return reversed_triples
    
    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()
    
    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()
    
    triples = np.array([line.replace("\n","").split("\t") for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:,0]),np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:,1]) + 1
    
    all_triples = np.concatenate([triples,reverse_triples(triples)],axis=0)
    all_triples = np.unique(all_triples,axis=0)
    
    return all_triples, node_size, rel_size*2 if reverse else rel_size

