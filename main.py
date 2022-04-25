import warnings
warnings.filterwarnings('ignore')

import os
import keras
import numpy as np
import numba as nb
from utils import *
from tqdm import *
from UBEA import*
#from evaluate import evaluate
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from layer import Gate_GraphAttention

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

seed = 12306
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

alignment_pair,train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features = load_data("data/fr_en/",train_ratio=0.30)
adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 125
rel_hidden = 125
batch_size = 1024
dropout_rate = 0.3
lr = 0.005
gamma = 5.5
depth = 3



class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""
    
    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def call(self, inputs):
        return self.embeddings

def train_model(node_hidden,rel_hidden,triple_size=triple_size,node_size=node_size,rel_size=rel_size,dropout_rate = 0,gamma = 3,lr = 0.005,depth = 2):
    adj_input = Input(shape=(None,2))
    index_input = Input(shape=(None,2),dtype='int64')
    val_input = Input(shape = (None,))
    rel_adj = Input(shape=(None,2))
    ent_adj = Input(shape=(None,2))
    
    ent_emb = TokenEmbedding(node_size,node_hidden,trainable = True)(val_input)
    rel_emb = TokenEmbedding(rel_size,node_hidden,trainable = True)(val_input)
    
    def avg(tensor,size):
        adj = K.cast(K.squeeze(tensor[0],axis = 0),dtype = "int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:,0],dtype = 'float32'), dense_shape=(node_size,size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj,tensor[1])
    
    opt = [rel_emb,adj_input,index_input,val_input]
    ent_feature = Lambda(avg,arguments={'size':node_size})([ent_adj,ent_emb])
    rel_feature = Lambda(avg,arguments={'size':rel_size})([rel_adj,rel_emb])
    
    e_encoder = Gate_GraphAttention(node_size,activation="tanh",
                                  rel_size = rel_size,
                                  use_bias = True,
                                  depth = depth,
                                  triple_size = triple_size)

                                  
    out_feature = Concatenate(-1)([e_encoder([ent_feature]+opt),e_encoder([rel_feature]+opt)])
    out_feature = Dropout(dropout_rate)(out_feature)
                                  
    alignment_input = Input(shape=(None,2))
                                  
    def align_loss(tensor):
        
        def squared_dist(x):
            A,B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
            return row_norms_A + row_norms_B - 2 * tf.matmul(A, B,transpose_b=True)
        def Gradient(loss):
            gradient = (loss - K.stop_gradient(K.mean(loss,axis=-1,keepdims=True))) / K.stop_gradient(K.std(loss,axis=-1,keepdims=True))
            losss = K.logsumexp(30*loss+10,axis=-1)
            return losss
        def Matrix(loss,l,r):
            matrix = loss *(1 - K.one_hot(indices=l,num_classes=node_size) - K.one_hot(indices=r,num_classes=node_size))
            return Gradient(matrix)
        
        
        emb = tensor[1]
        ps,pt = K.cast(tensor[0][0,:,0],'int32'),K.cast(tensor[0][0,:,1],'int32')
        ps_emb,pt_emb = K.gather(reference=emb,indices=ps),K.gather(reference=emb,indices=pt)
                                                                              
        Lr = K.sum(K.square(ps_emb-pt_emb),axis=-1,keepdims=True)
        Ls = squared_dist([ps_emb,emb])
        Lt = squared_dist([pt_emb,emb])
        LN1 = Lr - Ls + gamma
        LN2 = Lr - Lt + gamma
        
        l_loss = Matrix(LN1,ps,pt)
        r_loss = Matrix(LN2,ps,pt)
        return K.mean(l_loss + r_loss)
                                                                                                                          
    loss = Lambda(align_loss)([alignment_input,out_feature])
    
    inputs = [adj_input,index_input,val_input,rel_adj,ent_adj]
    train_model = keras.Model(inputs = inputs + [alignment_input],outputs = loss)
    train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=keras.optimizers.rmsprop(lr))
    
    feature_model = keras.Model(inputs = inputs,outputs = out_feature)
    
    return train_model,feature_model

model,get_emb = train_model(dropout_rate=dropout_rate,
                          node_size=node_size,
                          rel_size=rel_size,
                          depth=depth,
                          gamma =gamma,
                          node_hidden=node_hidden,
                          rel_hidden=rel_hidden,
                          lr=lr)

#evaluater = evaluate(dev_pair)

model.summary()

def get_embedding(index_a,index_b):

    inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix]
    inputs = [np.expand_dims(item,axis=0) for item in inputs]
    vec = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec,axis=-1,keepdims=True)+1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec,axis=-1,keepdims=True)+1e-5)
    return Lvec,Rvec,vec

dev_s = [e1 for e1, e2 in dev_pair]
dev_t = [e2 for e1, e2 in dev_pair]
#dev_s = dev_pair[:,0]
#dev_t = dev_pair[:,1]

np.random.shuffle(dev_s)
np.random.shuffle(dev_t)
#print('dev_s=',len(dev_s))
#print('dev_t=',len(dev_t))


#np.random.shuffle(all_s)
#np.random.shuffle(all_t)

epoch = 6
for turn in range(5):
    for i in range(epoch):
        np.random.shuffle(train_pair)
        for pairs in [train_pair[i*batch_size:(i+1)*batch_size] for i in range(len(train_pair)//batch_size + 1)]:
            inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix,pairs]
            inputs = [np.expand_dims(item,axis=0) for item in inputs]
            model.train_on_batch(inputs,np.zeros((1,1)))
#       evaluater
        if i==epoch-1:
            Lvec,Rvec,SI = get_embedding(dev_pair[:,0],dev_pair[:,1])
#            inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix]
#            inputs = [np.expand_dims(item,axis=0) for item in inputs]
#            SI = get_emb.predict_on_batch(inputs)
            number = len(dev_pair)
            eval_entity_alignment(dev_pair[:,0],dev_pair[:,1],SI,number)
#            evaluater.test(Lvec,Rvec)
    new_pair = []
    Lvec,Rvec,vec = get_embedding(dev_s,dev_t)
    inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix]
    inputs = [np.expand_dims(item,axis=0) for item in inputs]
    SI = get_emb.predict_on_batch(inputs)
    new_pair = UBEA(dev_s,dev_t,SI)
    train_pair = np.concatenate([train_pair,new_pair],axis = 0)

    for e1,e2 in new_pair:
        if e1 in dev_s:
            dev_s=remove(e1)

    for e1,e2 in new_pair:
        if e2 in dev_t:
            dev_t.remove(e2)
