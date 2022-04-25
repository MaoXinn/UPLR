from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np

class Gate_GraphAttention(Layer):

    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 depth = 1,
                 attn_heads=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.attn_heads = attn_heads
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.depth = depth
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False
        self.biases = []
        self.attn_kernels = []  
        self.gat_kernels = []
        self.gate_kernels = []
        super(Gate_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_F = input_shape[0][-1]
        rel_F = input_shape[1][-1]
        self.ent_F = node_F
        ent_F = self.ent_F
        self.gate_kernel = self.add_weight(shape=(ent_F*(self.depth+1),ent_F*(self.depth+1)),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name='gate_kernel')

        self.GAT = self.add_weight(shape=(64,node_F*(self.depth+1)),
                                   initializer=self.attn_kernel_initializer,
                                   regularizer=self.attn_kernel_regularizer,
                                   constraint=self.attn_kernel_constraint,
                                   name='GAT')

        self.bias = self.add_weight(shape=(1,ent_F*(self.depth+1)),
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint,
                                   name='bias')
            
        for l in range(self.depth):
            self.attn_kernels.append([])
            for head in range(self.attn_heads):                
                attn_kernel = self.add_weight(shape=(1*node_F ,1),
                                       initializer=self.attn_kernel_initializer,
                                       regularizer=self.attn_kernel_regularizer,
                                       constraint=self.attn_kernel_constraint,
                                       name='attn_kernel_self_{}'.format(head))
                self.attn_kernels[l].append(attn_kernel)
    
    def call(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]     
        adj = tf.SparseTensor(K.cast(K.squeeze(inputs[2],axis = 0),dtype = "int64"),
                         K.ones_like(inputs[2][0,:,0]),(self.node_size,self.node_size))
        sparse_indices = tf.squeeze(inputs[3],axis = 0)  
        sparse_val = tf.squeeze(inputs[4],axis = 0)
        features = self.activation(features)
        outputs.append(features)
        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]  
                rels_sum = tf.SparseTensor(indices=sparse_indices,values=sparse_val,dense_shape=(self.triple_size,self.rel_size))
                rels_sum = tf.sparse_tensor_dense_matmul(rels_sum,rel_emb)
                neighs = K.gather(features,adj.indices[:,1])
                selfs = K.gather(features,adj.indices[:,0])
                rels_sum = tf.nn.l2_normalize(rels_sum, 1)
                neighs = neighs - 2 * tf.reduce_sum(neighs * rels_sum, 1, keepdims=True) * rels_sum
                att = K.squeeze(K.dot(rels_sum,attention_kernel),axis = -1)
                att = tf.SparseTensor(indices=adj.indices, values=att, dense_shape=adj.dense_shape)
                att = tf.sparse_softmax(att)
                new_features = tf.segment_sum (neighs*K.expand_dims(att.values,axis = -1),adj.indices[:,0])
                features_list.append(new_features)
            features = K.concatenate(features_list)
            features = self.activation(features)
            outputs.append(features)
        outputs = K.concatenate(outputs)
        GAT_att = K.dot(tf.nn.l2_normalize(outputs,axis=-1),K.transpose(tf.nn.l2_normalize(self.GAT,axis=-1)))
        GAT_att = K.softmax(GAT_att,axis = -1)
        GAT_feature = outputs - K.dot(GAT_att,self.GAT)
        gate_rate = K.sigmoid(K.dot(GAT_feature,self.gate_kernel) + self.bias)
        outputs = (1-gate_rate) * GAT_feature + (gate_rate) * outputs
        return outputs


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


