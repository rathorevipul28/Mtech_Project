
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import gensim
import random
import pickle
import json
import codecs

from pprint import pprint
from collections import defaultdict


# In[2]:


config = tf.ConfigProto()
config.gpu_options.allow_growth=True


# In[3]:


embedding_dim = 300
lstm_dim = 128
gcn_dim = 256
gcn_layers = 2
paragraph_size = 174
sentence_size = 37
num_of_sentence = 10
entity_size = 9
num_class = 3
max_answer_len = 5
embedding_location = 'glove.6B.300d_word2vec.txt'
merge_edges = False
use_de_labels = True
batch_size = 32
patience = 30
max_epochs = 100
dropout = 0.9
rec_dropout = 0.9
wGate = False
regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)


# In[4]:


def updateEdges(data, merge_edges=False):
    for dtype in ['train', 'test', 'valid']:
        # Remove dependency edges with negative source/destination ids
        for i, edges in enumerate(data[dtype]['DepEdges']):
            for j in range(len(edges)-1, -1, -1):
                edge = edges[j]
                if edge[0] < 0 or edge[1] < 0:
                    del data[dtype]['DepEdges'][i][j]
    return data


# In[5]:


srl_labels = ["R-ARGM-COM", "C-ARGM-NEG", "C-ARGM-TMP", "R-ARGM-DIR", "ARGM-LOC", "R-ARG2", "ARGM-GOL", "ARG5", "ARGM-EXT", "R-ARGM-ADV", "C-ARGM-MNR", "ARGA", "C-ARG4", "C-ARG2", "C-ARG3", "C-ARG0", "C-ARG1", "ARGM-ADV", "ARGM-NEG", "R-ARGM-MNR", "C-ARGM-EXT", "R-ARGM-PRP", "C-ARGM-ADV", "R-ARGM-MOD", "C-ARGM-ADJ", "ARGM-LVB", "R-ARGM-PRD", "ARGM-MNR", "ARGM-ADJ", "C-ARGM-CAU", "ARGM-CAU", "C-ARGM-MOD", "R-ARGM-EXT", "C-ARGM-COM", "ARGM-COM", "R-ARGM-GOL", "R-ARGM-TMP", "R-ARG4", "ARGM-MOD", "R-ARG1", "R-ARG0", "R-ARG3", "V", "ARGM-REC", "C-ARGM-DSP", "R-ARG5", "ARGM-DIS", "ARGM-DIR", "R-ARGM-LOC", "C-ARGM-DIS", "ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARGM-TMP", "C-ARGM-DIR", "ARGM-PRD", "R-ARGM-PNC", "ARGM-PRX", "ARGM-PRR", "R-ARGM-CAU", "C-ARGM-LOC", "ARGM-PNC", "ARGM-PRP", "C-ARGM-PRP", "ARGM-DSP"]


# In[6]:


srl2id = {}
for i,j in enumerate(srl_labels):
    srl2id[j] = i


# In[7]:


num_srlLabel = len(srl_labels)


# In[8]:


num_srlLabel


# In[9]:


# Data loading
data = pickle.load(open('propara_processed.pkl', 'rb'))

with codecs.open('paragrap-id-mapping.json', 'r') as f:
    paragraph_id_mapping = json.load(f)

voc2id = data['voc2id']
de2id = data['de2id']

num_deLabel = len(de2id)
data = updateEdges(data, merge_edges)

# Get Word List
wrd_list = list(voc2id.items())    # Get vocabulary
wrd_list.sort(key=lambda x: x[1])       # Sort vocabulary based on ids
wrd_list, _ = zip(*wrd_list)

data_list = {}
key_list =  ['X', 'Y', 'DepEdges']

for dtype in ['train', 'test', 'valid']:
    if use_de_labels == False:
        for i, edges in enumerate(data[dtype]['DepEdges']): # if you want to ignore level information in dependency graph
            for j, edge in enumerate(edges): data[dtype]['DepEdges'][i][j] = (edge[0], edge[1], 0)
        num_deLabel = 1
    data_list[dtype] = []
    for i in range(len(data[dtype]['X'])):
        data_list[dtype].append([data[dtype][key][i] for key in key_list]) # data_list contains all the fields for train test and valid documents
    print ('Document count [{}]: {}'.format(dtype, len(data_list[dtype])))


# In[10]:


para2srl = {}
import json
with open("out1.txt" , "r") as f:
    for line in f:
        with open("para1a.json", "w") as f1:
            f1.write(line)
        f1.close()
        with open("para1a.json", "r") as f2:
            x = json.load(f2)
        f2.close()
        l=x['sentences']
        l1=[voc2id[word] for z in l for word in z]
        for i in range(len(l1),174):
            l1.append(0)
        x['sentences1'] = l1
        y=x['predicted_srl']
        srl_tup = []
        for i in range(len(y)):
            for k in range(y[i][1],y[i][2]+1):
                srl_tup.append((y[i][0],k,srl2id[y[i][3]]))
        para2srl[tuple(l1)] = srl_tup
f.close()
with open("out2.txt" , "r") as f:
    for line in f:
        with open("para1a.json", "w") as f1:
            f1.write(line)
        f1.close()
        with open("para1a.json", "r") as f2:
            x = json.load(f2)
        f2.close()
        l=x['sentences']
        l1=[voc2id[word] for z in l for word in z]
        for i in range(len(l1),174):
            l1.append(0)
        x['sentences1'] = l1
        y=x['predicted_srl']
        srl_tup = []
        for i in range(len(y)):
            for k in range(y[i][1],y[i][2]+1):
                srl_tup.append((y[i][0],k,srl2id[y[i][3]]))
        para2srl[tuple(l1)] = srl_tup
f.close()
for i in range(len(data_list['train'])):
    data_list['train'][i].append(para2srl[tuple(data_list['train'][i][0]['complete_paragraph'])])
for i in range(len(data_list['valid'])):
    data_list['valid'][i].append(para2srl[tuple(data_list['valid'][i][0]['complete_paragraph'])])
for i in range(len(data_list['test'])):
    data_list['test'][i].append(para2srl[tuple(data_list['test'][i][0]['complete_paragraph'])])


# In[11]:


for i in range(len(data_list['train'])):
    if len(data_list['train'][i]) != 4:
        print("Error")


# In[12]:


for i in range(len(data_list['test'])):
    if len(data_list['test'][i]) != 4:
        print("Error")


# In[13]:


for i in range(len(data_list['valid'])):
    if len(data_list['valid'][i]) != 4:
        print("Error")


# In[39]:


# Adding placeholder
input_x_paragraph = tf.placeholder(tf.int32, shape=[None, paragraph_size], name='input_data_paragragh')
input_x_sentence = tf.placeholder(tf.int32, shape=[None, num_of_sentence, sentence_size], name='input_data_sentence')
input_x_entity = tf.placeholder(tf.int32, shape=[None, entity_size], name='input_data_entity')
input_y_known = tf.placeholder(tf.int32, shape=[None, num_of_sentence, num_class], name='input_labels_is_known')
input_y_start = tf.placeholder(tf.int32, shape=[None, num_of_sentence], name='input_labels_start_position')
input_y_end = tf.placeholder(tf.int32, shape=[None, num_of_sentence], name='input_labels_end_position')

input_x_mask_sentence = tf.placeholder(tf.bool, shape=[None, num_of_sentence], name='input_sentence_mask')

input_x_len_paragraph = tf.placeholder(tf.int32, shape=[None], name='input_len_paragraph')
input_x_len_sentence = tf.placeholder(tf.int32, shape=[None, num_of_sentence], name='input_len_sentence')
input_x_len_entity = tf.placeholder(tf.float32, shape=[None], name='input_len_entity')

# Array of batch_size number of dictionaries, where each dictionary is mapping of label to sparse_placeholder
de_adj_mat_in = [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'de_adj_mat_in_{}'. format(lbl))) for lbl in range(num_deLabel)]) for i in range(batch_size) ]
de_adj_mat_out = [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'de_adj_mat_out_{}'.format(lbl))) for lbl in range(num_deLabel)]) for i in range(batch_size) ]

# Array of batch_size number of dictionaries, where each dictionary is mapping of label to sparse_placeholder
srl_adj_mat_in = [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'srl_adj_mat_in_{}'. format(lbl))) for lbl in range(num_srlLabel)]) for i in range(batch_size) ]
srl_adj_mat_out = [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'srl_adj_mat_out_{}'.format(lbl))) for lbl in range(num_srlLabel)]) for i in range(batch_size) ]
    
# seq_len = tf.placeholder(tf.int32, shape=(), name='seq_len') # Maximum number of words in documents of a batch

dropout = tf.placeholder_with_default(dropout, shape=(), name='dropout') # Dropout used in GCN Layer
rec_dropout = tf.placeholder_with_default(rec_dropout, shape=(), name='rec_dropout') # Dropout used in Bi-LSTM


# In[40]:


def getEmbeddings(embed_loc, wrd_list, embedding_dim):
    embed_list = []
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_loc, binary=False)

    for wrd in wrd_list:
        if wrd in model.vocab:
            embed_list.append(model.word_vec(wrd))
        else:
            embed_list.append(np.random.randn(embedding_dim))
    return np.array(embed_list, dtype=np.float32)

def GCNLayer(gcn_in,  # Input to GCN Layer
           in_dim,  # Dimension of input to GCN Layer 
           gcn_dim,  # Hidden state dimension of GCN
           batch_size,  # Batch size
           max_nodes,  # Maximum number of nodes in graph
           max_labels,  # Maximum number of edge labels
           adj_in,  # Adjacency matrix for in edges
           adj_out,  # Adjacency matrix for out edges
           regularizer, # regularizer for GCN
           num_layers, # Number of GCN Layers
           name="GCN",
           reuse=True):
    out = []
    out.append(gcn_in)

    for layer in range(num_layers):
        gcn_in    = out[-1] # out contains the output of all the GCN layers, intitally contains input to first GCN Layer
        if len(out) > 1: in_dim = gcn_dim  # After first iteration the in_dim = gcn_dim

        with tf.name_scope('%s-%d' % (name,layer)):

            act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])

            for lbl in range(max_labels):

                with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:

                    w_in   = tf.get_variable('w_in',   [in_dim, gcn_dim],  initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
                    b_in   = tf.get_variable('b_in',   [1, gcn_dim],   initializer=tf.constant_initializer(0.0), regularizer=regularizer)

                    w_out  = tf.get_variable('w_out',  [in_dim, gcn_dim], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
                    b_out  = tf.get_variable('b_out',  [1, gcn_dim],  initializer=tf.constant_initializer(0.0), regularizer=regularizer)

                    w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)

                    if wGate:
                        w_gin  = tf.get_variable('w_gin',  [in_dim, 1], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
                        b_gin  = tf.get_variable('b_gin',  [1],   initializer=tf.constant_initializer(0.0), regularizer=regularizer)

                        w_gout = tf.get_variable('w_gout', [in_dim, 1], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
                        b_gout = tf.get_variable('b_gout', [1],   initializer=tf.constant_initializer(0.0), regularizer=regularizer)

                        w_gloop = tf.get_variable('w_gloop',[in_dim, 1], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)

                with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                    inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)
                    in_t    = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], inp_in[i]) for i in range(batch_size)])
                    if dropout != 1.0: in_t    = tf.nn.dropout(in_t, keep_prob=dropout)

                    if wGate:
                        inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)
                        in_gate = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], inp_gin[i]) for i in range(batch_size)])
                        in_gsig = tf.sigmoid(in_gate)
                        in_act   = in_t * in_gsig
                    else:
                        in_act   = in_t

                with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                    inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
                    out_t    = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], inp_out[i]) for i in range(batch_size)])
                    if dropout != 1.0: out_t    = tf.nn.dropout(out_t, keep_prob=dropout)

                    if wGate:
                        inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
                        out_gate = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], inp_gout[i]) for i in range(batch_size)])
                        out_gsig = tf.sigmoid(out_gate)
                        out_act  = out_t * out_gsig
                    else:
                        out_act = out_t

                with tf.name_scope('self_loop'):
                    inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
                    if dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=dropout)

                    if wGate:
                        inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2,0])
                        loop_gsig = tf.sigmoid(inp_gloop)
                        loop_act  = inp_loop * loop_gsig
                    else:
                        loop_act = inp_loop


                act_sum += in_act + out_act + loop_act
            gcn_out = tf.nn.relu(act_sum)
            out.append(gcn_out)

    return out

def getBatches(data, shuffle = True):
    if shuffle: random.shuffle(data)
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        yield data[start_idx : start_idx + batch_size]
        
def get_placeholder_values(X):
    x_paragraph = np.zeros((len(X), paragraph_size), np.int32)
    x_sentences = np.zeros((len(X), num_of_sentence, sentence_size), np.int32)
    x_entity = np.zeros((len(X), entity_size), np.float32)
    
    x_len_paragraph = np.zeros(len(X), np.int32)
    x_len_sentence = np.zeros((len(X), num_of_sentence), np.int32)
    x_len_entity = np.zeros((len(X)), np.float32)
    
    x_mask_sentence = np.full((len(X), num_of_sentence), False, dtype=bool)

    para_id_list = []
    entity_list = []
    
    for i, x in enumerate(X):
        para_id_list.append(x['para_id'])
        entity_list.append(x['exact_entity'])
        index = np.argwhere(np.array(x['complete_paragraph'])==0)
        if index.size==0:
            x_len_paragraph[i] = paragraph_size
        else:
            x_len_paragraph[i] = index[0][0]
        x_paragraph[i] = np.array(x['complete_paragraph'])
        for j, x_sentence in enumerate(x['current_step']):
            index = np.argwhere(np.array(x_sentence)==0)
            if index.size==0:
                x_len_sentence[i][j] = sentence_size
                x_mask_sentence[i][j] = True
            else:
                x_len_sentence[i][j] = index[0][0]
                if index[0][0] != 0:
                    x_mask_sentence[i][j] = True
            x_sentences[i][j] = np.array(x_sentence)
        index = np.argwhere(np.array(x['entity'])==0)
        if index.size==0:
            x_len_entity[i] = entity_size
        else:
            x_len_entity[i] = index[0][0]
        x_entity[i] = np.array(x['entity'])
    return x_paragraph, x_sentences, x_entity, x_len_paragraph, x_len_sentence, x_len_entity, x_mask_sentence, para_id_list, entity_list

def get_placeholder_labels(Y):
    y_known = np.zeros((len(Y), num_of_sentence, num_class), np.int32)
    y_start = np.zeros((len(Y), num_of_sentence), np.int32)
    y_end = np.zeros((len(Y), num_of_sentence), np.int32)
    
    for i, y in enumerate(Y):
        for j, y_sentence in enumerate(y):
            y_known[i][j] = np.array(y_sentence['is_location_known'])
            if y_sentence['after_start']>=0:
                y_start[i][j] = y_sentence['after_start']
            if y_sentence['after_end']>=0:
                y_end[i][j] = y_sentence['after_end']
    return y_known, y_start, y_end
        
def get_adj(edgeList, batch_size, max_nodes, max_labels):
    adj_main_in, adj_main_out = [], []

    for edges in edgeList:
        adj_in, adj_out = {}, {}

        in_ind, in_data   = defaultdict(list), defaultdict(list)
        out_ind, out_data = defaultdict(list), defaultdict(list)

        for src, dest, lbl in edges:
            out_ind[lbl].append((src, dest))
            out_data[lbl].append(1.0)

            in_ind[lbl].append((dest, src))
            in_data[lbl].append(1.0)

        for lbl in range(max_labels):
            if lbl not in out_ind and lbl not in in_ind:
                adj_in[lbl] = sp.coo_matrix((max_nodes, max_nodes))
                adj_out[lbl] = sp.coo_matrix((max_nodes, max_nodes))
            else:
                adj_in[lbl] = sp.coo_matrix((in_data[lbl], zip(*in_ind[lbl])), shape=(max_nodes, max_nodes))
                adj_out[lbl] = sp.coo_matrix((out_data[lbl], zip(*out_ind[lbl])), shape=(max_nodes, max_nodes))

        adj_main_in.append(adj_in)
        adj_main_out.append(adj_out)

    return adj_main_in, adj_main_out
    
def create_feed_dict(batch, wLabels=True, dtype='train'):
    X, Y, DepEdges, SrlEdges = zip(*batch)

    x_paragraph, x_sentences, x_entity, x_len_paragraph, x_len_sentences, x_len_entity, x_mask_sentence, para_id_list, entity_list = get_placeholder_values(X)
#     print (x_entity)
    feed_dict = {}
    feed_dict[input_x_paragraph] = x_paragraph
    feed_dict[input_x_sentence] = x_sentences
    feed_dict[input_x_entity] = x_entity
    feed_dict[input_x_len_paragraph] = x_len_paragraph
    feed_dict[input_x_len_sentence] = x_len_sentences
    feed_dict[input_x_len_entity] = x_len_entity
    feed_dict[input_x_mask_sentence] = x_mask_sentence
    
    if wLabels:
        y_known, y_start, y_end = get_placeholder_labels(Y)
        feed_dict[input_y_known] = y_known
        feed_dict[input_y_start] = y_start
        feed_dict[input_y_end] = y_end

    de_adj_in, de_adj_out = get_adj(DepEdges, batch_size, paragraph_size, num_deLabel)
    srl_adj_in, srl_adj_out = get_adj(SrlEdges, batch_size, paragraph_size, num_srlLabel)
    
    for i in range(batch_size):
        for lbl in range(num_deLabel):
            feed_dict[de_adj_mat_in[i][lbl]] = tf.SparseTensorValue(indices = np.array([de_adj_in[i][lbl].row, de_adj_in[i][lbl].col]).T, values = de_adj_in[i][lbl].data, dense_shape = de_adj_in[i][lbl].shape)
            feed_dict[de_adj_mat_out[i][lbl]] = tf.SparseTensorValue(indices = np.array([de_adj_out[i][lbl].row, de_adj_out[i][lbl].col]).T, values = de_adj_out[i][lbl].data, dense_shape = de_adj_out[i][lbl].shape)
    
    for i in range(batch_size):
        for lbl in range(num_srlLabel):
            feed_dict[srl_adj_mat_in[i][lbl]] = tf.SparseTensorValue(indices = np.array([srl_adj_in[i][lbl].row, srl_adj_in[i][lbl].col]).T, values = srl_adj_in[i][lbl].data, dense_shape = srl_adj_in[i][lbl].shape)
            feed_dict[srl_adj_mat_out[i][lbl]] = tf.SparseTensorValue(indices = np.array([srl_adj_out[i][lbl].row, srl_adj_out[i][lbl].col]).T, values = srl_adj_out[i][lbl].data, dense_shape = srl_adj_out[i][lbl].shape)

    if dtype != 'train':
        feed_dict[dropout]     = 1.0
        feed_dict[rec_dropout] = 1.0

    return feed_dict, para_id_list, entity_list

def createEvalFile(category, start, end, mask, paragraph_id, entity, dtype):
    with codecs.open('data/propara-results2_a-'+dtype+'.txt', 'w') as f:
        for i, id_para in enumerate(paragraph_id):
            before_loc = "unk"
            for j, is_valid in enumerate(mask[i]): 
                if is_valid:
                    if category[i][j]==0:
                        after_loc = "null"
                    elif category[i][j]==1:
                        after_loc = "unk"
                    else:
                        after_loc = (" ").join(paragraph_id_mapping[str(id_para)][start[i][j]:(end[i][j]+1)])
                    f.write(str(id_para)+"\t"+str(j+1)+"\t"+str(entity[i])+"\t"+before_loc+"\t"+after_loc+"\n")
                    before_loc = after_loc

def run_epoch(sess, data, epoch, wLabels, shuffle=True):
    loss_total = 0
    category = []
    start = []
    end = []
    mask = []
    para_id = []
    actual_entity = []
    for step, batch in enumerate(getBatches(data, shuffle)):
        feed, para_id_list, entity_list = create_feed_dict(batch, wLabels=wLabels)
        loss, _, category_all, start_all, end_all, mask_all  = sess.run([total_loss, optimizer, all_category, all_start, all_end, input_x_mask_sentence], feed_dict=feed)
        loss_total+=loss
        category.append(np.array(category_all))
        start.append(np.array(start_all))
        end.append(np.array(end_all))
        mask.append(np.array(mask_all))
        para_id.extend(para_id_list)
        actual_entity.extend(entity_list)
    category = np.vstack(category)
    start = np.vstack(start)
    end = np.vstack(end)
    mask = np.vstack(mask)
    createEvalFile(category, start, end, mask, para_id, actual_entity, "train")
    return loss_total/((step+1)*num_of_sentence)

def fit(sess):
    # Train Model
    saver = tf.train.Saver()
    best_val_loss = 99
    save_path = "models/best_validation1a.ckpt"
    count = 0
    for epoch in range(max_epochs):
        train_loss = run_epoch(sess, data_list['train'], epoch, wLabels=True)
        val_loss = predict(sess, data_list['valid'], wLabels=True)
        count+=1
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            saver.save(sess=sess, save_path=save_path)
            count=1
            print ('[Epoch {}]: Training Loss: {:.5}, Valid Loss: {:.5}*'.format(epoch, train_loss, val_loss))
        else:
            print ('[Epoch {}]: Training Loss: {:.5}, Valid Loss: {:.5}'.format(epoch, train_loss, val_loss))
        if count>patience:
            break
        
        
def predict(sess, data, wLabels=True, shuffle=False):
    loss_total = 0
    category = []
    start = []
    end = []
    mask = []
    para_id = []
    actual_entity = []
    
    for step, batch in enumerate(getBatches(data, shuffle)):
        if not wLabels:
            feed, para_id_list, entity_list = create_feed_dict(batch, wLabels=wLabels, dtype='test')
            category_all, start_all, end_all, mask_all = sess.run([all_category, all_start, all_end, input_x_mask_sentence] , feed_dict = feed)
        else:
            feed, para_id_list, entity_list  = create_feed_dict(batch, wLabels=wLabels, dtype='valid')
            loss, category_all, start_all, end_all, mask_all = sess.run([total_loss, all_category, all_start, all_end, input_x_mask_sentence], feed_dict = feed)
            loss_total+=loss
        category.append(np.array(category_all))
        start.append(np.array(start_all))
        end.append(np.array(end_all))
        mask.append(np.array(mask_all))
        para_id.extend(para_id_list)
        actual_entity.extend(entity_list)
    category = np.vstack(category)
    start = np.vstack(start)
    end = np.vstack(end)
    mask = np.vstack(mask)
    
    if not wLabels:
        createEvalFile(category, start, end, mask, para_id, actual_entity, "test")
    else:
        createEvalFile(category, start, end, mask, para_id, actual_entity, "valid")
        
    return loss_total/((step+1)*num_of_sentence)
        
def run_epoch_sample(sess, data, epoch, wLabels, shuffle=True):
    for step, batch in enumerate(getBatches(data, shuffle)):
        feed = create_feed_dict(batch, wLabels=wLabels)
        all_out = sess.run([all_output], feed_dict=feed)
        break
    return all_out


# In[41]:


# Adding model

with tf.variable_scope('Embeddings') as scope:
    embed_init = getEmbeddings(embedding_location, wrd_list, embedding_dim)
    embed_init = np.vstack( (np.zeros(embedding_dim, np.float32), embed_init))
    embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True, regularizer=regularizer)
    embedding_paragraph = tf.nn.embedding_lookup(embeddings, input_x_paragraph)
    embedding_entity = tf.squeeze(tf.div(tf.reduce_sum(tf.nn.embedding_lookup(embeddings, input_x_entity), 1), tf.tile(tf.expand_dims(input_x_len_entity, 1), [1, embedding_dim])))


# In[42]:


#gcn_output1 = GCNLayer(gcn_in = embedding_paragraph, in_dim = embedding_dim, gcn_dim = gcn_dim,
#                      batch_size = batch_size, max_nodes = paragraph_size, max_labels = num_deLabel, 
#                      adj_in = de_adj_mat_in, adj_out = de_adj_mat_out,
#                      regularizer = regularizer,
#                      num_layers = gcn_layers, name = "GCN")

#gcn_out1 = gcn_output1[-1]


# In[21]:


gcn_output2 = GCNLayer(gcn_in = embedding_paragraph, in_dim = embedding_dim, gcn_dim = gcn_dim,
                      batch_size = batch_size, max_nodes = paragraph_size, max_labels = num_srlLabel, 
                      adj_in = srl_adj_mat_in, adj_out = srl_adj_mat_out,
                      regularizer = regularizer,
                      num_layers = gcn_layers, name = "GCN1")

gcn_out2 = gcn_output2[-1]

#gcn_out = tf.concat([gcn_out1,gcn_out2], axis =2)

# In[49]:


with tf.variable_scope('Bi-LSTM-paragraph') as scope:
    fw_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(lstm_dim), output_keep_prob=rec_dropout)
    bk_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(lstm_dim), output_keep_prob=rec_dropout)
    val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, gcn_out2, sequence_length=input_x_len_paragraph, dtype=tf.float32)
    bilstm_out_paragraph   = tf.concat((val[0], val[1]), axis=2)
    
bilstm_paragraph_dim = 2*lstm_dim

paragraph_mask = tf.sequence_mask(input_x_len_paragraph, paragraph_size, dtype=tf.float32, name='paragraph_mask')


# In[17]:


category_loss = 0
start_loss = 0
end_loss = 0
total_loss = 0

all_category = []
all_start = []
all_end = []
all_mask = []

for index in range(num_of_sentence):
    
    embedding_sentence = tf.nn.embedding_lookup(embeddings, input_x_sentence[:,index])
    
    with tf.variable_scope('Bi-LSTM-sentence', reuse=tf.AUTO_REUSE) as scope:
        fw_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(lstm_dim), output_keep_prob=rec_dropout)
        bk_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(lstm_dim), output_keep_prob=rec_dropout)
        val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embedding_sentence, sequence_length=input_x_len_sentence[:,index], dtype=tf.float32)
        bilstm_out_sentence   = tf.concat((val[0], val[1]), axis=2)
    
    bilstm_sentence_dim = 2*lstm_dim
    
    with tf.variable_scope('attention-entity-sentence', reuse = tf.AUTO_REUSE) as scope:
        w_e_s = tf.get_variable('w_e_s', [embedding_dim, bilstm_sentence_dim], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
        atten_temp = tf.tensordot(embedding_entity, w_e_s, axes=[[1],[0]])
        attention_coeff = tf.nn.softmax(tf.squeeze(tf.matmul(tf.expand_dims(atten_temp, 1), tf.transpose(bilstm_out_sentence, perm=[0,2,1]))))
        bilstm_entity_sentence = tf.multiply(tf.tile(tf.expand_dims(attention_coeff, 2), [1,1,bilstm_sentence_dim]), bilstm_out_sentence)
        bilstm_entity_sentence = tf.reduce_sum(bilstm_entity_sentence, 1)
        
    entity_sentence = tf.concat([embedding_entity, bilstm_entity_sentence], 1)
    entity_sentence_dim = embedding_dim+2*lstm_dim
    
    with tf.variable_scope('attention-entity-sentence-gcn', reuse = tf.AUTO_REUSE) as scope:
        w_e_s_g = tf.get_variable('w_e_s_g', [entity_sentence_dim, bilstm_paragraph_dim], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
        atten_temp_gcn = tf.tensordot(entity_sentence, w_e_s_g, axes=[[1],[0]])
        attention_coeff_gcn = tf.nn.softmax(tf.squeeze(tf.matmul(tf.expand_dims(atten_temp_gcn, 1), tf.transpose(bilstm_out_paragraph, perm=[0,2,1]))))
        entity_sentence_gcn = tf.multiply(tf.tile(tf.expand_dims(attention_coeff_gcn, 2), [1,1,bilstm_paragraph_dim]), bilstm_out_paragraph)
        
    if index==0:
        previous_entity_sentence_gcn = tf.zeros_like(entity_sentence_gcn)
        
    current_entity_sentence_gcn = tf.concat([previous_entity_sentence_gcn, entity_sentence_gcn], 2)
    current_entity_sentence_gcn_vector = tf.reduce_sum(current_entity_sentence_gcn, 1)
    
    with tf.variable_scope('FC1', reuse = tf.AUTO_REUSE) as scope:
        output_logits = tf.layers.dense(current_entity_sentence_gcn_vector, 3)
    
    category_predicted = tf.argmax(tf.nn.softmax(output_logits), 1)
    category_gold = tf.argmax(input_y_known[:,index], 1)
    
    mask_temp = tf.logical_and(tf.equal(category_gold, category_predicted), tf.equal(category_predicted, 2))
    final_mask = tf.to_float(tf.logical_and(mask_temp, input_x_mask_sentence[:,index]))
    
    
    with tf.variable_scope('FC2', reuse = tf.AUTO_REUSE) as scope:
        start_logit = tf.squeeze(tf.layers.dense(current_entity_sentence_gcn, 1))
    
    with tf.variable_scope('FC3', reuse = tf.AUTO_REUSE) as scope:
        end_logit = tf.squeeze(tf.layers.dense(current_entity_sentence_gcn, 1))
    
    start_logit -= (1 - paragraph_mask)*1e30
    end_logit -= (1 - paragraph_mask)*1e30

    start_prob = tf.nn.softmax(start_logit, axis=1)
    end_prob = tf.nn.softmax(end_logit, axis=1)

    # do the outer product
    outer = tf.matmul(tf.expand_dims(start_prob, axis=2), tf.expand_dims(end_prob, axis=1))
    outer = tf.matrix_band_part(outer, 0, max_answer_len)

    start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    
    all_category.append(category_predicted)
    all_start.append(start_pos)
    all_end.append(end_pos)
    
    category_loss += tf.reduce_mean(tf.losses.compute_weighted_loss(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits, labels=input_y_known[:,index]), weights=tf.to_float(input_x_mask_sentence[:,index])))
    start_loss += tf.losses.sparse_softmax_cross_entropy(labels=input_y_start[:, index], logits=start_logit, weights=final_mask)
    end_loss   += tf.losses.sparse_softmax_cross_entropy(labels=input_y_end[:, index], logits=end_logit, weights=final_mask)
        
    previous_entity_sentence_gcn = entity_sentence_gcn
    
all_category = tf.stack(all_category, 1)
all_start = tf.stack(all_start, 1)
all_end = tf.stack(all_end, 1)
    
total_loss = category_loss + start_loss + end_loss

optimizer = tf.train.AdamOptimizer(0.01).minimize(total_loss)


# In[12]:


# Training the model
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    fit(sess)


# In[16]:


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "models/best_validation1a.ckpt")
    val_loss = predict(sess, data_list['valid'], wLabels=True, shuffle=False)
    print ("Validation loss", val_loss)


# In[13]:


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "models/best_validation1a.ckpt")
    predict(sess, data_list['test'], wLabels=False, shuffle=False)

