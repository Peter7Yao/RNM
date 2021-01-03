import math
from .Init import *
import scipy.spatial
import json
import pickle as pkl
import os
import numpy as np


def get_vmat(e, KG):
    du = [1] * e
    rel = set()
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]] += 1
            du[tri[2]] += 1
        rel.add(tri[1])
        
    M = {}
    M_tri = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        
        if (tri[0], tri[1], tri[2]) not in M_tri:
            M_tri[(tri[0], tri[1], tri[2])] = 1
        
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass
        
        if (tri[2], tri[1], tri[0]) not in M_tri:
            M_tri[(tri[2], len(rel)+tri[1], tri[0])] = 1

    return M, du, M_tri


def rfunc(KG, e):
    head = {}
    cnt = {}
    rel_type = {}
    cnt_r = {}
    for tri in KG:
        r_e = str(tri[1]) + ' ' + str(tri[2])
        if r_e not in cnt:
            cnt[r_e] = 1
            head[r_e] = set([tri[0]])
        else:
            cnt[r_e] += 1
            head[r_e].add(tri[0])
        
        if tri[1] not in cnt_r:
            cnt_r[tri[1]] = 1

    r_num = len(cnt_r)
    
    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value
    
    del cnt
    del head
    del cnt_r
    cnt = {}
    head = {}
    
    for tri in KG:
        r_e_new = str(tri[1]+r_num) + ' ' + str(tri[0])
        if r_e_new not in cnt:
            cnt[r_e_new] = 1
            head[r_e_new] = set([tri[2]])
        else:
            cnt[r_e_new] += 1
            head[r_e_new].add(tri[2])
    
    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value
    
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
    
    return head_r, tail_r, rel_type


def compute_r(inlayer, head_r, tail_r, dimension):
    head_l=tf.transpose(tf.constant(head_r,dtype=tf.float32))
    tail_l=tf.transpose(tf.constant(tail_r,dtype=tf.float32))
    L=tf.matmul(head_l,inlayer)/tf.expand_dims(tf.reduce_sum(head_l,axis=-1),-1)
    R=tf.matmul(tail_l,inlayer)/tf.expand_dims(tf.reduce_sum(tail_l,axis=-1),-1)
    
    r_forward=tf.concat([L,R],axis=-1)
    r_reverse=tf.concat([-L,-R],axis=-1)
    r_embeddings = tf.concat([r_forward,r_reverse], axis=0)
    
    w_r = glorot([dimension * 2, dimension])
    r_embeddings_new = tf.matmul(r_embeddings, w_r)
    
    return r_embeddings, r_embeddings_new


def get_sparse_tensor(e, KG, rel_type):
    print('getting a sparse tensor...')
    M0, du, M_tri = get_vmat(e, KG)
    ind = []
    val = []
    for fir, sec in M0:
        ind.append((sec, fir))
        val.append(M0[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))

    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])

    return M0, M, M_tri


def get_se_input_layer(e, dimension, file_path):
    print('adding the primal input layer...')
    with open(file=file_path, mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    
    input_embeddings = tf.convert_to_tensor(embedding_list)
    ent_embeddings = tf.Variable(input_embeddings)
    return tf.nn.l2_normalize(ent_embeddings, 1)


def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a layer...')
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension, dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1


def get_loss_pre(outlayer, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right):
    left = ILL[:, 0]
    right = ILL[:, 1]
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    
    return (tf.reduce_mean(L1) + tf.reduce_mean(L2)) / 2.0


def get_loss_transe(output_h, output_r, head, rel, tail):
    h_ebd = tf.nn.embedding_lookup(output_h, head)
    r_ebd = tf.nn.embedding_lookup(output_r, rel)
    t_ebd = tf.nn.embedding_lookup(output_h, tail)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(h_ebd + r_ebd - t_ebd), 1))
    
    return loss
    

def build(dimension, act_func, gamma, k, vec_path, e, KG):
    tf.reset_default_graph()
    head_r, tail_r, rel_type = rfunc(KG, e)
    input_layer = get_se_input_layer(e, dimension, vec_path)
    M0, M, M_tri = get_sparse_tensor(e, KG, rel_type)

    print('KG structure embedding')
    hidden_layer_1 = add_diag_layer(
        input_layer, dimension, M, act_func, dropout=0.0)
    hidden_layer = highway(input_layer, hidden_layer_1, dimension)
    hidden_layer_2 = add_diag_layer(
        hidden_layer, dimension, M, act_func, dropout=0.0)
    output_h = highway(hidden_layer, hidden_layer_2, dimension)
    print('shape of output_h: ', output_h.get_shape())

    output_r, output_r_w = compute_r(output_h, head_r, tail_r, dimension)
    
    ILL = tf.placeholder(tf.int32, [None, 2], "ILL")
    
    print("compute pre-training loss")
    neg_left = tf.placeholder(tf.int32, [None], "neg_left") 
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    neg2_left = tf.placeholder(tf.int32, [None], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [None], "neg2_right")
    
    loss_pre = get_loss_pre(output_h, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
    
    head = tf.placeholder(tf.int32, [None], "head")
    rel = tf.placeholder(tf.int32, [None], "rel")
    tail = tf.placeholder(tf.int32, [None], "tail")
    
    loss_transe = get_loss_transe(output_h, output_r_w, head, rel, tail)
    loss_all = loss_pre + 0.001 * loss_transe

    return output_h, output_r, loss_pre, loss_all, M0, rel_type


def get_neg(ILL, output_layer, k, batchnum):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    for p in range(batchnum):
        head = int(t / batchnum * p)
        if p==batchnum-1:
            tail=t
        else:
            tail = int(t / batchnum * (p + 1))
        sim = scipy.spatial.distance.cdist(
            ILL_vec[head:tail], KG_vec, metric='cityblock')
        for i in range(tail - head):
            rank = sim[i, :].argsort()
            neg.append(rank[0: k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))

    return neg


def get_kg_data(M0, r_num):
    kg = {}
    for tri in M0:
        if tri[0] == tri[2]:
            continue
        if tri[0] not in kg:
            kg[tri[0]] = set()
        if tri[2] not in kg:
            kg[tri[2]] = set()
        
        kg[tri[0]].add((tri[1], tri[2]))
        kg[tri[2]].add((tri[1]+r_num, tri[0]))
    
    return kg


def get_nbr(L, R, ref, kg, max_len=100):
    nbr_L = np.zeros([len(L) ,max_len])
    nbr_R = np.zeros([len(L) ,max_len])
    mask = np.zeros([len(L) ,max_len])

    for i in range(len(L)):
        j = 0
        for n_1 in kg[L[i]]:
            for n_2 in kg[R[i]]:
                if (n_1[1], n_2[1]) in ref and j < max_len:
                    nbr_L[i,j] = n_1[0]
                    nbr_R[i,j] = n_2[0]
                    mask[i,j] = 1
                    j += 1

    return nbr_L, nbr_R, mask


def training(output_h, loss_pre, loss_all, learning_rate, epochs, pre_epochs, ILL, e, k, save_suffix, dimension,
            train_batchnum, test, M0, e1, e2, KG, rel_type, output_r, l1, r_num, ILL_r):
    from include.Test import get_hits, get_rel_hits
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_pre)
    train_all = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)

    print('initializing...')
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    print('running...')
    J = []
    ILL = np.array(ILL)
    t = len(ILL)
    ILL_reshape = np.reshape(ILL, 2 * t, order='F')
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    
    kg_tri = []
    for tri in KG:
        kg_tri.append([tri[0], tri[1], tri[2]])
    tri_num = len(kg_tri)
    kg_tri = np.array(kg_tri)
    
    dn = 'RNM'

    if not os.path.exists(dn + '_' + "model/"):
        os.makedirs(dn + '_' + "model/")
    
    if os.path.exists(dn + '_' + "model/save_"+save_suffix+".ckpt.meta"):
        saver.restore(sess, dn + '_' + "model/save_"+save_suffix+".ckpt")
        start_epoch=pre_epochs
    else:
        start_epoch=0
    
    for i in range(start_epoch, epochs):
        
        if i % pre_epochs == 0:
            out = sess.run(output_h)
            print('data preparation')
            neg2_left = get_neg(ILL[:, 1], out, k, train_batchnum)
            neg_right = get_neg(ILL[:, 0], out, k, train_batchnum)
        
        for j in range(train_batchnum):
            beg = int(t / train_batchnum * j)
            if j == train_batchnum-1:
                end = t
            else:
                end = int(t / train_batchnum * (j + 1))
            
            feeddict = {}
            feeddict["ILL:0"] = ILL[beg:end]
            feeddict["neg_left:0"] = neg_left.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg_right:0"] = neg_right.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg2_left:0"] = neg2_left.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg2_right:0"] = neg2_right.reshape(
                (t, k))[beg:end].reshape((-1,))
            
            if i < pre_epochs:
                _ = sess.run([train_step], feed_dict=feeddict)
            else:
                beg = int(tri_num / train_batchnum * j)
                if j == train_batchnum-1:
                    end = tri_num
                else:
                    end = int(tri_num / train_batchnum * (j + 1))
                
                feeddict["head:0"] = kg_tri[beg:end, 0]
                feeddict["rel:0"] = kg_tri[beg:end, 1]
                feeddict["tail:0"] = kg_tri[beg:end, 2]
                
                _ = sess.run([train_all], feed_dict=feeddict)
        
        if (i+1) % 10 == 0 or i == 0:
            print('%d/%d' % (i + 1, epochs), 'epochs...')

        if i == pre_epochs - 1:
            save_path = saver.save(sess, dn + '_' + "model/save_"+save_suffix+".ckpt")
            print("Save to path: ", save_path)
        
        if i == epochs - 1:
            print('Testing')
            iters = 3
            outvec, outvec_r = sess.run([output_h, output_r])
            print('iter: 1')
            
            sim_e, sim_r = get_hits(outvec, outvec_r, l1, KG, ILL, rel_type, test, None, None)
            for t in range(iters):
                print('iter: ' + str(t+2))
                sim_e, sim_r = get_hits(outvec, outvec_r, l1, KG, ILL, rel_type, test, sim_e, sim_r)
            
            get_rel_hits(outvec, outvec_r, l1, KG, ILL, rel_type, test, ILL_r)

    sess.close()
    
    return
