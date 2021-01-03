import numpy as np
import scipy.spatial


def get_hits(vec, vec_r, l1, M0, ref_data, rel_type, test_pair, sim_e, sim_r, top_k=(1, 10)):
    ref = set()
    for pair in ref_data:
        ref.add((pair[0], pair[1]))
    
    r_num = len(vec_r)//2
    
    kg = {}
    rel_ent = {}
    for tri in M0:
        if tri[0] == tri[2]:
            continue
        if tri[0] not in kg:
            kg[tri[0]] = set()
        if tri[2] not in kg:
            kg[tri[2]] = set()
        if tri[1] not in rel_ent:
            rel_ent[tri[1]] = set()
        
        kg[tri[0]].add((tri[1], tri[2]))
        kg[tri[2]].add((tri[1]+r_num, tri[0]))
        rel_ent[tri[1]].add((tri[0], tri[2]))

    
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])
    Lvec = vec[L]
    Rvec = vec[R]
    
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    
    if sim_e is None:
        sim_e = sim
    
    R_set = {}
    
    for i in range(len(L)):
        j = sim_e[i, :].argsort()[0]
        if sim_e[i,j] >= 5:
            continue
        if j in R_set and sim_e[i, j] < R_set[j][1]:
            ref.remove((L[R_set[j][0]], R[j]))
            ref.add((L[i], R[j]))
            R_set[j] = (i, sim_e[i, j])
        if j not in R_set:
            ref.add((L[i], R[j]))
            R_set[j] = (i, sim_e[i, j])
    
    if sim_r is None:
        sim_r = scipy.spatial.distance.cdist(vec_r[:l1], vec_r[l1:r_num], metric='cityblock')
    
    ref_r = set()
    for i in range(l1):
        j = sim_r[i, :].argsort()[0]
        if sim_r[i,j] < 3:
            ref_r.add((i, j+l1))
            ref_r.add((i+r_num,j+l1+r_num))
    
    
    for i in range(len(L)):
        rank = sim[i, :].argsort()[:100]
        for j in rank:
            if R[j] in kg:
                match_num = 0
                for n_1 in kg[L[i]]:
                    for n_2 in kg[R[j]]:
                        if (n_1[1], n_2[1]) in ref and (n_1[0], n_2[0]) in ref_r:
                            w = rel_type[str(n_1[0]) + ' ' + str(n_1[1])] * rel_type[str(n_2[0]) + ' ' + str(n_2[1])]
                            match_num += w
                sim[i,j] -= 10 * match_num / (len(kg[L[i]]) + len(kg[R[j]]))
    
    sim_r = scipy.spatial.distance.cdist(vec_r[:l1], vec_r[l1:r_num], metric='cityblock')
    
    for i in range(l1):
        rank = sim_r[i, :].argsort()[:20]
        for j in rank:
            if i in rel_ent and j+l1 in rel_ent:
                match_num = 0
                for n_1 in rel_ent[i]:
                    for n_2 in rel_ent[j+l1]:
                        if (n_1[0],n_2[0]) in ref and (n_1[1],n_2[1]) in ref:
                            match_num += 1
                sim_r[i,j] -= 200 * match_num / (len(rel_ent[i])+len(rel_ent[j+l1]))

    mrr_l = []
    mrr_r = []
    
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_l.append(1.0 / (rank_index+1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_r.append(1.0 / (rank_index+1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    print('Entity Alignment (left):')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f' % (np.mean(mrr_l)))
    
    print('Entity Alignment (right):')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.4f' % (np.mean(mrr_r)))
    
    return sim, sim_r


def get_rel_hits(vec, vec_r, l1, M0, ref_data, rel_type, test_pair, ILL_r, top_k=(1, 10)):
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])
    Lvec = vec[L]
    Rvec = vec[R]
    
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    
    ref = set()
    for pair in ref_data:
        ref.add((pair[0], pair[1]))
    
    for i in range(len(L)):
        j = sim[i, :].argsort()[0]
        if sim[i,j] < 5:
            ref.add((L[i],R[j]))

    rel_ent = {}
    for tri in M0:
        if tri[0] == tri[2]:
            continue
        if tri[1] not in rel_ent:
            rel_ent[tri[1]] = set()

        rel_ent[tri[1]].add((tri[0], tri[2]))
    
    L_r = np.array([r1 for r1, r2 in ILL_r])
    R_r = np.array([r2 for r1, r2 in ILL_r])
    Lvec_r = vec_r[L_r]
    Rvec_r = vec_r[R_r]
    
    sim_r = scipy.spatial.distance.cdist(Lvec_r, Rvec_r, metric='cityblock')
    
    for i in range(len(ILL_r)):
        rank = sim_r[i, :].argsort()[:300]
        for j in rank:
            if L_r[i] in rel_ent and R_r[j] in rel_ent:
                match_num = 0
                for n_1 in rel_ent[L_r[i]]:
                    for n_2 in rel_ent[R_r[j]]:
                        if (n_1[0],n_2[0]) in ref and (n_1[1],n_2[1]) in ref:
                            match_num += 1
                sim_r[i,j] -= 200 * match_num / (len(rel_ent[L_r[i]])+len(rel_ent[R_r[j]]))
                
    
    top_lr = [0] * len(top_k)
    for i in range(Lvec_r.shape[0]):
        rank = sim_r[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    for i in range(Rvec_r.shape[0]):
        rank = sim_r[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    print('Relation Alignment (left):')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(ILL_r) * 100))
    
    print('Relation Alignment (right):')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(ILL_r) * 100))
    
    return
