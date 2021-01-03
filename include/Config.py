import tensorflow as tf


class Config():
    def __init__(self, d='', l=''):
        dataset = d
        language = l
        prefix = 'data/' + dataset + '/' + language 
        self.kg1 = prefix + '/triples_1'
        self.kg2 = prefix + '/triples_2'
        self.e1 = prefix + '/ent_ids_1'
        self.e2 = prefix + '/ent_ids_2'
        self.ill = prefix + '/ref_ent_ids'
        self.ill_r = prefix + '/ref_r_ids'
        self.vec = prefix + '/vectorList.json'
        self.save_suffix = dataset+'_'+language

        self.epochs = 60
        self.pre_epochs = 50
        self.train_batchnum=20
        self.test_batchnum=50

        self.dim = 300
        self.act_func = tf.nn.relu
        self.gamma = 1.0  # margin based loss
        self.k = 125  # number of negative samples for each positive one
        self.seed = 3  # 30% of seeds
        self.lr = 0.001
