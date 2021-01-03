import tensorflow as tf
import argparse
from include.Config import Config
from include.Model import build, training, get_nbr
from include.Load import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='DBP15k')
parser.add_argument('--lang', type=str, help='zh_en, ja_en and fr_en')

args = parser.parse_args()

if __name__ == '__main__':
    config = Config(args.dataset,args.lang)
    e1 = set(loadfile(config.e1, 1))
    e2 = set(loadfile(config.e2, 1))
    e = len(e1 | e2)

    ILL = loadfile(config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * config.seed])
    test = ILL[illL // 10 * config.seed:]
    
    ILL_r = loadfile(config.ill_r, 2)
    
    KG1 = loadfile(config.kg1, 3)
    KG2 = loadfile(config.kg2, 3)
    
    r_kg_1 = set()
    r_kg = set()

    for tri in KG1:
        r_kg_1.add(tri[1])
        r_kg.add(tri[1])
    
    for tri in KG2:
        r_kg.add(tri[1])
    
    output_h, output_r, loss_pre, loss_all, M0, rel_type  = \
        build(config.dim, config.act_func, config.gamma, config.k, config.vec, e, KG1 + KG2)
    training(output_h, loss_pre, loss_all, config.lr, config.epochs, config.pre_epochs, train, e,
                         config.k, config.save_suffix, config.dim, config.train_batchnum, test, M0, 
                         e1, e2, KG1 + KG2, rel_type, output_r, len(r_kg_1), len(r_kg), ILL_r)
    