import numpy as np
import bcolz
import cpickle as pickle

def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def get_glove(path, res_path, name):
    with open(f'{path}glove.{name}.txt', 'r') as f:
        lines = [line.split() for line in f]
    
    words = [d[0] for d in lines]
    vecs = np.stack(np.array(d[1:], dtype=np.float32) for d in lines)
    wordidx = {o:i for i,o in enumerate(words)}

    save_array(f'{res_path}{name}.dat', vecs)
    pickle.dump(words, open(f'{res_path}{name}_words.pkl', 'wb'))
    pickle.dump(wordidx, open(f'{res_path}{name}_idx.pkl', 'wb'))


def load_glove(loc):
    return (load_array(f'{loc}.dat'),
        pickle.load(open(f'{loc}_words.pkl', 'rb')),
        pickle.load(open(f'{loc}_idx.pkl', 'rb')))

