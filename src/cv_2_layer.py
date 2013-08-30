#!/usr/bin/env python
"""
cross-validate my 2-layer MLP implementation with cross_val_score from sklearn

arg1 :: first layer params pickle
arg2 :: second layer params pickle
arg3 :: train.csv
arg4 :: test.csv

@author = Alex Susemihl
"""

import sys
import preprocess as pp
import cPickle as pic
import numpy
import autoencoder as ae
from train_classifier import print_preds_to_csv
import sklearn.cross_validation as cv

if __name__ == '__main__':
    print __doc__


    with open( sys.argv[1], 'r' ) as f:
        W, _, bh = pic.load( f )

    try:
        with open( sys.argv[2] ,'r') as f:

            W2, b2 = pic.load(f)
        pretrain = False
    except:
        W2 = None
        b2 = None
        pretrain = True

    try:
        from IPython.Parallel import Client
    except:
        print "No parallel ipython, sorry!"
        exit()

    rc = Client()
    print rc.ids
    dview = rc[:4]

    @dview.parallel
    def train_mlp_and_score( args ):
        train, tr_label, test, te_label = args
        mlp = ae.TwoLayerPerceptron( 784, W.shape[1], 10, W_1_init = W, b_1_init = bh, W_2_init = W2, b_2_init = b2 )
        mlp.fit(train, tr_label, nbatches = 100)
        return mlp.score( test, te_label )

    labels, train, test = pp.load_from_csv( sys.argv[3], sys.argv[4] )

    cross_val = cv.KFold( 42000, k=8) 
    
    sets = [(train[i],labels[i],train[j],labels[j]) for i,j in cross_val]

    scores = dview.map_sync( train_mlp_and_score, sets )

    print scores 
