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
import numpy as np

#    @dview.parallel
def train_mlp_and_score( args ):
    import imp
    ae = imp.load_source('autoencoder','/Users/alex/mnist/src/autoencoder.py')
    train, tr_label, test, te_label, W, bh, W2, b2 = args
    mlp = ae.TwoLayerPerceptron( 784, W.shape[1], 10, W_1_init = W, b_1_init = bh, W_2_init = W2, b_2_init = b2 )
    mlp.fit( train, tr_label, nbatches = 100, training_epochs = 300)
    return mlp.score( test, te_label )

if __name__ == '__main__':
    print __doc__

    try:
        with open( sys.argv[1], 'r' ) as f:
            W, bh = pic.load( f )
        ae_pretrain = False
        print 'loaded ae stuffz...'
    except:
        W = None
        bh = None
        ae_pretrain = True
    
    try:
        with open( sys.argv[2] ,'r') as f:
            W2, b2 = pic.load(f)
        pretrain = False
        print 'loaded lr stuffz...'
    except:
        W2 = None
        b2 = None
        pretrain = True


    try:
        from IPython.parallel import Client
        rc = Client()
        dview = rc[:4]
        print 'we iz gotz parallelz!!'
    except:
        print "No parallel ipython, sorry!"
        exit()


    labels, train, test = pp.load_from_csv( sys.argv[3], sys.argv[4] )

    train = train[:15000]
    labels = np.array(labels)[:15000]

    if ae_pretrain:
        print 'ae pretraining...'
        da = ae.dA(784,400)
        da.fit(train.reshape((420,100,784)), training_epochs = 200)
        W = da.W.get_value()
        bh = da.b_h.get_value()
        print 'pretrained ae'
        with open(sys.argv[1],'wb') as f:
            pic.dump([W,bh],f)
            print 'dumped ae stuffz to %s'%sys.argv[1]

    if pretrain:
        print 'lr pretraining...'
        da = ae.dA(784,400,W1 = W, bh = bh)
        lr = ae.LogisticRegression( 400, 10 )
        lr.fit( da.transform(train).reshape((420,100,400)), labels.reshape((420,100)), training_epochs = 200)
        W2 = lr.W.get_value()
        b2 = lr.b.get_value()
        print 'pretrained logreg'
        with open(sys.argv[2],'wb') as f:
            pic.dump([W2,b2],f)
            print 'dumped logreg stuffz to %s'%sys.argv[2]


    cross_val = cv.KFold( train.shape[0], n_folds=5) 
    
    sets = [(train[i],labels[i],train[j],labels[j], W, bh, W2, b2) for i,j in cross_val]
    
    print 'trying parallel evaluation...'

    scores = dview.map( train_mlp_and_score, sets )

    print scores.get()
