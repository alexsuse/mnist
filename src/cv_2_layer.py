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
    train, tr_label, test, te_label, W, bh = args
    W2, b2 = pre_train_log_reg( train, tr_label )
    mlp = autoencoder.TwoLayerPerceptron( 784, W.shape[1], 10,
                W_1_init = W, b_1_init = bh, W_2_init = W2, b_2_init = b2 )
    mlp.fit( train, tr_label, nbatches = 100, training_epochs = 1000)
    return mlp.score( test, te_label )

def pre_train_logreg( args ):
    train, labels = args
    logreg = autoencoder.LogisticRegression
    logreg.fit( train, labels )
    return logreg.W.get_value(), logreg.b.get_value()

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
        from IPython.parallel import Client
        rc = Client()
        dview = rc[:4]
        with dview.sync_imports():
            import autoencoder
        print 'we iz gotz parallelz!!'
    except:
        print "No parallel ipython, sorry!"
        exit()


    labels, train, test = pp.load_from_csv( sys.argv[3], sys.argv[4] )

    train = train
    labels = np.array(labels)

    unsuper = np.append( train, test, axis = 0 )
    print unsuper.shape

    if ae_pretrain:
        print 'ae pretraining...'
        da = ae.dA(784,400)
        da.fit(unsuper.reshape((unsuper.shape[0]/100,100,784)), training_epochs = 200)
        W = da.W.get_value()
        bh = da.b_h.get_value()
        print 'pretrained ae'
        with open(sys.argv[1],'wb') as f:
            pic.dump([W,bh],f)
            print 'dumped ae stuffz to %s'%sys.argv[1]
#
#    if pretrain:
#        print 'lr pretraining...'
#        da = ae.dA(784,400,W1 = W, bh = bh)
#        lr = ae.LogisticRegression( 400, 10 )
#        lr.fit( da.transform(train).reshape((420,100,400)), labels.reshape((420,100)), training_epochs = 200)
#        W2 = lr.W.get_value()
#        b2 = lr.b.get_value()
#        print 'pretrained logreg'
#        with open(sys.argv[2],'wb') as f:
#            pic.dump([W2,b2],f)
#            print 'dumped logreg stuffz to %s'%sys.argv[2]


    cross_val = cv.KFold( train.shape[0], n_folds=16) 
    
    sets = [(train[i],labels[i],train[j],labels[j], W, bh, W2, b2) for i,j in cross_val]
    
    print 'trying parallel evaluation...'

    scores = dview.map( train_mlp_and_score, sets )

    print scores.get()
