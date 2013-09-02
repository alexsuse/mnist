#!/usr/bin/env python
"""
cross-validate my 2-layer MLP implementation with cross_val_score from sklearn

arg1 :: first layer params pickle
arg2 :: train.csv
arg3 :: test.csv

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

def train_mlp_and_score( args ):
    import sys
    import theano.tensor as T
    sys.path.append('/home/susemihl/mnist/src')
    import autoencoder
    
    train, tr_label, test, te_label, W, bh = args
    
    logreg = autoencoder.LogisticRegression(n_in = 400, n_out = 10)
    #create da to transform data
    transf_train = T.nnet.sigmoid( T.dot( train, W ) + bh ).eval()
    logreg.fit( transf_train, tr_label )
    #get params from logreg layer
    W2, b2 = logreg.W.get_value(), logreg.b.get_value()
    del logreg
    
    mlp = autoencoder.TwoLayerPerceptron( 784, W.shape[1], 10,
                W_1_init = W, b_1_init = bh, W_2_init = W2, b_2_init = b2 )
    mlp.fit( train, tr_label, nbatches = 150, training_epochs = 1000)
    
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
        from IPython.parallel import Client
        rc = Client()
        dview = rc[:]
        with dview.sync_imports():
            pass
        print 'we iz gotz parallelz!!'
    except:
        print "No parallel ipython, sorry!"
        exit()

    labels, train, test = pp.load_from_csv( sys.argv[2], sys.argv[3] )

    labels = np.array(labels)

    if ae_pretrain:
        unsuper = np.append( train, test, axis = 0 )
        print unsuper.shape
        print 'ae pretraining...'
        da = ae.dA(784,400)
        da.fit(unsuper.reshape((unsuper.shape[0]/100,100,784)), training_epochs = 200)
        W = da.W.get_value()
        bh = da.b_h.get_value()
        print 'pretrained ae'
        with open(sys.argv[1],'wb') as f:
            pic.dump([W,bh],f)
            print 'dumped ae stuffz to %s'%sys.argv[1]

    cross_val = cv.KFold( train.shape[0], k=8) 
   
    sets = [(train[i],labels[i],train[j],labels[j], W, bh) for i,j in cross_val]
    
    print 'trying parallel evaluation...'

    scores = dview.map( train_mlp_and_score, sets )
    scores = scores.get()
    with open('scores.out','wb') as f:
        for s in scores:
             f.write(str(s))
    print scores
