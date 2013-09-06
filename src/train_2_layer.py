#!/usr/bin/env python
"""
train logreg layer on top of the autoencoder with backprop

arg1 :: autoencoder pickle file
arg2 :: train.csv
arg3 :: test.csv
arg4 :: output.csv

"""

import sys
import preprocess as pp
import cPickle as pic
import numpy
import autoencoder as ae
import sklearn.cross_validation as cv

if __name__ == '__main__':
    
    print __doc__

    with open( sys.argv[1], 'r' ) as f:
        W, _, bh = pic.load( f )

    try:
        with open('logreg_backup_'+str(W.shape[0]),+'.pkl','r') as f:

            W2, b2 = pic.load(f)
        pretrain = False
    except:
        W2 = None
        b2 = None
        pretrain = True

    labels, train, test = pp.load_from_csv( sys.argv[2], sys.argv[3] )

    train, test, labels, test_labels = cv.train_test_split( train, labels, test_size = 0.2 )

    mlp = ae.TwoLayerPerceptron( 784, W.shape[1], 10, W_1_init = W, b_1_init = bh, W_2_init = W2, b_2_init = b2 )

    if pretrain:
    #greedy pretraining

        transf_train = mlp.firstLayer.transform( train )

        mlp.secondLayer.fit( transf_train.reshape((transf_train.shape[0]/100,100,400)),
                            labels.reshape((labels.shape[0]/100,100)), training_epochs = 100)

        with open('logreg_backup.pkl','wb') as f:
            pic.dump([mlp.secondLayer.W.get_value(), mlp.secondLayer.b.get_value()],f)
    
    train = train.reshape((train.shape[0]/100,100,784))
    labels = labels.reshape((labels.shape[0]/100,100))

    try:
        mlp.fit_and_validate( train, labels, test, test_labels )
    except:
        pass

    preds = mlp.predict( test )

    pp.print_preds_to_csv( preds, sys.argv[4] ) 
