#!/usr/bin/env python
"""
preprocess files with autoencoder

argv1 :: train.csv - train data, csv, labels, features
argv2 :: test.csv - test data, csv, features
argv3 :: int, number of ae filters
argv4 :: train_out.csv - output of transformed training data
argv5 :: test_out.csv s output of transformed test data
argv6 :: 'whiten' for whitening

"""

import preprocess as pp
import sys
import autoencoder as ae
import csv
import cPickle as pic

import sklearn.decomposition as dec

if __name__=='__main__':
    
    print __doc__

    labels, train_features, test_features = pp.load_from_csv(sys.argv[1], sys.argv[2])

    try:

        if sys.argv[6]=='whiten':

            pca = dec.PCA( whiten=True )

            pca.fit( train_features )

            train_features = pca.transform( train_features )
            
            test_features = pca.transform( test_features )

        else:

            print 'unrecognized option'
    
    except:

        pass

    batch_size = 100
  
    n_batches = int( train_features.shape[0] / batch_size ) 

    assert n_batches*batch_size == train_features.shape[0]

    batch_train_features = train_features.reshape(( n_batches, batch_size, train_features.shape[1] ))

    aE = ae.dA(batch_train_features.shape[2], int( sys.argv[3] ), regL = 1e-3 )

    aE.fit( batch_train_features, l_rate = 5, tol=1e-5, training_epochs = 2000 )

    with open( 'pickle_ae_' + str( sys.argv[3] ) + '.pkl', 'wb' ) as fi:
        print 'dumping aE parmeters to %s'%'pickle_ae_' + str( sys.argv[3] ) + '.pkl'

        W = aE.W.get_value()
        bias_v = aE.b_h.get_value()
        bias_h = aE.b_v.get_value()
        
        pic.dump( [W, bias_h, bias_v] , fi ) 

    train_features = aE.transform( train_features )
    test_features = aE.transform( test_features )

    train_out = csv.writer( open( sys.argv[3] + sys.argv[4], 'wb') )
    
    train_out.writerow(['labels','features'])

    for i in range(train_features.shape[0]):
        train_out.writerow( [labels[i]]+ [train_features[i,j] for j in range(train_features.shape[1])] )

    test_out = csv.writer( open( sys.argv[3] + sys.argv[5], 'wb') ) 

    test_out.writerow(['features'])

    for i in range(test_features.shape[0]):
        test_out.writerow( [test_features[i,j] for j in range(test_features.shape[1])] )
