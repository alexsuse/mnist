#!/usr/bin/env python
"""
arg1 :: train.csv path
arg2 :: test.csv path
arg3 :: transformed training set output file
arg4 :: transformed test set output file

takes features from arg1 and arg2, fits pca, then transforms
and outputs to arg3 and arg4

"""


import sys
import csv
import sklearn.decomposition as dec
import numpy as np

def load_from_csv( train_file , test_file ):


    train = csv.reader( open( train_file ) )
    test = csv.reader( open( test_file ) )

    train.next()
    test.next()

    labels = []
    train_features = []
    test_features = []

    for line in train:
        labels.append( line[0] )
        train_features.append( np.array( line[1:] ) )

    print 'read training set'

    for line in test:
        test_features.append( np.array(line))

    print 'read test set'

    #features = np.array( test_features + train_features )

    labels = np.array( labels, dtype=int)
    train_features = np.array( train_features, dtype=float )/256.0
    test_features = np.array( test_features, dtype=float )/256.0

    return labels,train_features,test_features



def perform_sk_preprocessing( preprocessor, ppargs , train_features, test_features ):

    print 'now %s'%preprocessor.__name__

    pca = preprocessor( **ppargs )
    pca.fit( train_features )

    print 'finished fitting PCA'

    train_features = pca.transform( train_features )
    test_features = pca.transform( test_features )

    return pca,train_features,test_features

if __name__=='__main__':

    print __doc__
    
    labels, train_features, test_features = load_from_csv( sys.argv[1], sys.argv[2] )
    
    pca, train_features, test_features = perform_sk_preprocessing( dec.PCA,
            {'whiten':True,'n_components':100},
            train_features,
            test_features ) 

    train_output = csv.writer( open(sys.argv[3],'wb') ) 
    test_output = csv.writer( open(sys.argv[4],'wb') )

    for i in range( train_features.shape[0] ):
        train_output.writerow( train_features[i,:] )

    for i in range( test_features.shape[0] ):
        test_output.writerow( [labels[i],test_features[i,:]] )
