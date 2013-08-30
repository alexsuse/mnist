#!/usr/bin/env python
"""
trains a sklearn classifier
(default is logistic regression with L1 penalty)

arg1 :: train.csv - csv file with labels, features
arg2 :: test.csv - csv file with features
arg3 :: predictions.csv - csv file to output predicted class probabilities
arg4 :: preprocessig args - only 'whiten' supported

"""

import csv
#import sklearn.linear_model as lm
import sklearn.ensemble as ens
import preprocess as pp
import sys
from sklearn.cross_validation import KFold

def train_and_predict( classifier, class_args, train, labels, test ):
    cl = classifier( **class_args )
    cl.fit( train, labels )

    print 'training score %lf'%cl.score( train, labels )

    predictions = cl.predict( test )
    return predictions

def train_and_score( classifier, class_args, train, train_l, test, test_l ):
    cl = classifier( **class_args )
    cl.fit( train, train_l )

    print 'validation score %lf'%cl.score( test, test_l )

def print_preds_to_csv( preds, filename ):
    with csv.writer( open( filename, 'wb' ) ) as writer:
        for i in xrange(preds.shape[0]):
            writer.writerow([i,preds[i]])

if __name__=='__main__':
    print __doc__


    labels, train, test = pp.load_from_csv( sys.argv[1], sys.argv[2] )

    try:

        if sys.argv[4]=='whiten':
            print '\n...whitening and PCA-ing the data'

            pca = dec.PCA( whiten=True )
            pca.fit( train )

            train = pca.transform( train )
            test = pca.transform( test )

            print """PCA'd and whitened data\n"""

        else:

            print 'unrecognized option'
    
    except:
        pass

    kf = KFold( train.shape[0], k = 4 )

#    preds = train_and_predict( lm.LogisticRegression, {'penalty':'L1'},

    for train_index, test_index in kf:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        preds = train_and_score( ens.RandomForestClassifier,
                {'n_estimators':1000,'n_jobs':-1,'max_depth':None},
                X_train, y_train, X_test, y_test  )
   
    preds = train_and_predict( ens.RandomForestClassifier, {'n_estimators':1000,'n_jobs':-1,'max_depth':None},
                train, labels, test )
    
    print preds.shape

    out = csv.writer( open( sys.argv[3], 'wb' ) )
    out.writerow(['ImageId','Label'])
    for i in range(preds.shape[0]):
        out.writerow([i+1,preds[i]])

            
