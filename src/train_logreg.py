

import preprocess as pp
import train_classifier as clf
from autoencoder import LogisticRegression
import numpy as np

if __name__ == "__main__":

    labels, train, test = pp.load_from_csv( 'train.csv', 'test.csv' )
    
    train_batches = train.reshape((420,100,784))

    label_batches  = np.array(labels).reshape((420,100))

    logreg = LogisticRegression( 784, 10 )

    logreg.fit( train_batches , label_batches, nbatches = 420 )
