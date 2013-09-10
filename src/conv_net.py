#!/usr/bin/env python

"""
Trains a convnet two-layer on MNIST (Eventually)

arg1 :: path to train data (%s)
arg2 :: path to test data (%s)
arg3 :: path to output file (%s)

"""
import sys
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import time
import autoencoder
import preprocess as pp
import cPickle as pic
import os

from IPython.parallel import Client

class OneLayerConvNet(object):
    def __init__(self, filter_shape, image_shape, filters_init=None,
                bias_init=None, fix_filters=True, rng=None):
        self.fix_filters = fix_filters
        self.data = T.dtensor4('data')
        
        assert image_shape[1] == filter_shape[1]
        
        if filters_init == None:
            fan_in = np.prod(filter_shape[1:])
            W_values = np.asarray(self.rng.uniform(low=-np.sqrt(3. / fan_in),
                                high=np.sqrt(3. / fan_in), size=filter_shape),
                                  dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W')
        else:
            assert filters_init.shape == filter_shape
            self.W = theano.shared(value=np.asarray(filters_init,
                                        dtype=theano.config.floatX), name='W')
        if bias_init == None:
            b_value = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_value, name='b')
        else:
            assert filters_init.shape[0] == bias_init.shape[0]
            self.b = theano.shared(value=np.asarray(bias_init,
                                        dtype=theano.config.floatX), name='b')
        
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        conv_out = conv.conv2d( self.data, self.W, filter_shape=filter_shape,
                               image_shape=image_shape)

        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        output_dims = filter_shape[0]*(image_shape[3]-filter_shape[3]+1)*(image_shape[2]-filter_shape[2]+1)

        self.logreg = autoencoder.LogisticRegression( output_dims, 10, input=self.output.flatten(2) )

        self.params = [self.W, self.b, self.logreg.W, self.logreg.b]

    def negativeLL( self, y, x=None ):
        if x is None:
            x = self.data
        p_y_given_x = self.logreg.get_predictions( self.get_conv( x ).flatten(2) )
        return  -T.mean( T.log(p_y_given_x[ T.arange(y.shape[0]), y ]) )

    def get_errors( self, x, y ):
        return T.mean( T.neq( self.get_predictions( x ), y ) )
    
    def get_predictions( self, x ):
        return self.logreg.predict( self.get_conv( x ).flatten(2) )
    
    def predict( self, x ):
        return self.get_predictions( x ).eval()

    def get_conv(self, input_val ):
        return T.tanh(conv.conv2d(input_val, self.W, filter_shape=self.filter_shape,
                image_shape=self.image_shape) + self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_data_conv( self, data ):
        return T.tanh(conv.conv2d( data, self.W, filter_shape=self.filter_shape,
                image_shape=data.shape.eval() ) + self.b.dimshuffle('x', 0, 'x', 'x'))
        

    def get_grads( self, x, y ):

        cost = self.negativeLL( y, x )

        grad = T.grad( cost, self.params )

        grads = []

        for p,g in zip(self.params,grad):
            grads.append((p,g))

        return grads

    def get_cost_and_updates( self, y, x= None, learning_rate=1e-1 ):
        
        if x is None:
            x = self.data
        cost = self.negativeLL( y, x ) 

        grad = T.grad( cost, self.params )
        
        updates = []

        for p,g in zip(self.params, grad ):
            updates.append( ( p, p-learning_rate*g ) )

        return cost, updates

    def pretrain_logreg( self, train, labels, image_shape ):

        data = np.array([self.get_conv( t ).flatten(2).eval() for t in train])
        
        self.logreg.fit_parallel( data, labels, learning_rate=3e-3, training_epochs=100 )

    def fit( self, train, labels, learning_rate=1e-1, training_epochs=1000 ):
        
        #check if train data is in batches
        isbatches = len( train.shape ) == 5
        #in case it is, check if labels are accordingly formatted as well
        if isbatches:
            assert labels.shape[0] == train.shape[0]
            assert labels.shape[1] == train.shape[1]
       
        #shared variables for theano training
        train_x = theano.shared( value=np.array( train, dtype=theano.config.floatX ), name='train_x' )
        train_y = T.cast( theano.shared( value=np.array( labels ), name='train_y' ), 'int32' )
        
        #y is the label vector
        y = T.ivector('y')
        #l_rate is the theano object for the learning rate,
        l_rate = T.scalar('l_rate')
        #get the cost and update expressions for training
        cost,updates = self.get_cost_and_updates( y, learning_rate=l_rate )
        err = self.get_errors( self.data,  y )

        #mini-batch sgd
        if isbatches:
            #number of batches
            nbatches = train.shape[0]
            #batch index
            index = T.iscalar('index')
            #inputs are the index of the batch and the learning rate
            inputs = [index,theano.Param(l_rate, default=0.1)]
            #train_conv_net is the theano function that updates the conv_net parameters
            #and outputs the cost
            train_conv_net = theano.function(inputs = inputs, outputs = cost, updates=updates,
                                givens=[(y,train_y[index]),(self.data,train_x[index])] )
            #errors gives the errors on the batch
            errors = theano.function([index], err,
                                givens=[(y,train_y[index]),(self.data,train_x[index])] )
            
            costs = []
            for i in xrange(training_epochs):
                c = []#costs in batch
                e = []#errors in batch
                epoch_time = time.time()
                
                for j in xrange( nbatches ):
                    batch_time = time.time()                
                    c.append( train_conv_net(j, l_rate = learning_rate) )
                    e.append( errors(j) )
                    print 'batch %d, cost %lf, errors %lf, time %lf'%(j,c[-1],e[-1],time.time()-batch_time)
                    
                print "Training epoch %d, cost %lf, errors %lf, time %lf"\
                        %(i,np.mean(c),np.mean(e),time.time()-epoch_time) 
               
                costs = [np.mean(c)] + costs
                costs = costs[:10]
                rel_cost_change = (costs[0]-np.mean(costs))/np.abs(costs[0])

                if i > 15:
                    #after a certain number of iterations, if
                    #the costs is changing too quickly or is increasing,
                    #reduce the learning rate by .85
                    if np.abs(rel_cost_change) > tol_high or rel_cost_change > 0.0:
                        learning_rate = .85*learning_rate

                    #otherwise, if the costs are diminishing too slowly,
                    #increase the learning rate by 1.1
                    if np.abs(rel_cost_change) < tol_low and rel_cost_change < 0.0:
                        learning_rate = 1.1*learning_rate
                    
                    #if costs is changing less than the tolerance, stop, learning is done
                    if np.abs(rel_cost_change) < tolerance:
                        break

        else:
            inputs = [theano.Param(l_rate, default=0.1)]
            train_conv_net = theano.function(inputs = inputs, outputs = cost, updates=updates,
                                givens = [(y,train_y),(self.data,train_x)])
            errors = theano.function([], err,
                                givens = [(y,train_y),(self.data,train_x)])
            
            costs = []
            for i in xrange(training_epochs):
                epoch_time=time.time()
                c = train_conv_net(l_rate = learning_rate)
                e = errors()
                print "Training epoch %d, cost %lf, errors %lf, time %lf"\
                        %(i,np.mean(c),np.mean(e),time.time()-epoch_time)    
               
                costs = [np.mean(c)] + costs
                costs = costs[:10]
                rel_cost_change = (costs[0]-np.mean(costs))/np.abs(costs[0])

                if i > 15:
                    #after a certain number of iterations, if
                    #the costs is changing too quickly or is increasing,
                    #reduce the learning rate by .85
                    if np.abs(rel_cost_change) > tol_high or rel_cost_change > 0.0:
                        learning_rate = .85*learning_rate

                    #otherwise, if the costs are diminishing too slowly,
                    #increase the learning rate by 1.1
                    if np.abs(rel_cost_change) < tol_low and rel_cost_change < 0.0:
                        learning_rate = 1.1*learning_rate
                    
                    #if costs is changing less than the tolerance, stop, learning is done
                    if np.abs(rel_cost_change) < tolerance:
                        break


    def fit_parallel( self, train, labels, learning_rate=1e-3 ):
        y = T.ivector('y')
        x = T.tensor4('x')            

        nbatches = train.shape[0]
      
        train_x = theano.shared( value = np.array( train, dtype=theano.config.floatX ),
                                name = 'train_y')
        train_y = theano.shared( value = np.array( labels, dtype='int32' ),
                                name = 'train_y')
        index = T.iscalar('index')

        err = self.get_errors(x,y)
 
        errors = theano.function([index],err,givens=[(x,train_x[index]),(y,train_y[index])])
       
        grads= self.get_grads(x,y)


        outputs = []
        
        for p,g in grads:
            outputs.append(g)
        grad = theano.function([index],outputs = outputs,
                                givens=[(x,train_x[index]),(y,train_y[index])])

        try:
            c = Client()
            dview = c[:]
            print 'loaded direct view, all is good.\n Continuing with parallel training'
        except:
            print 'could not load direct view from ipcluster...'
            print 'falling back on serial training.'
            self.fit(train,labels)
            return
        
        dview.push({'grad':grad,'errors':errors})

        for epoch in xrange(training_epochs):
           
            epoch_time = time.time()
            
                
            calls = dview.map_async( grad, range(nbatches) )

            err_call = dview.map_async( errors, range(nbatches) )

            for gr in calls:
                for i,p in enumerate(self.params):
                    p.set_value( p.get_value() - learning_rate* gr[i] )
            
            print 'Epoch %d, errors %lf, time %lf'%(epoch,
                                                    np.mean(err_call.get(),axis=0),
                                                    time.time()-epoch_time)

if __name__ == '__main__':

    print __doc__%(sys.argv[1],sys.argv[2],sys.argv[3])

    labels, train, test = pp.load_from_csv( sys.argv[1], sys.argv[2] )

    #set values to something useful, batch_size and number of epochs
    # doesn't seem to make much of a difference
    training_epochs = 300
    training_batches = 100

    #patch_size, for 28x28 images, 10x10 patches seemed reasonable
    patch_size = 15

    batch_size = int(train.shape[0] / training_batches)

    n_filters = 500

    output_file = ('ae_' + str(patch_size) + 'x' + str(patch_size) + '_'
                   + str(n_filters) + '_filters_backup.pikle')


    # if the autoencoder hasn't been trained and pickled beforehand,
    # train it and back it up now
    if output_file not in os.listdir('.'):
        '''
        AUTOENCODER TRAINING STARTS HERE
        '''
        patches = pp.make_patches( train, patch_size = patch_size)
        print "generated patches"
        
        #Creates a denoising autoencoder with 500 hidden nodes,
        # could be changed as well
        a = autoencoder.dA( patch_size**2, n_filters)
        a.fit(patches[:10000], training_epochs = 1000, verbose=True)

        W_ae = np.reshape(a.W.get_value(), (n_filters, 1,
                                            patch_size, patch_size))
        b_ae = np.reshape(a.b_h.get_value(), (n_filters,))

        fi = open(output_file, 'w')
        pic.dump([W_ae, b_ae], fi)
        fi.close()
        '''
        AUTOENCODER TRAINING ENDS HERE
        '''

    else:
        # if autoencoder has been trained and backed up in the file named,
        # load it up from there
        fi = open(output_file, 'r')
        [W_ae, b_ae] = pic.load(fi)
        fi.close()

    #conv_net parameters
    filter_shape = (n_filters, 1, patch_size, patch_size)
    image_shape = (1000, 1, 28, 28)
    
    #reshape ae matrices for conv.conv2d function
    W_ae = np.reshape(W_ae, (n_filters, 1, patch_size, patch_size))
    b_ae = np.reshape(b_ae, (n_filters,))

    #reshaping inputs for conv_net
    #data_conv = inp.reshape((batch_size, 1, 28, 28))
    #data_val_conv = val_inp.reshape(validation_shape)

    #cbuild conv_net
    conv_net = OneLayerConvNet(filter_shape, image_shape,
                               filters_init=W_ae, bias_init=b_ae)
    #conv_net.image_shape = (500,1,28,28)
    #conv_net.pretrain_logreg( train[:4000].reshape((8,500,1,28,28)), np.array(labels[:4000]).reshape((8,500)), (4,500,1,28,28) )
    #conv_net.fit( train.reshape((42,1000,1,28,28)), labels.reshape((42,1000)), learning_rate=1e-4, training_epochs=50 )
    conv_net.image_shape = (1000,1,28,28)
    conv_net.fit_parallel(train.reshape((42,1000,1,28,28)), labels.reshape((42,1000)))

    pp.print_preds_to_csv( preds, sys.argv[3] )
