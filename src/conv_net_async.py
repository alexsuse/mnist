# !/usr/bin/env python

"""
Trains a convnet two-layer on MNIST (Eventually)

arg1 :: path to train data (%s)
arg2 :: path to test data (%s)
arg3 :: path to output file (%s)

"""
import sys
import time
import cPickle as pic
import os

from guppy import hpy

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from IPython.parallel import Client
from IPython.parallel import TimeoutError, RemoteError, EngineError, CompositeError
import IPython.parallel

import autoencoder
import preprocess as pp

rect = lambda x : T.maximum( 0.0, x )

class OneLayerConvNet(object):
    def __init__(self, filter_shape, image_shape, filters_init=None,
                bias_init=None, W2_init=None, b2_init=None,
                fix_filters=True, rng=None):
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

        self.output = rect(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        output_dims = filter_shape[0]*(image_shape[3]-filter_shape[3]+1)*(image_shape[2]-filter_shape[2]+1)

        self.logreg = autoencoder.LogisticRegression( 
                            output_dims, 10, input=self.output.flatten(2),
                            W=W2_init, b=b2_init
                            )

        self.params = [self.W, self.b, self.logreg.W, self.logreg.b]

    def get_params( self ):
        params = {}
        params['W1'] = self.W.get_value()
        params['b1'] = self.b.get_value()
        params['W2'] = self.logreg.W.get_value()
        params['b2'] = self.logreg.b.get_value()
        return params

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
        return rect(conv.conv2d(input_val, self.W, filter_shape=self.filter_shape,
                image_shape=self.image_shape) + self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_data_conv( self, data ):
        return rect(conv.conv2d( data, self.W, filter_shape=self.filter_shape,
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
        
        # check if train data is in batches
        isbatches = len( train.shape ) == 5
        # in case it is, check if labels are accordingly formatted as well
        if isbatches:
            assert labels.shape[0] == train.shape[0]
            assert labels.shape[1] == train.shape[1]
       
        # shared variables for theano training
        train_x = theano.shared( value=np.array( train, dtype=theano.config.floatX ), name='train_x' )
        train_y = T.cast( theano.shared( value=np.array( labels ), name='train_y' ), 'int32' )
        
        # y is the label vector
        y = T.ivector('y')
        # l_rate is the theano object for the learning rate,
        l_rate = T.scalar('l_rate')
        # get the cost and update expressions for training
        cost,updates = self.get_cost_and_updates( y, learning_rate=l_rate )
        err = self.get_errors( self.data,  y )

        # mini-batch sgd
        if isbatches:
            # number of batches
            nbatches = train.shape[0]
            # batch index
            index = T.iscalar('index')
            # inputs are the index of the batch and the learning rate
            inputs = [index,theano.Param(l_rate, default=0.1)]
            # train_conv_net is the theano function that updates the conv_net parameters
            # and outputs the cost
            train_conv_net = theano.function(inputs = inputs, outputs = cost, updates=updates,
                                givens=[(y,train_y[index]),(self.data,train_x[index])] )
            # errors gives the errors on the batch
            errors = theano.function([index], err,
                                givens=[(y,train_y[index]),(self.data,train_x[index])] )
            
            costs = []
            for i in xrange(training_epochs):
                c = []# costs in batch
                e = []# errors in batch
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
                    # after a certain number of iterations, if
                    # the costs is changing too quickly or is increasing,
                    # reduce the learning rate by .85
                    if np.abs(rel_cost_change) > tol_high or rel_cost_change > 0.0:
                        learning_rate = .85*learning_rate

                    # otherwise, if the costs are diminishing too slowly,
                    # increase the learning rate by 1.1
                    if np.abs(rel_cost_change) < tol_low and rel_cost_change < 0.0:
                        learning_rate = 1.1*learning_rate
                    
                    # if costs is changing less than the tolerance, stop, learning is done
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
                    # after a certain number of iterations, if
                    # the costs is changing too quickly or is increasing,
                    # reduce the learning rate by .85
                    if np.abs(rel_cost_change) > tol_high or rel_cost_change > 0.0:
                        learning_rate = .85*learning_rate

                    # otherwise, if the costs are diminishing too slowly,
                    # increase the learning rate by 1.1
                    if np.abs(rel_cost_change) < tol_low and rel_cost_change < 0.0:
                        learning_rate = 1.1*learning_rate
                    
                    # if costs is changing less than the tolerance, stop, learning is done
                    if np.abs(rel_cost_change) < tolerance:
                        break

    def fit_parallel( self, train, labels, 
                        learning_rate=1e-3, 
                        training_epochs=100, 
                        ip_profile='bryonia' ):
        y = T.ivector('y')
        x = T.tensor4('x')            
        index = T.iscalar('index')
    
        # non-shared variables for distributed stuffs
        W1 = T.tensor4('W1')
        b1 = T.dvector('b1')
        W2 = T.dmatrix('W2')
        b2 = T.dvector('b2')

        # numpy holders for the values of the parameters
        np_W1 = self.W.get_value()
        np_b1 = self.b.get_value()
        np_W2 = self.logreg.W.get_value()
        np_b2 = self.logreg.b.get_value()

        hidden = rect( conv.conv2d( x, W1,
                                     image_shape=self.image_shape,
                                     filter_shape=self.filter_shape ) + \
                                     b1.dimshuffle('x',0,'x','x') ).flatten(2)

        rng = T.shared_randomstreams.RandomStreams(
                    np.random.randint(1e8) )
        drop_out = rng.binomial( n=1, p=0.5,size=hidden.shape )
        drop_out_hidden = drop_out*hidden

        probs = T.nnet.softmax( T.dot( hidden, W2 )/2 + b2 )
        drop_out_probs = T.nnet.softmax( T.dot( drop_out_hidden, W2 ) + b2 )

        negLL = -T.mean( T.log( drop_out_probs[ T.arange( y.shape[0] ), y] ) )

        preds = T.argmax( probs, axis = 0 )

        errors = T.mean( T.neq( preds, y ) )

        grs = T.grad( negLL, [W1,b1,W2,b2] )

        outputs = [index,negLL] + grs

        grads = theano.function( [ index, x, y, W1, b1, W2, b2 ], outputs )

        err = theano.function( [x, y, W1, b1, W2, b2], errors )

        nbatches = train.shape[0]
      
        predict = T.argmax( probs, axis=1 )
     
        err = T.mean( T.neq( predict, y ) )
 
        errors = theano.function([x,y,W1,b1,W2,b2],err)
       

        try:
            c = Client(profile=ip_profile)
            dview = c[:]
            with dview.sync_imports():
                import os
            dview.execute("os.environ['MLK_NUM_THREADS']='1'").get()
            print 'loaded load_balanced_view with %d nodes, life is good.'%len(c.ids)
            print 'Continuing with parallel training\n'
        except:
            print 'could not load direct view from ipcluster...'
            print 'falling back on serial training.'
            self.fit(train,labels, learning_rate=learning_rate, training_epochs=training_epochs)
            return
        
        train = np.array( train, dtype=theano.config.floatX)
        labels = np.array( labels, dtype='int32')

        gradients = {}
        
        gradients['W1'] = [np.zeros_like(np_W1)]*nbatches
        gradients['W2'] = [np.zeros_like(np_W2)]*nbatches
        gradients['b1'] = [np.zeros_like(np_b1)]*nbatches
        gradients['b2'] = [np.zeros_like(np_b2)]*nbatches
        d = {'np_W1':np_W1,'np_W2':np_W2,'np_b1':np_b1,'np_b2':np_b2}
      
        calls = []
        err_calls = []
        costs = []
       
        def process_result(i):
            gr = list(i.get(0.0001))
            batch = gr[0]
            eng_id = i.metadata['engine_id']
            gradients['W1'][batch] = gr[2]
            gradients['b1'][batch] = gr[3]
            gradients['W2'][batch] = gr[4]
            gradients['b2'][batch] = gr[5]
            d['np_W1'] = d['np_W1'] - learning_rate*np.mean(gradients['W1'], axis=0)
            d['np_b1'] = d['np_b1'] - learning_rate*np.mean(gradients['b1'], axis=0)
            d['np_W2'] = d['np_W2'] - learning_rate*np.mean(gradients['W2'], axis=0)
            d['np_b2'] = d['np_b2'] - learning_rate*np.mean(gradients['b2'], axis=0)
            costs.append(gr[1])

            if n_calls<training_epochs*nbatches: 
                #c[eng_id].results.clear()
                calls.append(
                        c[eng_id].apply( grads, batch, train[batch], labels[batch],\
                                        d['np_W1'],d['np_b1'],d['np_W2'],d['np_b2'])
                        )
        def print_results( epoch_time, n_calls, nbatches ):
            if n_calls%nbatches == 0 and n_calls/nbatches > 1:
                print 'printing results'
                old_time = epoch_time
                epoch_time = time.time()
                errs = c[:].map( errors,
                                   [train[i] for i in range(train.shape[0])],
                                   [labels[i] for i in range(labels.shape[0])],
                                   [d['np_W1']]*labels.shape[0],
                                   [d['np_b1']]*labels.shape[0],
                                   [d['np_W2']]*labels.shape[0],
                                   [d['np_b2']]*labels.shape[0]).get()
                ti = epoch_time-old_time
                old_time = epoch_time
                epoch_time = time.time()
                print '''Training epoch %d, cost %lf, time %lf, '''%(n_calls/nbatches,np.mean(costs),ti),
                print '''errors %lf, calls waiting %d'''%(np.mean( errs ), len(calls))
                return epoch_time
            return None
            
        def commit_results():
            self.W.set_value( d['np_W1'] )
            self.b.set_value(d['np_b1'] )
            self.logreg.W.set_value(d['np_W2'] )
            self.logreg.b.set_value(d['np_b2'] )
             
        try:
            
            n_calls = 0
            n_cores = len(c.ids)
            for i in range(nbatches):
                core = c.ids[ i % n_cores ]
                calls.append(
                        c[ core ].apply(grads,i,train[i],labels[i],
                                        np_W1,np_b1,np_W2,np_b2)
                        )
                n_calls += 1
            epoch_time= time.time()
            while True:
                new_time = None
                for n,i in enumerate(calls):
                    if i.ready():
                        process_result(i)
                        calls.remove(i)
                        costs = costs[-2*nbatches:]
                        n_calls +=1
                #if n_calls > 2*nbatches:
                new_time = print_results( epoch_time, n_calls, nbatches )
                if new_time is not None:
                    epoch_time = new_time
                    commit_results()
                if n_calls/nbatches > training_epochs:
                    break

        except KeyboardInterrupt:
            print 'You interrupted training! No probs...'
            print 'Setting values of shared variables'
            dview.abort()
            # set self.parameters to learned values
            commit_results()
        except MemoryError:
            hp = hpy().heap()
            with open('debug','w') as fi:
                for i in range(len(hp)):
                    fi.write(str(hp[i]))


        except (RemoteError, EngineError, CompositeError):
            epoch = int(n_calls/nbatches)
            print 'Apparently some Engines died',
            print ' retrying training for remaining %d epochs!'%\
                    ( training_epochs - epoch )
            
            dview.abort()
            dview.purge_results('all')
            dview.results.clear()
            dview.clear()
            # saving intermediate results
            commit_results()
            
            self.fit_parallel( train, labels,
                               learning_rate=learning_rate,
                               training_epochs=training_epochs-epoch )
        
        # set self.parameters to learned values
        commit_results()


if __name__ == '__main__':

    print __doc__%(sys.argv[1],sys.argv[2],sys.argv[3])

    labels, train, test = pp.load_from_csv( sys.argv[1], sys.argv[2] )

    if len(sys.argv)>=4:
        ip_profile = sys.argv[4]
        print 'using profile %s for ipcluster'%ip_profile
    else:
        print 'using bryonia as default ipcluster...'
        ip_profile='bryonia'

    # set values to something useful, batch_size and number of epochs
    # doesn't seem to make much of a difference
    training_epochs = 300
    training_batches = 100

    # patch_size, for 28x28 images, 10x10 patches seemed reasonable
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
        
        # Creates a denoising autoencoder with 500 hidden nodes,
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

    # conv_net parameters
    filter_shape = (n_filters, 1, patch_size, patch_size)
    image_shape = (1000, 1, 28, 28)
    
    # reshape ae matrices for conv.conv2d function
    W_ae = np.reshape(W_ae, (n_filters, 1, patch_size, patch_size))
    b_ae = np.reshape(b_ae, (n_filters,))

    # reshaping inputs for conv_net
    # data_conv = inp.reshape((batch_size, 1, 28, 28))
    # data_val_conv = val_inp.reshape(validation_shape)

    try:
        f = open('temp_pickle_conv_net_params.pkl','rb')
        my_params = pic.load(f)
        conv_net = OneLayerConvNet( filter_shape, image_shape,
                                    filters_init=my_params['W1'],
                                    bias_init=my_params['b1'],
                                    W2_init=my_params['W2'],
                                    b2_init=my_params['b2'])
        print 'Loaded params %s from file %s'%(str(my_params.keys()),f.name)

    except Exception as inst:
        print 'no params backup found'
        conv_net = OneLayerConvNet(filter_shape, image_shape,
                                   filters_init=W_ae, bias_init=b_ae)

    try:
        conv_net.fit_parallel(train.reshape((42,1000,1,28,28)), 
                              labels.reshape((42,1000)),
                              learning_rate=1e-4, training_epochs=100,
                              ip_profile=ip_profile)
        preds = []
        test = test.reshape((test.shape[0]/1000,1000,1,28,28))
        print 'Computing test predictions'
        for n,t in enumerate(test):
            print 'batch %d out of %d'%(n,test.shape[0])
            preds = preds+ list(conv_net.predict( t ))
        pp.print_preds_to_csv( preds, sys.argv[3] )
        print 'Saving parameters to pickle'
        params = conv_net.get_params()
        with open('temp_pickle_conv_net_params.pkl','wb') as f:
            pic.dump(params,f)

    except Exception as e:
        print 'Exception info: ',sys.exc_info()
        print 'OK dumping last params to pickle'
        params = conv_net.get_params()
        with open('temp_pickle_conv_net_params.pkl','wb') as f:
            pic.dump(params,f)

