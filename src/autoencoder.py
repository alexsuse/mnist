
"""
Implements a simple autoencoder class, with a simple training method.
"""
from theano import tensor as T
from theano import shared
import numpy as np
import theano
import cPickle as pic
from theano.printing import Print
import time
import random

class dA:
    """
    The classic autoencoder of yore.
    hidden activation is given by h = logistic(W_vtoh*visible+b_h)
    visible activation is givens by z = logistic(W_htov*hidden+b_v)
    """
    def __init__(self, nvisible, nhidden, data=None, W1=None, bv=None,
                 bh=None, rng_seed=None, regL=None):
        self.nhidden = nhidden
        self.nvisible = nvisible

        #hidden to visible matrix
        if W1 == None:
            Wi = np.asarray(np.random.uniform(
                low=-1. * np.sqrt(10. / (nhidden + nvisible)),
                high=1. * np.sqrt(10. / (nhidden + nvisible)),
                size=(nvisible, nhidden)), dtype=theano.config.floatX)
            W = shared(value=Wi, name='W')
        else:
            W = shared(value=W1, name='W')
        self.W = W
        self.Wprime = W.T

        #biases
        if bh == None:
            bi_h = np.asarray(np.zeros(nhidden), dtype=theano.config.floatX)
            b_h = shared(value=bi_h, name='b_h')
        else:
            b_h = shared(value=bh, name='b_h')
        if bv == None:
            bi_v = np.asarray(np.zeros(nvisible), dtype=theano.config.floatX)
            b_v = shared(value=bi_v, name='b_v')
        else:
            b_v = shared(value=bv, name='b_v')
        self.b_v = b_v
        self.b_h = b_h

        #regularization parameter lambda
        if regL == None:
            self.lamb = None
        else:
            self.lamb = shared(value=regL, name='lamb')
            self.lamb = T.cast(self.lamb, dtype=theano.config.floatX)
        if data:
            self.data = data
        else:
            self.data = T.matrix('data')
        if rng_seed:
            self.rng = np.random.RandomState(rng_seed)
            self.theano_rng = T.shared_randomstreams.RandomStreams(
                                                self.rng.randint(2 ** 30))
        else:
            self.rng = np.random.RandomState(1234)
            self.theano_rng = T.shared_randomstreams.RandomStreams(
                                                self.rng.randint(2 ** 30))


        self.output = T.nnet.sigmoid( T.dot( self.data, self.W) + self.b_h )

        self.params = [self.W, self.b_h, self.b_v]

    def fit(self,train_data,training_epochs=400,corruption = 0.4, l_rate = 5,tol = 1e-4):
        isbatches = len(train_data.shape)==3
        change_tol_low = 1e-3
        change_tol_high = 3e-1
        learning_rate = T.scalar('learning_rate')
        data_x = theano.shared(value = np.asarray(train_data,
                dtype = theano.config.floatX), name = 'data_x')
        cost,updates = self.get_cost_and_updates(corruption,learning_rate)

        if not isbatches:
            inputs = [theano.Param(learning_rate, default =0.1)]
            train_da = theano.function(inputs = inputs,
                    outputs = cost,
                    updates=updates,
                    givens = [(self.data,data_x)],
                    on_unused_input='ignore')
            costs = []
            for epoch in xrange(training_epochs):
                epoch_time = time.clock()
                c = train_da(learning_rate = l_rate)
                costs = [c]+costs
                
#                print('Training epoch %d, cost %.5f,'
#                        'time %dsecs')%(epoch,c,time.clock()-epoch_time)
                costs = costs[:10]

                rel_cost_change = (c-np.mean(costs))/np.abs(c)

                if np.abs(rel_cost_change) <tol and epoch>10:
                    break 

                if np.abs( rel_cost_change ) < change_tol_low and rel_cost_change < 0.0:
                    l_rate = l_rate*1.1
                    print 'adapting learning rate to %lf'%l_rate

                if np.abs(rel_cost_change) >change_tol_high or rel_cost_change > 0.0:
                    l_rate = l_rate*0.85
                    print 'adapting learning rate to %lf'%l_rate

        else:
            index = T.lscalar('index')
            inputs = [index,theano.Param(learning_rate,default=0.1)]
            train_da = theano.function(inputs = inputs,
                    outputs = cost,
                    updates=updates,
                    givens = [(self.data,data_x[index])])
            costs = []
            for epoch in xrange(training_epochs):
                epoch_time = time.clock()
                c = []
                order = range(train_data.shape[0])
                random.shuffle(order)
                for batch in order:
                    c.append(train_da(index = batch, learning_rate = l_rate))
                costs = [np.mean(c)]+costs 
                print('Training epoch %d, cost %.5f,'
                        'time %dsecs')%(epoch,costs[0],time.clock()-epoch_time)
                costs = costs[:10]
                rel_cost_change = (costs[0]-np.mean(costs))/np.abs(costs[0])
                
                if np.abs(rel_cost_change)<tol and epoch >10:
                    break 
                
                if np.abs(rel_cost_change) <change_tol_low and rel_cost_change < 0.0:
                    l_rate = l_rate*1.1
                    print 'adapting learning rate to %lf'%l_rate
                
                if np.abs(np.mean(costs)-costs[0])/costs[0]>change_tol_high or rel_cost_change > 0.0:
                    l_rate = l_rate*0.85
                    print 'adapting learning rate to %lf'%l_rate

    def get_reconstruction_function(self, input_val):
        return T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(input_val, self.W)
                                        + self.b_h), self.Wprime) + self.b_v)

    def transform(self, input_val):
        return T.nnet.sigmoid(T.dot(input_val,self.W)+self.b_h).eval()

    def get_transform( self, input_val ):
        return T.nnet.sigmoid( T.dot( input_val, self.W ) + self.b_h )        
        
    def corrupt_input(self, input_val, corruption_level):
        return self.theano_rng.binomial(size=input_val.shape, n=1,
                                    p=(1 - corruption_level)) * input_val

    def get_cost_and_updates(self, corruptionlevel, learning_rate):

        reconst_x = self.get_reconstruction_function(
                            self.corrupt_input(self.data, corruptionlevel))
        #L = -T.mean(self.data * T.log(reconst_x)
        #           + (1 - self.data) * T.log(1 - reconst_x), axis=1)
        L = T.mean( T.sqr( reconst_x - self.data ) )
        
        cost = T.mean(L)
        if self.lamb != None:
            L += self.lamb * T.mean(T.sqr(self.W))

        gparams = T.grad(cost, self.params)

        updates = []
        for param, gparam in zip(self.params, gparams):
            #updates.append((param, param - learning_rate))
            updates.append((param, param - learning_rate *
                             T.cast(gparam, dtype=theano.config.floatX)))

        return (cost, updates)


class assymetric_dA(dA):
    """
    Assymetric AE, extend from dA
    hidden activation is given by h = logistic(W_vtoh*visible+b_h)
    visible activation is givens by z = logistic(W_htov*hidden+b_v)
    """
    def __init__(self, nvisible, nhidden, data=None, W1=None, W2=None,
                 b_v=None, b_h=None, rng_seed=None, regL=None):
        dA.__init__(self, nvisible, nhidden, data, W1, b_v, b_h,
                    rng_seed, regL)

        #visible to hidden matrix
        Wi_htov = np.asarray(np.random.uniform(
            low=-4 * np.sqrt(6. / (nhidden + nvisible)),
            high=4 * np.sqrt(6. / (nhidden + nvisible)),
            size=(nhidden, nvisible)), dtype=theano.config.floatX)
        Wprime = shared(value=Wi_htov, name='Wprime')
        self.Wprime = Wprime
        self.params.append(self.Wprime)

    def get_cost_and_updates(self, corruptionlevel, learning_rate):

        reconst_x = self.get_reconstruction_function(
                                self.corrupt_input(self.data, corruptionlevel))
        L = -T.sum(self.data * T.log(reconst_x)
                   + (1 - self.data) * T.log(1 - reconst_x), axis=1)
        cost = T.mean(L)
        if self.lamb != None:
            L += self.lamb * (T.mean(T.dot(self.W, self.W))
                              + T.mean(T.dot(self.Wprime, self.Wprime)))

        gparams = T.grad(cost, self.params)

        updates = []
        for param, gparams in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparams))

        return (cost, updates)



class LogisticRegression(object):
    '''
    Simple stupid class for logistic regression
    Simply takes the input, calculates the softmax of the log-linear prediction
    and that's it

    self.W
    '''
    def __init__(self, n_in, n_out,lam=0.01, input=None, W2 = None, b = None):
        if input is None:
            self.data = T.dmatrix('data')
        else:
            self.data = input
        if W2 is None:
            Wi = np.random.uniform( low=-3./n_in, high=3./n_in, size=(n_in, n_out) )
            self.W = theano.shared(value=np.asarray( Wi,
                                    dtype=theano.config.floatX), name='W')
        else:
            self.W = theano.shared(value = np.array(W2, dtype = theano.config.floatX), name = 'W')
        if b is None:
            bi = np.random.uniform( low=-1./n_out, high=1./n_out, size=(n_out,))
            self.b = theano.shared(value=np.asarray( bi,
                                    dtype=theano.config.floatX), name='b')
        else:
            self.b = theano.shared( value = np.array(b, dtype = theano.config.floatX), name = 'b')

        self.p_y_given_x = T.nnet.softmax(T.dot(self.data, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.lam = lam
    
        self.params = [self.W, self.b]

    def negativeLL(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def fit(self, train, label, learning_rate=1, training_epochs = 10000):
        y = T.ivector('y')
        isbatches = len(train.shape) == 3
        train_x = theano.shared(value = np.asarray(train,dtype=theano.config.floatX), name = 'train_x')
        train_y = T.cast(theano.shared(value = np.asarray(label), name = 'train_y'),'int32')
        cost, updates = self.get_cost_and_updates(y,learning_rate)
        
        if not isbatches:
            train_lr = theano.function([], cost,
                    updates = updates,
                    givens = [(y,train_y),(self.data,train_x)],
                    on_unused_input='ignore')
            err_train = self.errors(self.data,y)
            errors_train = theano.function([], err_train,
                    givens = {y:train_y,self.data:train_x})

            for epoch in xrange(training_epochs):
                c = train_lr()
                e = errors_train()
                print "Training epoch %d gave cost %lf, errors %lf"%(epoch,np.mean(c),np.mean(e))
        else:
            nbatches = train.shape[0]
            index = T.lscalar('index')
            train_lr = theano.function([index], cost,
                    updates = updates,
                    givens = [(y,train_y[index]),(self.data,train_x[index])])
            err_train = self.errors(self.data,y)
            errors_train = theano.function([index], err_train,
                    givens= [(y,train_y[index]),(self.data,train_x[index])])

            for epoch in xrange(training_epochs):
                c = []
                e = []
                for batch in range(nbatches):
                    c.append( train_lr(batch))
                    e.append(errors_train(batch))
                print "Training epoch %d gave cost %lf, errors %lf"%(epoch,np.mean(c),np.mean(e))


    def predict(self,x):
        p_y_given_x = self.get_predictions(x)
        return T.argmax( p_y_given_x, axis = 1)

    def get_cost_and_updates(self, y, learning_rate):                                                                                                                             
        cost = self.negativeLL(y)+self.lam*T.mean(self.W*self.W)
        grads = T.grad(cost, self.params)
        updates = []
        for i, p in enumerate(self.params):
            updates.append((p, p - learning_rate * grads[i]))

        return cost, updates

    def errors(self, x, y):
        y_pred = T.argmax(T.nnet.softmax(T.dot(x, self.W) + self.b), axis=1)
        return T.mean(T.neq(y_pred, y))

    def get_predictions(self,x):
        return T.nnet.softmax(T.dot(x,self.W)+self.b)

    def get_cost(self, x, y):
        p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

 
class TwoLayerPerceptron(object):
    def __init__(self, n_vis, n_hidden, n_out,
            W_1_init = None,
            W_2_init = None, 
            b_1_init = None, 
            b_2_init = None, 
            lReg1 = 1e-1,
            lReg2 = 1e-2):

        self.data = T.dmatrix( 'data' )

        self.lReg1 = lReg1
        self.lReg2 = lReg2

        self.firstLayer = dA( n_vis, n_hidden, data = self.data, W1 = W_1_init, bh = b_1_init )
         
        self.secondLayer = LogisticRegression( n_hidden, n_out, input=self.firstLayer.output, W2 = W_2_init, b = b_2_init )

        self.params = [self.firstLayer.W, self.firstLayer.b_h] + self.secondLayer.params

    def get_cost_and_updates(self, y, learning_rate = 1e-1):
        
        cost = self.negativeLL( y )

        cost += self.lReg1*T.mean( T.sqr( self.firstLayer.W ) )\
                +self.lReg2*T.mean( T.sqr(self.secondLayer.W ) ) 

        grad = T.grad( cost, self.params )

        updates = []

        for param, gparam in zip(self.params,grad):
            updates.append( (param, param - learning_rate*gparam ) )

        return (cost, updates)

    def predict( self, x ):
        return self.secondLayer.predict( self.firstLayer.get_transform( x ) )

    def negativeLL( self, y ):
        p_y_given_x = self.secondLayer.get_predictions( self.firstLayer.get_transform( self.data ) )
        return -T.mean( T.log( p_y_given_x )[T.arange(y.shape[0]), y])

    def get_predictions( self, x ):
        return self.secondLayer.get_predictions( self.firstLayer.get_transform( x ) )

    def score( self, x, y ):
        y_pred = self.predict( x )
        return 1.0-T.mean( T.neq( y_pred, y ) ).eval()

    def errors( self, x, y ):
        y_pred = self.predict( x )
        return T.mean(T.neq(y_pred, y))

    def fit_and_validate( self, train, label_train, test, label_test, learning_rate=1e-4, training_epochs = 10000 ):
        """
        fits MLP with two layers, checking the validation error on the test set
        supports batches on training set, but not on test set
        """
        
        y = T.ivector ('y' )

        isbatches = len(train.shape) == 3

        train_x = theano.shared(value = np.asarray(train,dtype=theano.config.floatX), name = 'train_x')
        train_y = T.cast(theano.shared(value = np.asarray(label_train), name = 'train_y'),'int32')
    
        cost, updates = self.get_cost_and_updates(y,learning_rate)

        test_x = theano.shared( value = np.asarray( test, theano.config.floatX), name = 'test_x' )
        test_y = T.cast( theano.shared( value = np.asarray( label_test ), name = 'test_y'), 'int32' )
    
        if not isbatches:

            train_lr = theano.function([], cost,
                    updates = updates,
                    givens = [(y,train_y),(self.data,train_x)],
                    on_unused_input='ignore')
            err_train = self.errors(self.data,y)
            errors_train = theano.function([], err_train,
                    givens = {y:train_y,self.data:train_x})
            errors_test = theano.function([], err_train,
                    givens = {y:test_y,self.data:test_x})

            for epoch in xrange(training_epochs):
                c = train_lr()
                e = errors_train()
                val_e = errors_test()
 #               print "Training epoch %d gave cost %lf, training errors %lf, validation errors %lf"%(epoch,np.mean(c),np.mean(e), np.mean( val_e) )
        else:
            index = T.lscalar('index')
            nbatches = train.shape[0]
            train_lr = theano.function([index], cost,
                    updates = updates,
                    givens = [(y,train_y[index]),(self.data,train_x[index])])
            err_train = self.errors(self.data,y)
            errors_train = theano.function([index], err_train,
                    givens= [(y,train_y[index]),(self.data,train_x[index])])
            errors_test = theano.function([], err_train,
                    givens = [(y,test_y),(self.data,test_x)])
            for epoch in xrange(training_epochs):
                c = []
                e = []
                val_e = []
                for batch in range(nbatches):
                    c.append( train_lr(batch))
                    e.append(errors_train(batch))
                val_e = errors_test()
                
#                print "Training epoch %d gave cost %lf, training errors %lf, validation errors %lf"%(epoch,np.mean(c),np.mean(e), np.mean( val_e) )

    def fit(self, train, label, learning_rate=1e-4, training_epochs = 1000, tolerance=1e-4, nbatches = None):
        y = T.ivector('y')
        if isbatches is not None:
            train = train.reshape( train.shape[0]/nbatches, nbatches, train.shape[1] )
            labels = labels.reshape( labels.shape[0]/nbatches, nbatches )
            isbatches = True
        else:
            isbatches = len(train.shape) == 3
    
        train_x = theano.shared(value = np.asarray(train,dtype=theano.config.floatX), name = 'train_x')
        train_y = T.cast(theano.shared(value = np.asarray(label), name = 'train_y'),'int32')
    
        cost, updates = self.get_cost_and_updates(y,learning_rate)
   
    
        if not isbatches:

            train_lr = theano.function([], cost,
                    updates = updates,
                    givens = [(y,train_y),(self.data,train_x)],
                    on_unused_input='ignore')
            err_train = self.errors(self.data,y)
            errors_train = theano.function([], err_train,
                    givens = {y:train_y,self.data:train_x})
            costs = []
            for epoch in xrange(training_epochs):
                c = train_lr()
                costs = [c] + costs
                costs = costs[:10]
                e = errors_train()
#                print "Training epoch %d gave cost %lf, errors %lf"%(epoch,np.mean(c),np.mean(e))

                if epoch>50 and np.abs( np.mean( costs ) - c )/c < tolerance:
                    break

        else:
            index = T.lscalar('index')
            nbatches = train.shape[0]
            train_lr = theano.function([index], cost,
                    updates = updates,
                    givens = [(y,train_y[index]),(self.data,train_x[index])])
            err_train = self.errors(self.data,y)
            errors_train = theano.function([index], err_train,
                    givens= [(y,train_y[index]),(self.data,train_x[index])])
            costs = []
            for epoch in xrange(training_epochs):
                c = []
                e = []
                for batch in range(nbatches):
                    c.append( train_lr(batch))
                    e.append(errors_train(batch))
                costs = [np.mean( c )] + costs
                costs = costs[:10]
#                print "Training epoch %d gave cost %lf, errors %lf"%(epoch,np.mean(c),np.mean(e))
                if epoch> 50 and np.abs( np.mean(costs) - c.mean() )/c.mean() < tolerance:
                    break

