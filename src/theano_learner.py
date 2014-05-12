from IPython.parallel import Client,TimeoutError
'''
defines superclass for theano deep learning models
@author: Alex Susemihl (2013.09.24)
'''
import numpy

import theano.tensor as T
import theano
from IPython.parallel import Client, TimeoutError, RemoteError
from sklearn.cross_validation import train_test_split

class Learner( object ):

    def __init__( self, **kwargs ):
        self.params = {}
        self.set_params( **kwargs )
   
    def fit( self, train, labels ):
        '''
        fits the training data (supervised or unsupervised,
        depending on the presence of labels) by serial
        gradient descent. Uses shared variables, which should
        default to a GPU via theano if it is present.

        @param train : training data, in batch format
        @param labels: training labels, not necessarily present
        '''
    


    def fit_parallel( self, train, labels=None, training_epochs=100,
                      validate=None, val_labels=None ):
        '''
        fits the training data (supervised or unsupervised,
        depending on the presence of labels) by distributing
        the calculation of the gradients onto the nodes of 
        an IPython.parallel cluster. Additionally it uses
        Le Roux's Stochastic Assynchronous Gradient Descent.
        
        @param train : training data, must be in batch format
        @param labels: training labels, not necessarily present
        
        '''
        d = {}
        grads = {}
        param_names = []
        for i in self.params.keys():
             d[i] = numpy.zeros_like(self.params[i])
             grads[i] = numpy.zeros_like(self.params[i])
             param_names.append(i)
        
        g = self.get_gradients(batches=True)

        try:
            c = Client()
            ncores = len(c.ids)
        except:
            print "Could not load parallel interface."
            print "Defaulting to serial training."
            self.fit( train, labels )    

        # async_grads stores the last calculated
        # gradient for each batch
        
        nbatches = train.shape[0] 
        async_grads = {}        
        for i in self.params.keys():
            # for every parameter of the model
            # async_grads has a list of nbatches gradients
            async_grads[i] = [np.zeros_like(self.params[i])]*nbatches

        def process_result( i ):
            '''
            process_result takes an async_result
            ipython object and tries to get its result
            and update the parameter values according
            '''
            gr = i.get(0.001) # small timeout to cycle through all results
            batch = gr[0]
            for n,i in enumerate(params):
                async_grads[i][batch] = gr[1+n]
            for i in params:
                d[i] = d[i] - learning_rate*numpy.mean(async_grads[i], axis = 0)
            
            eng_id = i.metada['engine_id']
            return eng_id
            
        def place_call( b, eng_id ):
            '''
            place call to cluster engine eng_id to calculate
            the gradient of the cost on batch b
            '''
            if labels is None:
                args = [b, train[b]]
            else:
                args = [b, train[b], labels[b]]

            for i in params:
                args.append( d[i] )
            call = c[eng_id].apply( g, *args )
        
        maxcalls = training_epochs*nbatches
        ncalls = 0
        calls = []
        for i in range(nbatches):
            calls.append( place_call( i, i%ncores ))
            ncalls += 1
        old = 0.0
        while ncalls<=maxcalls:

            for i in calls:
                try:
                    eng_id = process_result( i )
                    calls.append( place_call( ncalls % nbatches,
                                              ncalls % ncores ))
                    ncalls += 1
                except TimeoutError:
                    pass
            if ncalls/nbatches > old:
                old = ncalls/nbatches
                print '''Training Epoch %d, errors %lf'''%\
                        (old,self.errors(validate,val_labels))

        self.set_params( d )

    def errors( self, validate, val_labels, W, b ):
        predict = T.dot(W,validate)+b
        return T.mean( T.sqr( predict-val_labels ) )

    def set_params( self, d ):
        for i in d.keys():
            self.params[i] = d[i]

    def get_gradients( self, batches=False ):
        # returns a theano function taking
        # argumentns with shape given in
        # self.param_shapes and returning
        # a list of gradients with the same shapes
        # this implements linear regression with
        # a square loss for testing
        W = T.dmatrix('W')
        b = T.dvector('b')
        dat = T.dvector('data')
        true = T.dvector('true')
        predict = T.dot(W,dat)+b
        cost = T.mean( T.sqr( predict - true ) )
        grad = T.grad( cost, [W,b] )
        
        if batches:
            index = T.scalar('index')
            inputs = [index,dat,true,W,b]
            outputs = [index].append(grad)
        else:
            inputs = [dat,true,W,b]
            outputs = grad
        grad_function = theano.function(inputs ,outputs)
        return grad_function

    def get_cost( self, batches=False ):
        pass


if __name__=='__main__':
    print __doc__
    print "Debugging Run"
    x_data = numpy.random.rand(100,100)
    y_data = 2*x_data - 0.5
    learner = Learner(w1=numpy.zeros((1,1)), b1=numpy.zeros((1,)))
    learner.fit_parallel( x_data , y_data )
