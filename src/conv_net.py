
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import time
from autoencoder import dA
import load_data as ld
import cPickle as pic
import os


class OneLayerConvNet(object):
    def __init__(self, input_val, filter_shape, image_shape, filters_init=None,
                bias_init=None, fix_filters=True, rng=None):
        self.fix_filters = fix_filters
        if rng == None:
            self.rng = np.random.RandomState(23455)
        else:
            self.rng = rng
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

        conv_out = conv.conv2d(input_val, self.W, filter_shape=filter_shape,
                               image_shape=image_shape)
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

    def get_output(self, input_val, filter_shape, image_shape):
        return T.tanh(conv.conv2d(input_val, self.W, filter_shape=filter_shape,
                image_shape=image_shape) + self.b.dimshuffle('x', 0, 'x', 'x'))


class LogisticRegression(object):
    '''
    Simple stupid class for logistic regression
    Simply takes the input, calculates the softmax of the log-linear prediction
    and that's it

    self.W
    '''
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=np.zeros((n_out,),
                                    dtype=theano.config.floatX), name='b')

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negativeLL(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def get_cost_and_updates(self, y, learning_rate):
        cost = self.negativeLL(y)
        grads = T.grad(cost, self.params)
        updates = []
        for i, p in enumerate(self.params):
            updates.append((p, p - learning_rate * grads[i]))

        return cost, updates

    def errors(self, x, y):
        y_pred = T.argmax(T.nnet.softmax(T.dot(x, self.W) + self.b), axis=1)
        return T.mean(T.neq(y_pred, y))

    def get_cost(self, x, y):
        p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

if __name__ == '__main__':
    #Mnist has 70000 examples, we use 50000 for training,
    # set 20000 aside for validation
    train_size = 50000
    train_data, validation_data = ld.load_data_mnist(train_size=train_size)

    #taking only a subset of the data,
    # as my laptop will burn to the ground otherwise
    #COMMENT THIS AWAY FOR FULL DATASET
    #train_data['images'] = train_data['images'][:20000,:,:,:]
    #validation_data['images'] = validation_data['images'][:5000,:,:,:]
    #train_data['labels'] = train_data['labels'][:20000]
    #validation_data['labels'] = validation_data['labels'][:5000]
    #ALL OF IT UP TO HERE

    #Random backup name, change at will, it will get a pickle of the
    # autoencoder weight and bias

    #set values to something useful, batch_size and number of epochs
    # doesn't seem to make much of a difference
    training_epochs = 300
    training_batches = 100

    #patch_size, for 28x28 images, 10x10 patches seemed reasonable
    patch_size = 10

    batch_size = int(train_data['images'].shape[0] / training_batches)
    print 'Training Epochs %d, Batch Size %d, Training Batches %d' % (
                            training_epochs, batch_size, training_batches)
    # n_filters is 30 so I can run it on my laptop,
    # reasonable values should be around 100, 200 at least.
    n_filters = 200

    output_file = ('ae_' + str(patch_size) + 'x' + str(patch_size) + '_'
                   + str(n_filters) + '_filters_backup.pikle')

    #conv_net parameters
    filter_shape = (n_filters, 1, patch_size, patch_size)
    image_shape = (batch_size, 1, 28, 28)
    validation_shape = (validation_data['images'].shape[0], 1, 28, 28)

    #theano batch index
    index = T.lscalar()

    # if the autoencoder hasn't been trained and pickled beforehand,
    # train it and back it up now
    if output_file not in os.listdir('.'):
        '''
        AUTOENCODER TRAINING STARTS HERE
        '''
        batches = ld.make_vector_patches(train_data, training_batches,
                                         batch_size, patch_size)
        validation_images = ld.make_vector_patches(validation_data, 1,
                                validation_data['images'].shape[0], patch_size)
        #batches,ys = ld.make_vector_patches(train_data,
        #            training_batches,batch_size,patch_size)
        #validation_images,validation_ys = ld.make_vector_batches(
        #        validation_data,1,validation_data['images'].shape[0])

        x = T.matrix('x')
        #Creates a denoising autoencoder with 500 hidden nodes,
        # could be changed as well
        a = dA(patch_size * patch_size, n_filters, data=x, regL=0.05)
        #sEt theano shared variables for the train and validation data
        data_x = theano.shared(value=np.asarray(batches,
                                    dtype=theano.config.floatX), name='data_x')
        validation_x = theano.shared(
                            value=np.asarray(validation_images[0, :, :],
                                             dtype=theano.config.floatX),
                            name='validation_x')
        #get cost and update functions for the autoencoder
        cost, updates = a.get_cost_and_updates(0.4, 0.02)
        #train_da returns the current cost and updates the dA parameters
        # index gives the batch index.
        train_da = theano.function([index], cost, updates=updates,
                                   givens=[(x, data_x[index])],
                                   on_unused_input='ignore')
        #validation_error just returns the cost on the validation set
        validation_error = theano.function([], cost,
                                           givens=[(x, validation_x)],
                                           on_unused_input='ignore')
        #loop over training epochs
        for epoch in xrange(training_epochs):
            epoch_time = time.clock()
            c = []
            ve = validation_error()
            #loop over batches
            for batch in xrange(training_batches):

                #collect costs for this batch
                c.append(train_da(batch))

            # print mean training cost in this epoch and
            # final validation cost for checking
            print ('Training epoch %d, cost %.5f,'
                   ' validation cost %.5f, time %dsecs') % (epoch, np.mean(c),
                                                ve, time.clock() - epoch_time)

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

    #reshape ae matrices for conv.conv2d function
    W_ae = np.reshape(W_ae, (n_filters, 1, patch_size, patch_size))
    b_ae = np.reshape(b_ae, (n_filters,))

    #make vector batches, put in proper sizes
    t_x, t_y = ld.make_vector_batches(train_data, training_batches, batch_size)
    v_x, v_y = ld.make_vector_batches(validation_data, 1,
                                      validation_data['images'].shape[0])

    #theano shared variables for log_reg_training
    train_x = theano.shared(value=t_x, name='train_x')
    train_y = theano.shared(value=np.asarray(t_y, dtype='int'), name='train_y')

    #theano shared variables for log_reg_testing
    validation_x = theano.shared(value=v_x[0, :, :], name='validation_x')
    validation_y = theano.shared(value=np.asarray(v_y[0, :], dtype='int'),
                                 name='validation_y')

    #function inputs for cost and validation cost
    inp = T.matrix('inp')
    val_inp = T.matrix('val_inp')
    y = T.ivector('y')
    val_y = T.ivector('val_y')

    #reshaping inputs for conv_net
    data_conv = inp.reshape((batch_size, 1, 28, 28))
    data_val_conv = val_inp.reshape(validation_shape)

    #cbuild conv_net
    conv_net = OneLayerConvNet(data_conv, filter_shape, image_shape,
                               filters_init=W_ae, bias_init=b_ae)

    #log_reg inputs are conv_net_outputs but flattened
    log_reg_input = conv_net.output.flatten(2)
    log_reg_validation_input = conv_net.get_output(data_val_conv, filter_shape,
                                                validation_shape).flatten(2)

    #dimension of log_reginput
    n_in = n_filters * (28 - patch_size + 1) ** 2
    #10 classes
    n_out = 10

    #creat logistic regression object
    log_reg = LogisticRegression(log_reg_input, n_in, n_out)

    #cost and update functions for log_reg with labels and learning rate
    cost, updates = log_reg.get_cost_and_updates(y, 0.01)

    #val_cost gives the validation cost of log_reg
    val_cost = log_reg.get_cost(log_reg_validation_input, val_y)

    #theano functions for training and testing of log_reg
    train_lr = theano.function([index], cost,
            updates=updates,
            givens=[(y, train_y[index]), (inp, train_x[index])])
    validation_lr = theano.function([], val_cost,
            givens=[(val_y, validation_y), (val_inp, validation_x)])

    #errors on train and test set for lr
    err_train = log_reg.errors(log_reg_input, y)
    err_test = log_reg.errors(log_reg_validation_input, val_y)

    #theano functions for the errors
    errors_train = theano.function([index], err_train,
                        givens=[(y, train_y[index]), (inp, train_x[index])])
    errors_test = theano.function([], err_test,
                    givens=[(val_y, validation_y), (val_inp, validation_x)])

    print '\n\n------------\n'
    print 'Now training Logistic Regression with ConvNet'
    print '\n------------\n'

    for epoch in xrange(training_epochs):
        c = []  # accumulates costs
        train_es = []  # accumulates errors
        for batch in xrange(training_batches):
            c.append(train_lr(batch))
        for batch in xrange(training_batches):
            train_es.append(errors_train(batch))
        #test errors and validation
        test_e = errors_test()
        ve = validation_lr()
        train_e = np.mean(train_es)

        print ('LR training epoch %d, train cost %lf, training error %lf,'
               'validation cost %lf, validation error %lf') % (epoch,
                                            np.mean(c), train_e, ve, test_e)
    print ('---->\n.....Dumping logistic regression parameters'
           'to log_reg_params.pickle')

    fi = open("log_reg_params.pickle", 'w')
    pic.dump([log_reg.W.get_value(), log_reg.b.get_value()], fi)
    fi.close()
