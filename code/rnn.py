import math, time
import tensorflow as tf
import numpy as np
import svdrnn
import Params
import sys,os
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


class RNNModel (object):
    def __init__(self, params):
        self.rnn_cell = None
        # feature
        self.x = tf.placeholder("float", [None, params.time_steps, params.input_size])
        # label
        self.y = tf.placeholder("float", [None, params.output_size])
        # train_flag placeholder
        self.train_flag = tf.placeholder(tf.bool, [], name="train_flag")
        # learning rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        self.init_epoch = 0
        print 'Var names: ', self.x.name, self.y.name, self.train_flag.name, self.learning_rate.name

        sys.stdout.flush()
        # set random seed before build the graph
        tf.set_random_seed(params.random_seed)

        # build graph
        logits = self.build(params)

        # prediction
        # Define loss and optimizer
        # evaluation
        if params.regression_flag:
            self.pred = logits
            self.loss_op = tf.reduce_mean(tf.pow(self.pred-self.y, 2))
            self.accuracy = self.loss_op
        else:
            self.pred = tf.nn.softmax(logits)
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.y))
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        config = tf.ConfigProto(device_count={'GPU' : int(params.gpu_flag)})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # running session
        self.session = tf.Session(config=config)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    """
    call this function to destroy globally defined variables in tensorflow
    """
    def close(self):
        self.session.close()
        tf.reset_default_graph()


    def set_cell(self, params):
        if params.cell == "LSTM":
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    num_units=params.num_units
                    )
        elif params.cell == "RNN":
            self.rnn_cell = tf.contrib.rnn.BasicRNNCell(
                    num_units=params.num_units
                    )
        elif params.cell == "svdRNN":
            self.rnn_cell = svdrnn.svdRNNCell(
                    n_h=params.num_units,
                    n_r=params.r_size,
                    r_margin = params.r_margin
                    )
        else:
            assert 0, "unsupported cell %s" % (params.cell)

    def build(self, params):

        self.set_cell(params)
        # last linear layer
        last_w = tf.get_variable("last_w", initializer=tf.truncated_normal([self.rnn_cell.output_size, params.output_size], stddev=0.1))
        last_b = tf.get_variable("last_b", initializer=tf.truncated_normal([params.output_size], stddev=0.1))

        # Unstack to get a list of 'time_steps' tensors of shape (batch_size, n_input)
        # assume time_steps is on axis 1
        x = tf.unstack(self.x, params.time_steps, 1)
        # get RNN cell output
        output, states = tf.contrib.rnn.static_rnn(self.rnn_cell, x, dtype=np.float32)
	    # Apply Dropout
        output = tf.cond(self.train_flag, lambda: tf.nn.dropout(output, params.dropout_keep_rate), lambda: tf.identity(output))
        # linear activation, using rnn inner loop last output
        logits = tf.matmul(output[-1], last_w) + last_b
        print "output[-1].shape = ", output[-1].get_shape()
        print "last_w.shape = ", last_w.get_shape()

        self.vars = tf.trainable_variables()
        self.normalize_vars = [v for v in tf.trainable_variables() if 'Householder' in v.name]

        self.validate_batch_size = params.batch_size * 4
        print "trainable_variables = ", [v.name for v in self.vars]
        print "normalize_variables = ", [v.name for v in self.normalize_vars]

        sys.stdout.flush()
        return logits

    """
    @brief model training
    @param params parameters
    """
    def train(self, params, train_x, train_y, test_x, test_y):
        if params.regression_flag:
            metric = "RMS"
        else:
            metric = "accuracy"
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        normalize_op = [tf.assign(v,tf.nn.l2_normalize(tf.matrix_band_part(v,0,-1),1)) for v in self.normalize_vars]
        #normalize_op = [tf.assign(v,tf.nn.l2_normalize(v,1)) for v in self.normalize_vars]
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        if not params.load_model:
            self.session.run(init)
        else:
            uninitialized_vars = self.get_un_init_vars()
            if len(uninitialized_vars) > 0:
                print "Sth not right, these vars are not loaded: ", [x.name for x in uninitialized_vars]
        # only initialize if not train
        if not params.train_flag:
            print("model not trained")
            return None, None

        print "start trainging! "
        sys.stdout.flush()
        train_error = []
        test_error = []
        iterations = 0
        time_used = 0
        num_batches = math.ceil(len(train_x)/float(params.batch_size))
        for epoch in range(self.init_epoch, params.num_epochs):
            # reduce learning rate by epoch
            learning_rate = params.initial_learning_rate*math.pow(params.lr_decay, int(epoch))
            if epoch == self.init_epoch:
                train_error.append(self.validate(train_x, train_y, batch_size=self.validate_batch_size))
                test_error.append(self.validate(test_x, test_y, batch_size=self.validate_batch_size))
                print("Epoch %d, iterations = %d, time = %.6f, training %s = %.6f, testing %s = %.6f" % (self.init_epoch-1, iterations, time_used,  metric, train_error[-1], metric, test_error[-1]))
                sys.stdout.flush()
                t0 = time.time()
            # permuate batches
            perm = np.random.permutation(len(train_x))

            # run on batches
            batch_index = 0
            for batch_begin in range(0, len(train_x), params.batch_size):
                # get batch x and y
                batch_x = train_x[perm[batch_begin:min(batch_begin+params.batch_size, len(train_x))]]
                batch_y = train_y[perm[batch_begin:min(batch_begin+params.batch_size, len(train_x))]]
                feed_dict = {self.x: batch_x,
                        self.y: batch_y,
                        self.train_flag: True,
                        self.learning_rate: learning_rate}
                # Run optimization op (backprop)
                self.session.run(self.train_op, feed_dict=feed_dict)
                if params.cell=='svdRNN' or params.cell=='oRNN':
                    self.session.run(normalize_op)

                batch_index += 1
                iterations += 1

                # decay the display intervals for speedup
                if batch_index % (num_batches//params.display_epoch_num) == 0:
                    time_used += time.time() - t0
                    train_error.append(self.validate(train_x, train_y, batch_size=self.validate_batch_size))
                    test_error.append(self.validate(test_x, test_y, batch_size=self.validate_batch_size))
                    print("Epoch %.6f, iterations = %s, time = %.6f, training %s = %.6f, testing %s = %.6f, learning rate = %f" %
                            ( self.init_epoch+float(iterations)/num_batches, '{:05}'.format(iterations), time_used,  metric, train_error[-1], metric, test_error[-1], learning_rate))
                    sys.stdout.flush()
                    t0 = time.time()
            # save model
            if params.model_dir and iterations%(5*num_batches)==0:
                if os.path.isdir(os.path.dirname(params.model_dir+'/'+params.dataset)) == False:
                    os.makedirs(params.model_dir+'/'+params.dataset)
                    print 'making dir: '+params.model_dir+'/'+params.dataset
                if params.cell=='svdRNN' or params.cell=='oRNN':
                    self.save("%s/%s/%s.%s.%s.%s.%s" % (params.model_dir,params.dataset,params.cell,params.r_size,params.num_units,"init"+str(params.initial_learning_rate),"epoch"+str(epoch) ))
                else:
                    self.save("%s/%s/%s.%s.%s.%s" % (params.model_dir,params.dataset,params.cell,params.num_units,"init"+str(params.initial_learning_rate),"epoch"+str(epoch) ))
            ## early exit if reach best training
            #if params.regression_flag and train_error[-1] == 0.0:
            #    break
            #elif not params.regression_flag and train_error[-1] == 1.0:
            #    break

            if np.isnan(train_error[-1]) or np.isinf(train_error[-1]) or np.isnan(test_error[-1]) or np.isinf(test_error[-1]):
                print("found nan or inf, stop training")
                break

        print("Optimization Finished!")

        return train_error, test_error

    """
    @brief prediction
    @param params parameters
    """
    def predict(self, x, batch_size=128):
        # Launch the graph
        pred = np.zeros((len(x), self.pred.get_shape().as_list()[1]))
        # run on batches
        for batch_begin in range(0, len(x), batch_size):
            # get batch x and y
            batch_x = x[batch_begin:min(batch_begin+batch_size, len(x))]
            # Run optimization op (backprop)
            pred[batch_begin:min(batch_begin+batch_size, len(x))] = self.session.run(self.pred, feed_dict={self.x: batch_x,
                                           self.train_flag: False})
        return pred

    """
    @brief validate prediction
    @params x feature
    @params y label
    @param batch_size batch size
    @return accuracy
    """
    def validate(self, x, y, batch_size=128):
        # error
        cost = self.accuracy
        # relative error
        #cost = tf.reduce_sum(tf.pow(tf.divide(self.pred-self.y, self.y), 2))
        #cost = tf.reduce_sum(tf.pow((self.pred-self.y)/self.y, 2))
        validate_cost = 0.0
        for batch_begin in range(0, len(x), batch_size):
            # get batch x and y
            batch_x = x[batch_begin:min(batch_begin+batch_size, len(x))]
            batch_y = y[batch_begin:min(batch_begin+batch_size, len(x))]
            feed_dict = {self.x: batch_x,
                    self.y: batch_y,
                    self.train_flag: False}
            # Calculate batch loss and accuracy
            validate_cost += self.session.run(cost, feed_dict=feed_dict)*len(batch_y)
        return validate_cost/len(x)

    """
    @brief save model
    @param filename file name
    """
    def save(self, filename):
        print "save model ", filename

        #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #    print i.name  # list all var to save

        saver = tf.train.Saver()
        saver.save(self.session, filename)

    """
    @brief load model
    @param filename model file name
    """
    def load(self, filename):
        print "load model ", filename

        saver = tf.train.Saver()
        saver.restore(self.session, filename)

        #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #    print i.name  # list all var to save
        # restore variables

        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("Placeholder:0")
        self.y = graph.get_tensor_by_name("Placeholder_1:0")
        self.trainFlag = graph.get_tensor_by_name("train_flag:0")
        self.learningRate = graph.get_tensor_by_name("learning_rate:0")

        self.init_epoch = int(filename.split('epoch')[1])+1

    def get_un_init_vars(self):
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.session.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        return uninitialized_vars
