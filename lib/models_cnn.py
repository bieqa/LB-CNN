"""
Original code is "models_gvcnn.py" from [1] (https://bieqa.github.io/deeplearning.html),
where some functions was modified from "models.py" in [2] (https://github.com/mdeff/cnn_graph)

[1] Liu, C., Ji, H. and Qiu, A. Convolutional Neural Network on Semi-Regular 
Triangulated Meshes and its Application to Brain Image Data. arXiv preprint 
arXiv:1903.08828, 2019.

[2] Defferrard, M., Bresson, X. and Vandergheynst, P. "Convolutional neural 
networks on graphs with fast localized spectral filtering." 
In Advances in neural information processing systems, pp. 3844-3852. 2016.


Sep. 12, 2020  Modified by Shih-Gu Huang,  where main changes are list below: 
    1. modifly "lmax" in chebyshev() for LB-operator and unnormalization ;
    2. add new functions laguerre() and hermite() into class cgcnn()  ;     
    3. output varaible "logits" from evaluate() ;
    4. remove unused functions .
"""

"""
Copyright (C) 2019,
CFA Lab and Dept. of Biomedical Engineering, National University of Singapore. 
All rights reserved.
"""

#from . import graph
from . import graph_LB as graph
from . import helper_func as hf

import tensorflow as tf
import pandas as pd
from datetime import datetime
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil






# Common methods for all models

class base_model(object):
    
    def __init__(self):
        self.regularizers = []
    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        logits = np.empty([size,np.unique(labels).size])        #20200912
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                if labels.ndim == 1:
                    batch_labels = np.zeros(self.batch_size)
                    batch_labels[:end-begin] = labels[begin:end]
                    feed_dict[self.ph_labels] = batch_labels
#                    batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                    batch_pred, batch_loss, batch_logits = sess.run([self.op_prediction, self.op_loss, self.op_logits], feed_dict) # output logits #20200912               
                    
                else:
                    batch_labels = np.zeros((self.batch_size, labels.shape[1]))
                    batch_labels[:end-begin,:] = labels[begin:end,:]
                    feed_dict[self.ph_labels] = batch_labels
#                    batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                    batch_pred, batch_loss, batch_logits = sess.run([self.op_prediction, self.op_loss, self.op_logits], feed_dict) # output logits #20200912     
#                    batch_pred = tf.argmax(batch_pred, axis=1)

                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
            logits[begin:end,:] = batch_logits[:end-begin,:]        #20200912
             
        if labels is not None:
#            return predictions, loss * self.batch_size / size       
            return predictions, loss * self.batch_size / size, logits   #20200912
        else:
            return predictions
        
    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
#        t_process, t_wall = time.process_time(), time.time() # version Python 3
        t_process, t_wall = time.clock(), time.time() # version Python 2
#        predictions, loss = self.predict(data, labels, sess)
        predictions, loss, logits = self.predict(data, labels, sess)     
        if labels.ndim > 1:
            labels = np.copy(labels[:,1])
        ncorrects = sum(predictions == labels)
        prevalence = np.sum(labels) / labels.shape[0]
        accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean = self.top_k_error(labels, predictions, prevalence, 1)
        accuracy = 100 * accuracy
        f1 = 100 * f1
        
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
#            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall) # version Python 3
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.clock()-t_process, time.time()-t_wall) # version Python 2
#        return string, loss, accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean, predictions # predictions are the predicted labels       #20200912     
        return string, loss, accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean, predictions, logits          


    def fit(self, train_data, train_labels, val_data, val_labels, learning_rate, num_epochs, decay_steps, finetune=False, finetune_fc=False):
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction=0.9 # set the portion of memory used
        sess = tf.Session(graph=self.graph, config=config)
        
        sess.run(self.op_init)
        
        ckpt = tf.train.get_checkpoint_state(self._get_path('checkpoints'))
#        
        # Load the latest model
        if finetune:
            # Restore from check point
            shutil.copytree(self._get_path('checkpoints'), self._get_path('checkpoints_orig')) # keep a copy of the pretrained model
            path = os.path.join(self._get_path('checkpoints'), 'model')
            self.op_saver.restore(sess, ckpt.model_checkpoint_path)
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            self.decay_steps = decay_steps
            self.finetune_fc = finetune_fc
        else:
            shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
            shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
            os.makedirs(self._get_path('checkpoints'))
            path = os.path.join(self._get_path('checkpoints'), 'model')
            self.finetune_fc = finetune_fc
       
        # Start the queue runners
        tf.train.start_queue_runners(sess = sess)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)

        # Training.
        max_gmean = 0.
        
        step_list = []
        train_accuracy_list = []
        train_fscore_list = []
        train_sensitivity_list = []
        train_specificity_list = []
        train_precision_list = []
        train_ppv_list = []
        train_npv_list = []
        train_gmean_list = []
        
        val_loss_list = []
        val_accuracy_list = []
        val_fscore_list = []
        val_sensitivity_list = []
        val_specificity_list = []
        val_precision_list = []
        val_ppv_list = []
        val_npv_list = []
        val_gmean_list = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if self.random_batch_sampling_train==False:
                batch_data, batch_labels = hf.generate_train_batch(train_data, train_labels, self.batch_size, 0.5, random_batch_sampling_train=False)
            else:
                if len(indices) < self.batch_size:
                    indices.extend(np.random.permutation(train_data.shape[0]))
                idx = [indices.popleft() for i in range(self.batch_size)]
                batch_data, batch_labels = train_data[idx,:], train_labels[idx]
            
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
                
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout, self.ph_lr: self.learning_rate} # new learning rate for pretraining
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
            
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps or step == 1:
                epoch = step * self.batch_size / train_data.shape[0]
                start_time = time.time()

                # Check performance of the training set
#                string, loss_train, acc_train, fscore_train, sensitivity_train, specificity_train, precision_train, ppv_train, npv_train, gmean_train, predictions = self.evaluate(train_data, train_labels, sess)            
                string, loss_train, acc_train, fscore_train, sensitivity_train, specificity_train, precision_train, ppv_train, npv_train, gmean_train, predictions,logits = self.evaluate(train_data, train_labels, sess)    #20200912
                step_list.append(step)
                train_accuracy_list.append(acc_train)
                train_fscore_list.append(fscore_train)
                train_sensitivity_list.append(sensitivity_train)
                train_specificity_list.append(specificity_train)
                train_precision_list.append(precision_train)
                train_ppv_list.append(ppv_train)
                train_npv_list.append(npv_train)
                train_gmean_list.append(gmean_train)
                
                duration = time.time() - start_time
                
                train_summ = tf.Summary()
                train_summ.value.add(tag='train_accuracy', simple_value=acc_train)
                train_summ.value.add(tag='train_fscore', simple_value=fscore_train)
                train_summ.value.add(tag='train_sensitivity', simple_value=sensitivity_train)
                train_summ.value.add(tag='train_specificity', simple_value=specificity_train)
                train_summ.value.add(tag='train_precision', simple_value=precision_train)
                train_summ.value.add(tag='train_ppv', simple_value=ppv_train)
                train_summ.value.add(tag='train_npv', simple_value=npv_train)
                train_summ.value.add(tag='train_gmean', simple_value=gmean_train)
                writer.add_summary(train_summ, step)
                writer.flush()
                

                # Check performance of the validation set
#                string, loss_valid, acc_valid, fscore_valid, sensitivity_valid, specificity_valid, precision_valid, ppv_valid, npv_valid, gmean_valid, predictions = self.evaluate(val_data, val_labels, sess)   
                string, loss_valid, acc_valid, fscore_valid, sensitivity_valid, specificity_valid, precision_valid, ppv_valid, npv_valid, gmean_valid, predictions,logits = self.evaluate(val_data, val_labels, sess)   #20200912
                val_loss_list.append(loss_valid)
                val_accuracy_list.append(acc_valid)
                val_fscore_list.append(fscore_valid)
                val_sensitivity_list.append(sensitivity_valid)
                val_specificity_list.append(specificity_valid)
                val_precision_list.append(precision_valid)
                val_ppv_list.append(ppv_valid)
                val_npv_list.append(npv_valid)
                val_gmean_list.append(gmean_valid)
      
                
                valid_summ = tf.Summary()
                valid_summ.value.add(tag='valid_accuracy', simple_value=acc_valid)
                valid_summ.value.add(tag='valid_fscore', simple_value=fscore_valid)
                valid_summ.value.add(tag='valid_sensitivity', simple_value=sensitivity_valid)
                valid_summ.value.add(tag='valid_specificity', simple_value=specificity_valid)
                valid_summ.value.add(tag='valid_precision', simple_value=precision_valid)
                valid_summ.value.add(tag='valid_ppv', simple_value=ppv_valid)
                valid_summ.value.add(tag='valid_npv', simple_value=npv_valid)
                valid_summ.value.add(tag='valid_gmean', simple_value=gmean_valid)
                writer.add_summary(valid_summ, step)
                writer.flush()
      
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                if batch_labels.ndim == 1:
                    prevalence = np.sum(batch_labels)/batch_labels.shape[0]
                else:
                    prevalence = np.sum(batch_labels[:,1]>0.5)/(1.*batch_labels.shape[0])
      
                print('%s: ' % datetime.now())
                print('step: {} / {} (epoch: {:.2f} / {}):'.format(step, num_steps, int(epoch), self.num_epochs))
                print('learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                format_str = ('%.1f examples/sec; %.3f ' 'sec/batch')
                print(format_str % (examples_per_sec, sec_per_batch))
                print('Prevalence = %.4f' % prevalence)
                print('Train loss = %.4f' % loss_train)
                print('Train top1 accuracy = %.4f' % acc_train)
                print('Train top1 fscore = %.4f' % fscore_train)
                print('Train top1 sensitivity = %.4f' % sensitivity_train)
                print('Train top1 specificity = %.4f' % specificity_train)
                print('Train top1 precision = %.4f' % precision_train)
                print('Train top1 ppv = %.4f' % ppv_train)
                print('Train top1 npv = %.4f' % npv_train)
                print('Train top1 gmean = %.4f' % gmean_train)
                print('*******')
                print('Validation loss = %.4f' % loss_valid)
                print('Validation top1 accuracy = %.4f' % acc_valid)
                print('Validation top1 fscore = %.4f' % fscore_valid)
                print('Validation top1 sensitivity = %.4f' % sensitivity_valid)
                print('Validation top1 specificity = %.4f' % specificity_valid)
                print('Validation top1 precision = %.4f' % precision_valid)
                print('Validation top1 ppv = %.4f' % ppv_valid)
                print('Validation top1 npv = %.4f' % npv_valid)
                print('Validation top1 gmean = %.4f' % gmean_valid)
                print('----------------------------')
      
                # Save the training and validation performance for every step
                df = pd.DataFrame(data={'step':step_list, 'train_accuracy':train_accuracy_list, 'train_fscore':train_fscore_list, 'train_sensitivity':train_sensitivity_list, 
                      'train_specificity':train_specificity_list, 'train_precision':train_precision_list, 'train_ppv':train_ppv_list, 'train_npv':train_npv_list,
                      'validation_accuracy': val_accuracy_list, 'validation_fscore': val_fscore_list, 'validation_sensitivity':val_sensitivity_list, 'validation_specificity':val_specificity_list, 
                      'validation_precision':val_precision_list, 'validation_ppv':val_ppv_list, 'validation_npv':val_npv_list, 'validation_gmean':val_gmean_list})
#                df.to_csv(self._get_path('') + 'performance.csv')
                df.to_csv(self._get_path('') + 'performance_trainvalid.csv')       #20200912
    
            # Save the model (checkpoint) with the largest adjusted gmean (for future evaluation) 
#            string, loss_valid, acc_valid, fscore_valid, sensitivity_valid, specificity_valid, precision_valid, ppv_valid, npv_valid, gmean_valid, predictions = self.evaluate(val_data, val_labels, sess)
            string, loss_valid, acc_valid, fscore_valid, sensitivity_valid, specificity_valid, precision_valid, ppv_valid, npv_valid, gmean_valid, predictions,logits = self.evaluate(val_data, val_labels, sess)   #20200912
            gmean_valid_new = gmean_valid - np.divide(np.power(sensitivity_valid - specificity_valid, 2), 2.)
            if gmean_valid_new > max_gmean:    
                max_gmean = gmean_valid_new
                self.op_saver.save(sess, path, write_state=True, global_step=step)

        print('Validation accuracy: Peak = {:.2f}, Mean = {:.2f}'.format(max(val_accuracy_list), np.mean(val_accuracy_list[-10:])))
        print('Validation geometric accuracy: Peak = {:.2f}, Mean = {:.2f}'.format(max(val_gmean_list), np.mean(val_gmean_list[-10:])))
        writer.close()
        sess.close()
        
        return val_loss_list, val_accuracy_list, val_fscore_list, val_sensitivity_list, val_specificity_list, val_precision_list, val_ppv_list, val_npv_list, val_gmean_list


    def top_k_error(self, labels, predictions, prevalence, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        cm = sklearn.metrics.confusion_matrix(labels, predictions)
        
        if np.size(cm)==1:
            cm = np.zeros([2, 2])
            if predictions[0]==1.0:
                cm[0][0] = 0.0
                cm[0][1] = 0.0
                cm[1][0] = 0.0
                cm[1][1] = np.size(predictions)
            else:    
                cm[0][0] = np.size(predictions)
                cm[0][1] = 0.0
                cm[1][0] = 0.0
                cm[1][1] = 0.0
    
        # error 
        nume_err = np.add(cm[1][0], cm[0][1])
        den_err1 = np.add(cm[0][0], cm[1][1])
        den_err2 = np.add(cm[1][0], cm[0][1])
        deno_err = np.add(den_err1, den_err2)
        if nume_err == 0:
            error = 0.
        else:
            error = np.divide(nume_err, 1.*deno_err)
        acc = 1. - error
        
        # sensitivity
        deno_sen = np.add(cm[1][0], cm[1][1])
        if deno_sen == 0:
            sensitivity = 0.
        else:
            sensitivity = np.divide(cm[1][1], 1.*deno_sen) 
            
        # specificity
        deno_spe = np.add(cm[0][0], cm[0][1])
        if deno_spe == 0:
            specificity = 0.
        else:
            specificity = np.divide(cm[0][0], 1.*deno_spe)
        
        # precision
        deno_pre = np.add(cm[1][1], cm[0][1])
        if deno_pre == 0:
            precision = 0.
        else:
            precision = np.divide(cm[1][1], 1.*deno_pre)
    
        # fscore
        nume = np.multiply(sensitivity, precision)
        if nume == 0.:
            fscore = 0.
        else:
            fscore = 2. * np.divide( np.multiply(sensitivity, precision), 1.*np.add(sensitivity, precision) )
    
        # ppv
        nume_ppv = np.multiply(sensitivity, prevalence)
        deno_ppv = np.multiply(1.0-specificity, 1.0-prevalence)
        if np.add(nume_ppv, deno_ppv) == 0:
            ppv = 0.
        else:
            ppv = np.divide(nume_ppv, np.add(nume_ppv, 1.*deno_ppv)) 
    
        # npv
        nume_npv = np.multiply(specificity, 1.0-prevalence)
        deno_npv = np.multiply(1.0-sensitivity, prevalence)
        if np.add(nume_npv, deno_npv) == 0:
            npv = 0.
        else:
            npv = np.divide(nume_npv, np.add(nume_npv, 1.*deno_npv)) 
        
        # G-mean (geometric mean) = squared root of (sensitivity x specificity)
        gmean = np.sqrt(np.multiply(sensitivity, specificity))
        
        return acc, fscore, sensitivity, specificity, precision, ppv, npv, gmean

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.   
    def build_graph(self, M_0, C):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0), 'data')
                if C == 1:
                    self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                else:    
                    self.ph_labels = tf.placeholder(tf.int32, (self.batch_size, C), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.ph_lr = tf.placeholder(tf.float32, (), 'learning_rate')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_logits = op_logits     #20200912
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.ph_lr,
                    self.decay_steps, self.decay_rate, self.momentum, self.finetune_fc) # changed for finetuning with different learning rate
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        
        self.graph.finalize()
    
    def inference(self, data, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, dropout)
#        logits = tf.Print(logits, [logits], message="Logits: ")
        return logits
    
    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                print("Shape of labels", labels.get_shape().as_list())
                print("Shape of logits", logits.get_shape().as_list())
                if labels.get_shape().as_list() == logits.get_shape().as_list():
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                    cross_entropy = tf.Print(cross_entropy, [cross_entropy], message="Cross entropy: ")
                else:
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
                cross_entropy = tf.Print(cross_entropy, [cross_entropy], message="Mean cross entropy: ")
#                print('Cross entropy: ', cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
 
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9, finetune_fc=False):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
                              
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            
            if finetune_fc == True:
                fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')
                logits_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'logits')
                train_vars = []
                train_vars.append(fc_vars)
                train_vars.append(logits_vars)
                print(train_vars)
            else:
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                print(train_vars)

            grads = optimizer.compute_gradients(loss, var_list=train_vars) 
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train


    def _get_path(self, folder):
        return os.path.join(self.dir_name, folder)
    

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            print('Checkpoint filename: ', filename)
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var


class cgcnn(base_model):
    """
    Graph-CNN and LB-CNN which uses the Chebyshev/Laguerre/Hermite approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of convolutional layers.
        F: Number of features.
        K: Polynomial orders.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of FC layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, including chebyshev, Laguerre and Hermite.
        brelu: bias and RELU, including b1relu or b2relu.
        pool: pooling strategy, including apool1 and mpool1.
    
    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout in FC layers: probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for save directory (including summaries and model parameters).
    """
    def __init__(self, L, C, F, K, p, M, filter='chebyshev', normalized=False, algo='LB',
                 brelu='b1relu', pool='mpool1', finetune=False, finetune_fc=False,
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=100, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                dir_name='', random_batch_sampling_train=False, decay_factor=0.1):
#        super().__init__() # version Python 3
        super(cgcnn, self).__init__() # version Python 2
        
        print('Size L: ', len(L))
        print('Size F: ', len(F))
        print('Size K: ', len(K))
        print('Size p: ', len(p))
        
        # Verify the consistency w.r.t. the number of layers.
        assert len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
                
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L
        
        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i-1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
        
        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.finetune = finetune
        self.finetune_fc = finetune_fc
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
#        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.algo, self.normalized = algo , normalized     #20200912        

        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.random_batch_sampling_train = random_batch_sampling_train

        self.decay_rate, self.momentum = decay_rate, momentum
        self.decay_factor = decay_factor
        
        if type(decay_steps) is list == True:
            self.decay_steps = decay_steps
        else:
            self.decay_steps = decay_steps
        
        print ('decay_steps = ', self.decay_steps)
        print ('dir_name = ', self.dir_name)
        
        # Build the computational graph.
        self.build_graph(M_0, C)
        

    # Chebyshev polynomial approximation
    def chebyshev(self, x, L, Fout, K , normalized=False, algo='LB'):
        '''normalized or not,  algo='LB' or 'gL' (graph Laplacian)
        will affact the value of "lmax" (maximum eigenvalue)'''
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)  
        lmax=graph.lmax(L, normalized, algo)   # 202000912
#        L = graph.rescale_L(L, lmax=2)            
        L = graph.rescale_L(L, lmax)        # 202000912
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(1, K-1):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout


    # Laguerre polynomial approximation
    def laguerre(self, x, L, Fout, K , normalized=False, algo='LB'):
        '''normalized or not,  algo='LB' or 'gL' (graph Laplacian)
        will affact the value of "lmax" (maximum eigenvalue)'''        
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        lmax=graph.lmax(L, normalized, algo)
#        L = graph.rescale_L(L, lmax=2)          
        scale = lmax/2.0          
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Laguerre basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = -tf.sparse_tensor_dense_matmul(L, x0)/scale + x0
            x = concat(x, x1)
        for k in range(1, K-1):
            x2 = -tf.sparse_tensor_dense_matmul(L, x1)/scale + x1*(2*k+1) - x0*k  # M x Fin*N
            x2 = x2/(k+1.0)
            x = concat(x, x2)
            x0, x1 = x1, x2
            
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout
 

    # Hermite polynomial approximation
    def hermite(self, x, L, Fout, K , normalized=False, algo='LB'):
        '''normalized or not,  algo='LB' or 'gL' (graph Laplacian)
        will affact the value of "lmax" (maximum eigenvalue)'''    
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        lmax=graph.lmax(L, normalized, algo)
#        L = graph.rescale_L(L, lmax=2)          
        scale= lmax / 2.0               
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Hermite basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)/scale*np.sqrt(2)
            x = concat(x, x1)
        for k in range(1, K-1):
            x2 = tf.sparse_tensor_dense_matmul(L, x1)/scale*np.sqrt(2) - x0*np.sqrt(k)  # M x Fin*N
            x2 = x2/np.sqrt(k+1)
            x = concat(x, x2)
            x0, x1 = x1, x2
            
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout



    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, dropout):
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
#                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                    x = self.filter(x, self.L[i], self.F[i], self.K[i], self.normalized, self.algo) #20200912              
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])
        
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)
        
        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return x
