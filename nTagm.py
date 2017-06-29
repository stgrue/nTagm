import time
import tensorflow as tf
import numpy as np
import argparse

from sys import argv
from random import shuffle

from yaml_parser import Options
from load_data import SequenceBatchContainer, create_lookup_matrix, load_mappings, load_sequences, compute_dimensionality

class LSTMTagger:
    def __init__(self, session, model):
        self.name = model.name
        self.model = model
        self.session = session

        self.input_size = [compute_dimensionality(inp) for inp in model.inputs]
        self.output_size = [compute_dimensionality(outp) for outp in model.outputs]
        
        with tf.variable_scope(self.name):
            # Define inputs and targets
            self.inputs = list()
            self.targets = list()
            self.flat_targets = list()
            for i in range(len(model.inputs)):
                self.inputs.append(tf.placeholder(tf.int32, shape=(None,None))) # Shape: batch_size x sequence_length
            for i in range(len(model.outputs)):
                self.targets.append(tf.placeholder(tf.int32, shape=(None,None))) # Shape: batch_size x sequence_length
                self.flat_targets.append(tf.reshape(self.targets[i], (-1,))) # Shape: (batch_size * sequence_length)


            # Define lookup tables (one-hot or embeddings)
            self.lookup_matrices = self.create_lookup_matrices(model.inputs)


            # Calculate RNN inputs
            self.input_vectors = list() # Shapes: batch_size x sequence_length x input dimensionality (varies)
            for i in range(len(self.inputs)):
                self.input_vectors.append(tf.gather(self.lookup_matrices[i], self.inputs[i]))
            self.rnn_input = tf.concat(self.input_vectors, 2) # Shape: batch_size x sequence_length x (sum of input dimensionalities)


            # Create forward RNN
            self.fw_cell = list()
            self.fw_input_dropout = list()
            self.fw_output_dropout = list()
            self.fw_state_dropout = list()
            for i in range(len(model.rnn.forward_layers)):
                self.fw_input_dropout.append(tf.placeholder(shape=(), dtype=tf.float32))
                self.fw_output_dropout.append(tf.placeholder(shape=(), dtype=tf.float32))
                self.fw_state_dropout.append(tf.placeholder(shape=(), dtype=tf.float32))
                cell = None
                if model.rnn.forward_layers[i].cell_class == "BasicRNNCell":
                    cell = tf.contrib.rnn.BasicRNNCell(model.rnn.forward_layers[i].num_units)
                elif model.rnn.forward_layers[i].cell_class == "LSTMCell":
                    cell = tf.contrib.rnn.LSTMCell(model.rnn.forward_layers[i].num_units)
                elif model.rnn.forward_layers[i].cell_class == "GRUCell":
                    cell = tf.contrib.rnn.GRUCell(model.rnn.forward_layers[i].num_units)
                else:
                    raise Exception("'{}' is not a supported cell class".format(model.rnn.forward_layers[i].cell_class))                  
                cell_with_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.fw_input_dropout[i], output_keep_prob=self.fw_output_dropout[i], state_keep_prob=self.fw_state_dropout[i])
                self.fw_cell.append(cell_with_dropout)
            self.fw_rnn = tf.contrib.rnn.MultiRNNCell(self.fw_cell)

   
            # Create backward RNN (optional)
            if model.rnn.bidirectional:
                self.bw_cell = list()
                self.bw_input_dropout = list()
                self.bw_output_dropout = list()
                self.bw_state_dropout = list()
                for i in range(len(model.rnn.backward_layers)):
                    self.bw_input_dropout.append(tf.placeholder(shape=(), dtype=tf.float32))
                    self.bw_output_dropout.append(tf.placeholder(shape=(), dtype=tf.float32))
                    self.bw_state_dropout.append(tf.placeholder(shape=(), dtype=tf.float32))
                    cell = None
                    if model.rnn.backward_layers[i].cell_class == "BasicRNNCell":
                        cell = tf.contrib.rnn.BasicRNNCell(model.rnn.backward_layers[i].num_units)
                    elif model.rnn.backward_layers[i].cell_class == "LSTMCell":
                        cell = tf.contrib.rnn.LSTMCell(model.rnn.backward_layers[i].num_units)
                    elif model.rnn.backward_layers[i].cell_class == "GRUCell":
                        cell = tf.contrib.rnn.GRUCell(model.rnn.backward_layers[i].num_units)
                    else:
                        raise Exception("'{}' is not a supported cell class".format(model.rnn.backward_layers[i].cell_class))
                    cell_with_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.bw_input_dropout[i], output_keep_prob=self.bw_output_dropout[i], state_keep_prob=self.bw_state_dropout[i])
                    self.bw_cell.append(cell_with_dropout)
                self.bw_rnn = tf.contrib.rnn.MultiRNNCell(self.bw_cell)


            # Compute "raw" RNN outputs
            if not model.rnn.bidirectional:
                self.rnn_output, _ = tf.nn.dynamic_rnn(self.fw_rnn, self.rnn_input, dtype=tf.float32) # Shape: batch_size x sequence_length x size of last RNN layer
                self.flat_rnn_output = tf.reshape(self.rnn_output, (-1, model.rnn.forward_layers[-1].num_units)) # Shape: (batch_size * sequence_length) x size of last (forward) RNN layer
            else:
                (self.fw_rnn_output, self.bw_rnn_output), _ = tf.nn.bidirectional_dynamic_rnn(self.fw_rnn, self.bw_rnn, self.rnn_input, dtype=tf.float32)
                self.rnn_output = tf.concat([self.fw_rnn_output, self.bw_rnn_output], 2)
                self.flat_rnn_output = tf.reshape(self.rnn_output, (-1, model.rnn.forward_layers[-1].num_units + model.rnn.backward_layers[-1].num_units)) # Shape: (batch_size * sequence_length) x sum of sizes of last layers


            # Create biases, and weights that map from raw RNN outputs to dimensionality of each task   
            self.output_bias = list()    
            self.output_weights = list()
            for i in range(len(model.outputs)):
                bias = tf.Variable(tf.random_uniform((self.output_size[i],), minval=-0.1, maxval=0.1), dtype=tf.float32)
                self.output_bias.append(bias)
                if model.rnn.bidirectional:
                    weights = tf.Variable(tf.random_uniform((model.rnn.forward_layers[-1].num_units + model.rnn.backward_layers[-1].num_units, self.output_size[i]), minval=-0.1, maxval=0.1), dtype=tf.float32)
                else:    
                    weights = tf.Variable(tf.random_uniform((model.rnn.forward_layers[-1].num_units, self.output_size[i]), minval=-0.1, maxval=0.1), dtype=tf.float32)
                self.output_weights.append(weights)


            # Compute actual outputs for each task 
            self.flat_output = list()
            self.structured_output = list()
            for i in range(len(model.outputs)):
                self.flat_output.append(tf.matmul(self.flat_rnn_output, self.output_weights[i]) + self.output_bias[i])
                self.structured_output.append(tf.reshape(self.flat_output[i], (tf.shape(self.inputs[0])[0], tf.shape(self.inputs[0])[1], self.output_size[i])))


            # Loss, optimization & training
            # TODO: Different optimizers
            self.loss = list()
            for i in range(len(model.outputs)):
                self.loss.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.flat_output[i], labels=self.flat_targets[i])))
            self.overall_loss = sum(self.loss)

            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(model.optimizer.learning_rate, self.global_step, model.optimizer.decay_step, model.optimizer.decay_factor, staircase=True)
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.overall_loss, global_step=self.global_step)

    def create_lookup_matrices(self, inputs):
        lookup_matrices = list()
        for inp in inputs:
            if inp.learn_lookups:
                lookup_matrix = tf.constant(create_lookup_matrix(inp), dtype=tf.float32)
            else:
                lookup_matrix = tf.Variable(create_lookup_matrix(inp), dtype=tf.float32)                    
            lookup_matrices.append(lookup_matrix)
        return lookup_matrices


    def run_many(self, sequence_batches, training, shuffle_batches=True):
        # WARNING: This code does not yet handle batching properly
        # -> When batching, prevent paddings from inflating correct prediction counts
                
        if shuffle_batches:
            shuffle(sequence_batches)

        sum_loss = [0 for _ in range(len(self.model.outputs))]
        correct_predictions = [0 for _ in range(len(self.model.outputs))]
        num_predictions = 0
        
        for seq_batch in sequence_batches:
            outputs, loss = self.run(seq_batch, evaluation=True, training=training)
            sum_loss = list(map(sum, zip(sum_loss, loss)))
            for i in range(len(outputs)):
                # TODO: Put counting of correct predictions in its own function
                predictions_batch = [list(map(np.argmax, seq)) for seq in outputs[i]]
                gold_batch = seq_batch.output_batches[i]
                assert len(predictions_batch) == len(gold_batch)
                assert all(len(predictions_batch[x]) == len(gold_batch[x]) for x in range(len(predictions_batch)))
                for j in range(len(predictions_batch)):
                    for k in range(len(predictions_batch[j])):
                        if predictions_batch[j][k] == gold_batch[j][k]:
                            correct_predictions[i] += 1 
            num_predictions += sum(len(seq) for seq in outputs[0])

        accuracy = [corr_pr / num_predictions for corr_pr in correct_predictions]
        return sum_loss, accuracy


    def run(self, sequence_batch, evaluation, training):
        '''Run the network on a given batch of sequences, calculating outputs
           and loss, and training the network if desired'''
        assert not (training and not evaluation) # Training implies evaluation
        
        feed_dict = dict()
        for i in range(len(sequence_batch.input_batches)):
            feed_dict[self.inputs[i]] = sequence_batch.input_batches[i]
        if evaluation:    
            for i in range(len(sequence_batch.output_batches)):
                feed_dict[self.targets[i]] = sequence_batch.output_batches[i]
        for i in range(len(self.model.rnn.forward_layers)):
            if training:
                feed_dict[self.fw_input_dropout[i]] = self.model.rnn.forward_layers[i].dropout_input_keep_prob
                feed_dict[self.fw_output_dropout[i]] = self.model.rnn.forward_layers[i].dropout_output_keep_prob
                feed_dict[self.fw_state_dropout[i]] = self.model.rnn.forward_layers[i].dropout_state_keep_prob
                if self.model.rnn.bidirectional:
                    feed_dict[self.bw_input_dropout[i]] = self.model.rnn.backward_layers[i].dropout_input_keep_prob
                    feed_dict[self.bw_output_dropout[i]] = self.model.rnn.backward_layers[i].dropout_output_keep_prob
                    feed_dict[self.bw_state_dropout[i]] = self.model.rnn.backward_layers[i].dropout_state_keep_prob
            else:
                feed_dict[self.fw_input_dropout[i]] = 1.0
                feed_dict[self.fw_output_dropout[i]] = 1.0
                feed_dict[self.fw_state_dropout[i]] = 1.0
                if self.model.rnn.bidirectional:
                    feed_dict[self.bw_input_dropout[i]] = 1.0
                    feed_dict[self.bw_output_dropout[i]] = 1.0
                    feed_dict[self.bw_state_dropout[i]] = 1.0
                    
            
        if training:
            outputs, loss, _ = sess.run([self.structured_output, self.loss, self.train_op], feed_dict=feed_dict)
            return outputs, loss
        else:
            if evaluation:
                outputs, loss = sess.run([self.structured_output, self.loss], feed_dict=feed_dict)
                return outputs, loss
            else:
                outputs = sess.run(self.structured_output, feed_dict=feed_dict)
                return outputs


    def predict_sequence(self, inputs, use_labels=None):
        '''Tag a sequence based on the given inputs, using the model'''
        if use_labels is not None:
            input_labels_to_ix, _, _, output_ix_to_labels = use_labels
            for i in range(len(inputs)):
                inputs[i] = [[input_labels_to_ix[i][lbl] for lbl in inputs[i]]]
        else:
            for i in range(len(inputs)):
                inputs[i] = [inputs[i]]
        
        dummy_batch = SequenceBatchContainer(inputs, []) # Batches of size 1 and no outputs
        outputs = self.run(dummy_batch, evaluation=False, training=False)
        predictions = list()
        for i in range(len(self.model.outputs)):
            curr_preds = [np.argmax(output_vec) for output_vec in outputs[i][0]]
            if use_labels is not None:
                curr_preds = [output_ix_to_labels[i][ix] for ix in curr_preds]
            predictions.append(curr_preds)  
        return predictions


def compute_accuracy(path, data_source):
    num_constraints = 0
    num_correct = 0
    with open(path + "tokens.txt") as tokens_file:
        with open(path + "tags.txt") as tags_file:
            with open(path + constraint_type + ".txt") as constraints_file:
                for line in tokens_file:
                    tokens = line.strip().split("\t")
                    tags = next(tags_file).strip().split("\t")
                    gold_constraints = next(constraints_file).strip().split("\t")
                    gold_constraints = [int(constr) for constr in gold_constraints]
                    predicted_constraints = net.constrain_sequence(tokens, tags)
                    assert len(tokens) == len(tags) == len(gold_constraints) == len(predicted_constraints)
                    num_constraints += len(gold_constraints)
                    for (predicted_constraint, gold_constraint) in zip(predicted_constraints, gold_constraints):
                        if predicted_constraint == gold_constraint:
                            num_correct += 1

    return num_correct / num_constraints


if __name__ == "__main__":
    options = Options("config001.yaml")
    
    # Load data
    mappings = input_labels_to_ix, input_ix_to_labels, output_labels_to_ix, output_ix_to_labels = load_mappings(options.model)
    training_batches = load_sequences(options.training_data, mappings)
    development_batches = load_sequences(options.development_data, mappings) # Once batch size is properly implemented, this should always be loaded with batch size 1

    # Set up network
    sess = tf.InteractiveSession()
    net = LSTMTagger(sess, options.model)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for epoch in range(10):
        starttime = time.time()
        print("Epoch {} started.".format(epoch+1))
        loss, acc = net.run_many(training_batches, training=True)
        print("Epoch {} finished. ({:.1f})".format(epoch, time.time()-starttime))

        print("Evaluation on training set:")
        print("Loss:", loss)
        print("Accuracy:", acc)
        print()
        
        loss, acc = net.run_many(development_batches, training=False)
        print("Evaluation on development set:")
        print("Loss:", loss)
        print("Accuracy:", acc)
        print()

    saver.save(sess, "./model/tagging_model")

    # Optional: Load model
    # (Adjust dropout first and disable initialization of global variables!)
    # saver.restore(sess, "./model/tagging-model")


