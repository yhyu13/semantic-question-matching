import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


from config import get_config, print_config
from data_loader import Data_loader
from nnet import Model


# Get running configuration
config, _ = get_config()
print_config()

# reproducibility
np.random.seed(46)
tf.set_random_seed(46)

# Init data_loader
data_loader = Data_loader()
weights, word2idx, idx2word = data_loader.load_w2v(w2v_size=config.w2v_size)
Xs_ids, Ys_ids, Xa_ids, Ya_ids = data_loader.corpus2ids(w2v_size=config.w2v_size, max_words=config.maxlen)

# Build tensorflow graph from config
print("Building graph...")
model = Model(embedding_weights=weights, config=config)


# Saver to save & restore all the variables.
variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name and 'global_step' not in v.name]
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)  

print("Starting session...")
with tf.Session() as sess:

    # run init op
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    if config.restore_model==True:
        saver.restore(sess, "save/"+config.restore_from+"/actor.ckpt")
        print("Model restored.")

    print("Starting testing...")
    autoenc_s, autodec_s = [], []
    autoenc_a, autodec_a = [], []

    num_batch_per_epoch = int(len(Xs_ids)/config.batch_size)
    print(len(Xs_ids),'question pairs','(',num_batch_per_epoch,' batch / epoch)')

    # Random batch indices
    batches = data_loader.create_batches(len(Xs_ids), batch_size=config.batch_size)

    # test: Forward pass
    for i, idx_batch in tqdm(enumerate(batches)):

        # fetch duplicate questions
        input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xs_ids, Ys_ids, idx_batch, padlen=config.padlen)

        # REFORMULATE
        feed = {model.q1: input_batch, model.len1: input_length, model.q2: output_batch, model.len2: output_length} # q1 -> q2
        loss_ = sess.run(model.loss, feed_dict=feed)
        autodec_s.append(loss_)
        
        #feed = {model.q1: output_batch, model.len1: output_length, model.q2: input_batch, model.len2: input_length} # q2 -> q1
        #loss_ = sess.run(model.loss, feed_dict=feed)
        #autodec_s.append(loss_)

        # REPEAT
        feed = {model.q1: input_batch, model.len1: input_length, model.q2: input_batch, model.len2: input_length} # q1 -> q1
        loss_ = sess.run(model.loss, feed_dict=feed)
        autoenc_s.append(loss_)

        #feed = {model.q1: output_batch, model.len1: output_length, model.q2: output_batch, model.len2: output_length} # q2 -> q2
        #loss_ = sess.run(model.loss, feed_dict=feed)
        #autoenc_s.append(loss_)

        ###############################################################################################################
        ###############################################################################################################

        # fetch not duplicate questions
        input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xa_ids, Ya_ids, idx_batch, padlen=config.padlen)

        # REFORMULATE
        feed = {model.q1: input_batch, model.len1: input_length, model.q2: output_batch, model.len2: output_length} # q1 -> q2
        loss_ = sess.run(model.loss, feed_dict=feed)
        autodec_a.append(loss_)

        #feed = {model.q1: output_batch, model.len1: output_length, model.q2: input_batch, model.len2: input_length} # q2 -> q1
        #loss_ = sess.run(model.loss, feed_dict=feed)
        #autodec_a.append(loss_)

        # REPEAT
        feed = {model.q1: input_batch, model.len1: input_length, model.q2: input_batch, model.len2: input_length} # q1 -> q1
        loss_ = sess.run(model.loss, feed_dict=feed)
        autoenc_a.append(loss_)

        #feed = {model.q1: output_batch, model.len1: output_length, model.q2: output_batch, model.len2: output_length} # q2 -> q2
        #loss_ = sess.run(model.loss, feed_dict=feed)
        #autoenc_a.append(loss_)

        

        if i%100 == 0:
            print('S: AutoEnc loss',np.mean(autoenc_s),'AutoDec loss',np.mean(autodec_s),' // A: AutoEnc loss',np.mean(autoenc_a),'AutoDec loss',np.mean(autodec_a))

    print("Testing COMPLETED !")
    # Histogram
    n1, bins1, patches1 = plt.hist(autoenc_s, 100, facecolor='b', alpha=0.75) # q2 = q1
    n2, bins2, patches2 = plt.hist(autoenc_a, 100, facecolor='b', alpha=0.75) # q2 = q1
    n3, bins3, patches3 = plt.hist(autodec_s, 100, facecolor='g', alpha=0.75) # q1 and q2 duplicates
    n4, bins4, patches4 = plt.hist(autodec_a, 100, facecolor='r', alpha=0.75) # q1 and q2 not duplicates
    plt.xlabel('-log_prob(q2|q1)')
    plt.ylabel('Counts')
    plt.title('Generative AutoDecoder')
    plt.axis([0., 15., 0, 1500])
    plt.grid(True)
    plt.show()
