import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os


from config import get_config, print_config
from data_loader import Data_loader
from nnet import Model

# Get running configuration
config, _ = get_config()
print_config()

# reproducibility
np.random.seed(0)
tf.set_random_seed(0)

# Init data_loader
data_loader = Data_loader()
weights, word2idx, idx2word = data_loader.load_w2v()
Xs_ids, Ys_ids, Xa_ids, Ya_ids = data_loader.corpus2ids()

# Build tensorflow graph from config
print("Building graph...")
tf.reset_default_graph()
model = Model(embedding_weights=weights, config=config)

# Saver to save & restore all the variables.
variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)  

print("Starting session...")
with tf.Session() as sess:

    # run init op
    sess.run(tf.global_variables_initializer())

    # print variables
    #variables_names = [v.name for v in tf.global_variables() if 'Adam' not in v.name]
    #values = sess.run(variables_names)
    #for k, v in zip(variables_names, values):
    #    print("Variable: ", k, "Shape: ", v.shape)

    # Restore variables from disk.
    if config.restore_model==True:
        saver.restore(sess, "save/"+config.restore_from+"/actor.ckpt")
        print("Model restored.")

    # Summary writer
    writer = tf.summary.FileWriter('summary/'+config.log_dir, sess.graph)
    print("Starting training...")

    num_batch_per_epoch = int(len(Xs_ids)/config.batch_size)
    for epoch in range(config.training_epochs):
        # Random batch indices
        batches = data_loader.create_batches(len(Xs_ids), batch_size=config.batch_size)

        # train
        for switch in range(0, 2):
            if switch == 0:
                optim = model.optim_dec
                print('updating decoder')
            else:
                optim = model.optim_enc
                print('updating encoder')

            # train
            for i, idx_batch in tqdm(enumerate(batches)):
                # fetch duplicate questions
                input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xs_ids, Ys_ids, idx_batch, padlen=config.padlen)

                # choose i and f iid in {0,1}. random task: qi -> qf or qf -> qi.
                r1, r2 = np.random.rand(1), np.random.rand(1) # two iid rrv
                if r1 < config.reformulate_proba: # reformulate q1 -> q2
                    if r2<0.5:
                        feed = {model.q1: input_batch, model.len1: input_length, model.q2: output_batch, model.len2: output_length}
                    else:
                        feed = {model.q1: output_batch, model.len1: output_length, model.q2: input_batch, model.len2: input_length}
                else: # repeat q1 -> q2
                    if r2<0.5:
                        feed = {model.q1: input_batch, model.len1: input_length, model.q2: input_batch, model.len2: input_length}
                    else:
                        feed = {model.q1: output_batch, model.len1: output_length, model.q2: output_batch, model.len2: output_length}

                # Forward pass & train step
                _, loss_, kld_, summary = sess.run([optim, model.loss, model.kld, model.merged], feed_dict=feed)
                if i%100 == 0:
                    print(loss_,kld_)
                    writer.add_summary(summary,i+(epoch+switch)*num_batch_per_epoch)

        # Save the variables to disk
        if epoch % max(1,int(config.training_epochs/5)) == 0 and i!=0 :
            if not os.path.exists("save/"+config.save_to):
                os.makedirs("save/"+config.save_to)
            save_path = saver.save(sess,"save/"+config.save_to+"/tmp.ckpt", global_step=i)
            print("\n Model saved in file: %s" % save_path)
            
    print("Training COMPLETED !")
    saver.save(sess,"save/"+config.save_to+"/actor.ckpt")

