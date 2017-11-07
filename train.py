import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os


from config import get_config, print_config
from data_loader import Data_loader
from nnet import Model


###################### USE DYNAMIC RNN DECODER WITH LEN2-1
###################### Use Xs, Ys AND Xa, Ya !!! (self.y = [batch size,] = +-1)



# Get running configuration
config, _ = get_config()
print_config()

# reproducibility
np.random.seed(57)
tf.set_random_seed(57)

# Init data_loader
data_loader = Data_loader()
weights, word2idx, idx2word = data_loader.load_w2v(w2v_size=config.w2v_size)
Xs_ids, Ys_ids, Xa_ids, Ya_ids = data_loader.corpus2ids(w2v_size=config.w2v_size, max_words=config.maxlen)

# Build tensorflow graph from config
print("Building graph...")
#tf.reset_default_graph()
model = Model(embedding_weights=weights, config=config)


# Saver to save & restore all the variables.
variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name and 'global_step' not in v.name]
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
    print(len(Xs_ids),'question pairs','(',num_batch_per_epoch,' batch / epoch)')
    for epoch in range(config.training_epochs):
        # Random batch indices
        batches = data_loader.create_batches(len(Xs_ids), batch_size=config.batch_size)

        # train
        if config.alternate_encdec == False:
            switchs = [model.optim]
        else:
            switchs = [model.optim_dec, model.optim_enc] # alternate encoder / decoder training

        for optim in switchs: 
            # train
            for i, idx_batch in tqdm(enumerate(batches)):

                # choose i and f iid in {0,1}. random task: qi -> qf or qf -> qi.
                r0, r1, r2 =  np.random.rand(1), np.random.rand(1), np.random.rand(1) # 2 iid rrv

                # REFORMULATE
                if r1 < config.reformulate_proba: # reformulate q1 -> q2
                    #if r0<0.5:
                    # fetch duplicate questions
                    input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xs_ids, Ys_ids, idx_batch, padlen=config.padlen)
                    #flip = [1.]

                    #else:
                    # fetch not duplicate questions
                    #input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xa_ids, Ya_ids, idx_batch, padlen=config.padlen) #######################
                    #flip = [-1.]                                                                                                                         #######################

                    if r2<0.5: # q1 -> q2
                        feed = {model.q1: input_batch, model.len1: input_length, model.q2: output_batch, model.len2: output_length} #, model.flip: flip}
                    else: # q2 -> q1
                        feed = {model.q1: output_batch, model.len1: output_length, model.q2: input_batch, model.len2: input_length} #, model.flip: flip}


                # REPEAT
                else: # repeat qi -> qi
                    if r0<0.5:
                        # fetch duplicate questions
                        input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xs_ids, Ys_ids, idx_batch, padlen=config.padlen)
                    else:
                        # fetch not duplicate questions
                        input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xa_ids, Ya_ids, idx_batch, padlen=config.padlen)

                    if r2<0.5: # q1 -> q1
                        feed = {model.q1: input_batch, model.len1: input_length, model.q2: input_batch, model.len2: input_length} #, model.flip: [1.]}
                    else: # q2 -> q2
                        feed = {model.q1: output_batch, model.len1: output_length, model.q2: output_batch, model.len2: output_length} #, model.flip: [1.]}

                # Forward pass & train step
                _, loss_, mu_, sigma_, kld_, lb_, summary, global_step_ = sess.run([optim, model.loss, model.mu, model.sigma, model.kld, model.lb, model.merged, model.global_step], feed_dict=feed)

                if i%100 == 0:
                    if r1 >= config.reformulate_proba:
                        print(' Step:',global_step_,'loss=',loss_,'kld=',kld_,'anneal=',1.-lb_,'std=', sigma_, '|mu|2=',mu_,'VAE')
                    else:
                        print(' Step:',global_step_,'loss=',loss_,'kld=',kld_,'anneal=',1.-lb_,'std=', sigma_, '|mu|2=',mu_,'VAD')
                    writer.add_summary(summary, global_step_)

        # Save the variables to disk
        if epoch % max(1,int(config.training_epochs/5)) == 0 and i!=0 :
            if not os.path.exists("save/"+config.save_to):
                os.makedirs("save/"+config.save_to)
            save_path = saver.save(sess,"save/"+config.save_to+"/tmp.ckpt", global_step=i)
            print("\n Model saved in file: %s" % save_path)
            
    print("Training COMPLETED !")
    saver.save(sess,"save/"+config.save_to+"/actor.ckpt")

