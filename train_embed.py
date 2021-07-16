#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
import pickle
import random
from src.gbn_3 import gbn_3_model
from tensorflow.core.protobuf import rewriter_config_pb2
import src.data_process as dp


import src.model as model, src.sample as sample, src.encoder as encoder
from src.load_dataset import load_dataset, load_raw_dataset, Sampler
from src.accumulate import AccumulatingOptimizer
import src.memory_saving_gradients as memory_saving_gradients

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


## setting 
dataset = 'ptb'  ## 'ptb', 'wikitext-2', 'wikitext-103'
run_name = 'topic_embedding'



CHECKPOINT_DIR = './checkpoint'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
SAMPLE_DIR = 'samples'
datapath = './data/' + dataset + '/'
pretrain_gpt_path = './models'
checkpoint_path = CHECKPOINT_DIR + '/' + dataset + '/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
phi_save = dataset
phi_save_check = pretrain_gpt_path + '/' + phi_save
if not os.path.exists(checkpoint_path+'/'+run_name):
    os.makedirs(checkpoint_path+'/'+run_name)
if not os.path.exists(phi_save_check):
    os.makedirs(phi_save_check)





parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', metavar='PATH', type=str, default=datapath+'train.npz', help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='124M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd>.')
parser.add_argument('--noise', type=float, default=0.0, help='Add noise to input training data to regularize against typos.')

parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')

parser.add_argument('--restore_from', type=str, default='fresh', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='layer3_phi_theta', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=1000, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=5, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=500, help='Write a checkpoint every N steps')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default=datapath+'test.npz', help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=1, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=40, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=100, help='Calculate validation loss every STEPS steps.')
parser.add_argument('--valid_sequence_len', type=int, default=1024, help='valid_sequence_len.')
parser.add_argument('--train_sequence_len', type=int, default=1024, help='train_sequence_len.')
parser.add_argument('--test_sequence_len', type=int, default=1024, help='train_sequence_len.')

parser.add_argument('--test_raw_dataset', metavar='PATH', type=str, default=datapath+'test.txt', help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--test_every', metavar='STEPS', type=int, default=500, help='Calculate validation loss every STEPS steps.')
parser.add_argument('--test_dataset', metavar='PATH', type=str, default=datapath+'test.npz', help='Dataset for validation loss, defaults to --dataset.')

parser.add_argument('--theta_size',type=int, default=[100,80,50], help='theta size in topic model')
parser.add_argument('--Hidden_rgbn',type=int, default=[100,80,50], help='hidden in topic model')
parser.add_argument('--seed',type=int, default=1, help='random seed in topic model')
parser.add_argument('--tm_learning_rate',type=int, default=0.001, help='learning_rate of pretrained topic model')
parser.add_argument('--gbn_pretrain',type=bool, default=True, help='wether to pretrain the gbn')
parser.add_argument('--phi_num',type=int, default=80, help='sequence_len of pretrained gbn')
parser.add_argument('--sent_J',type=int, default=4, help='sequence len of context')

## pretain for topic model
parser.add_argument('--train_corpus',type=str, default=datapath+'train.txt', help='train_corpus')
parser.add_argument('--vocab_minfreq',type=int, default=10, help='vocab_minfreq')
parser.add_argument('--vocab_maxfreq',type=float, default=0.001, help='vocab_maxfreq')
parser.add_argument('--doc_sents_num',type=int, default=10, help='doc_sents_num')
parser.add_argument('--TM_Batch_Size',type=int, default=64, help='TM_atch_Size')
parser.add_argument('--tm_epoch_size',type=int, default=101, help='epoch_size of pretrained topic model')
parser.add_argument('--phi_save_path',type=str, default=phi_save_check, help='phi_save_path')


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context

def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    hparams = model.default_hparams()
    with open(os.path.join('./models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if args.model_name == '345M':
        args.memory_saving_gradients = True
        if args.optimizer == 'adam':
            args.only_train_transformer_layers = True

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:

        def sample_batch():
            return [data_sampler.sample(args.train_sequence_len) for _ in range(args.batch_size)]
        def sample_batch_rgbn():
            return [data_sampler.sample_rgbn(args.train_sequence_len, args.sent_J) for _ in range(args.batch_size)]

        def bow(x, V):
            train_bow = np.zeros([len(x), V])
            for doc_index in range(len(x)):
                for word in x[doc_index]:
                    train_bow[doc_index][word] += 1
            return train_bow

        def bow_sequence(x, V):
            train_bow_sequence = []
            for doc_index in range(len(x)):
                train_bow = np.zeros([V])
                for word in x[doc_index]:
                    train_bow[word] += 1
                    train_bow_sequence.append(train_bow.copy())

            return train_bow_sequence

        def bow_sequence_gbn(x, V):
            sent_J = len(x[-1])
            train_bow_sequence = []
            for doc_index in range(len(x)):
                sent_bow = np.zeros([V])
                for sent in range(sent_J-1):
                    for word in x[doc_index][sent]:
                        sent_bow[word] += 1
                for word in x[doc_index][-1]:
                    sent_bow[word] += 1
                    train_bow_sequence.append(sent_bow.copy())

            return train_bow_sequence   #### (Batch_size * sequence) * J * V

        def bow_sequence_test_gbn(x, V):
            sent_J = len(x)
            train_bow_sequence = []
            sent_bow = np.zeros([V])
            for sent in range(sent_J-1):
                for word in x[sent]:
                    if word != 0.0:
                        sent_bow[word] += 1
            for word in x[-1]:
                sent_bow[word] += 1
                train_bow_sequence.append(sent_bow.copy())

            return train_bow_sequence   #### (Batch_size * sequence) * J * V


        print('Loading dataset...')
        chunks = load_dataset(enc, args.dataset, args.combine)
        data_sampler = Sampler(chunks)
        if args.val_every > 0:
            val_chunks = load_dataset(enc, args.val_dataset, args.combine) if args.val_dataset else chunks
        if args.test_every > 0:
            test_chunks = load_dataset(enc, args.test_dataset, args.combine) if args.test_dataset else chunks
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')
        print(args.dataset)

        if args.gbn_pretrain:

            ####pretraining###########
            tf.set_random_seed(args.seed)


            print("Processing train corpus to collect document data...")
            sents, sents_bow, docs, docs_bow = dp.gen_data_bpe(enc,args.doc_sents_num, args.train_corpus)

            train_data_save = open(phi_save_check+'/topic_train_data.pckl', 'wb')
            pickle.dump([sents,sents_bow, docs,docs_bow], train_data_save)
            train_data_save.close()

            doc_num_batches = int(np.floor(float(len(sents_bow)) / args.TM_Batch_Size))
            batch_ids = [item for item in range(doc_num_batches)]

            gbn3 = gbn_3_model(len(enc.encoder), args, len(sents_bow))
            sess.run(tf.global_variables_initializer())
            print("---------------------------pretraining gbn--------------------------")
            pretrain_tm_cost = []
            Phi = gbn3.init_phi()

            topic_vars = [v for v in tf.trainable_variables() if 'Topic' in v.name]
            topic_saver = tf.train.Saver(
                var_list=topic_vars,
                max_to_keep=20)

            for e_tm in range(args.tm_epoch_size):
                MBratio = len(sents_bow)
                pretrain_theta = []
                print("\npretrain_epoch_", e_tm)
                time_start = time.time()
                random.shuffle(batch_ids)
                for batch_id in batch_ids:
                    MBObserved = int(e_tm * doc_num_batches + batch_id)
                    input_batch = np.array(sents_bow[(batch_id * args.TM_Batch_Size):((batch_id + 1) * args.TM_Batch_Size)])
                    pre_tm_cost_batch = 0
                    _, pre_tm_cost, Theta = sess.run([gbn3.tm_train, gbn3.tm_Loss, [gbn3.theta1,gbn3.theta2,gbn3.theta3]],
                                                     feed_dict={gbn3.input_x: input_batch, gbn3.phi1: Phi[0], gbn3.phi2: Phi[1], gbn3.phi3: Phi[2]})
                    Phi = gbn3.updatePhi(input_batch, Phi, Theta, MBratio, MBObserved)
                    pretrain_theta.append(Theta)
                    pretrain_tm_cost.append(pre_tm_cost)
                    pre_tm_cost_batch += pre_tm_cost
                print("topic model Cost:", pre_tm_cost_batch / args.TM_Batch_Size)
                print("per epoch time:", time.time() - time_start )
                if e_tm % 10 == 0:
                    pretrain_gbn_save = open(phi_save_check+'/pretrain_gbn_' + str(e_tm) + '.pckl', 'wb')
                    pickle.dump([pretrain_theta, Phi], pretrain_gbn_save)
                    pretrain_gbn_save.close()

                    topic_saver.save(
                        sess,
                        os.path.join(phi_save_check+'/pretrain_gbn_model'),
                        global_step=e_tm)

        context = tf.placeholder(tf.int32, [args.batch_size, None])
        context_in = randomize(context, hparams, args.noise)
        Phi_tensor_1 = tf.convert_to_tensor(Phi[0].astype('float32'))
        Phi_tensor_2 = tf.matmul(Phi_tensor_1,tf.convert_to_tensor(Phi[1].astype('float32')))
        Phi_tensor_3 = tf.matmul(Phi_tensor_2,tf.convert_to_tensor(Phi[2].astype('float32')))

        Phi_vec_1 = tf.gather(Phi_tensor_1, context[0])
        Phi_vec_2 = tf.gather(Phi_tensor_2, context[0])
        Phi_vec_3 = tf.gather(Phi_tensor_3, context[0])

        gbn3 = gbn_3_model(len(Phi[0]), args, len(docs_bow))
        theta = [tf.transpose(gbn3.theta1/tf.reduce_sum(gbn3.theta1,axis=0,keep_dims=True)),
                 tf.transpose(gbn3.theta2/tf.reduce_sum(gbn3.theta2,axis=0,keep_dims=True)),
                 tf.transpose(gbn3.theta3/tf.reduce_sum(gbn3.theta3,axis=0,keep_dims=True))]
        topic_vec_1 = tf.multiply(Phi_vec_1, theta[0])
        topic_vec_2 = tf.multiply(Phi_vec_2, theta[1])
        topic_vec_3 = tf.multiply(Phi_vec_3, theta[2])
        topic_vec = tf.concat([topic_vec_1,topic_vec_2,topic_vec_3],axis=-1)
        theta_size = np.sum(args.theta_size)
        output = model.model_layer3_embed(hparams=hparams, X=context_in, topic = topic_vec, thetasize=theta_size)


        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))
        loss_all = loss


        if args.val_every > 0:
            val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
            val_Phi_vec_1 = tf.gather(Phi_tensor_1, val_context[0])
            val_Phi_vec_2 = tf.gather(Phi_tensor_2, val_context[0])
            val_Phi_vec_3 = tf.gather(Phi_tensor_3, val_context[0])
            val_theta = [tf.transpose(gbn3.theta1 / tf.reduce_sum(gbn3.theta1, axis=0, keep_dims=True)),
                     tf.transpose(gbn3.theta2 / tf.reduce_sum(gbn3.theta2, axis=0, keep_dims=True)),
                     tf.transpose(gbn3.theta3 / tf.reduce_sum(gbn3.theta3, axis=0, keep_dims=True))]
            val_topic_vec_1 = tf.multiply(val_Phi_vec_1, val_theta[0])
            val_topic_vec_2 = tf.multiply(val_Phi_vec_2, val_theta[1])
            val_topic_vec_3 = tf.multiply(val_Phi_vec_3, val_theta[2])
            val_topic_vec = tf.concat([val_topic_vec_1, val_topic_vec_2, val_topic_vec_3], axis=-1)
            val_output = model.model_layer3_embed(hparams=hparams, X=val_context, topic=val_topic_vec, thetasize=theta_size)

            val_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
            val_loss_all = val_loss

            val_loss_summary = tf.summary.scalar('val_loss', val_loss)

        if args.test_every > 0:
            test_context = tf.placeholder(tf.int32,  [1,None])
            test_Phi_vec_1 = tf.gather(Phi_tensor_1, test_context[0])
            test_Phi_vec_2 = tf.gather(Phi_tensor_2, test_context[0])
            test_Phi_vec_3 = tf.gather(Phi_tensor_3, test_context[0])
            test_theta = [tf.transpose(gbn3.theta1 / tf.reduce_sum(gbn3.theta1, axis=0, keep_dims=True)),
                     tf.transpose(gbn3.theta2 / tf.reduce_sum(gbn3.theta2, axis=0, keep_dims=True)),
                     tf.transpose(gbn3.theta3 / tf.reduce_sum(gbn3.theta3, axis=0, keep_dims=True))]
            test_topic_vec_1 = tf.multiply(test_Phi_vec_1, test_theta[0])
            test_topic_vec_2 = tf.multiply(test_Phi_vec_2, test_theta[1])
            test_topic_vec_3 = tf.multiply(test_Phi_vec_3, test_theta[2])
            test_topic_vec = tf.concat([test_topic_vec_1, test_topic_vec_2, test_topic_vec_3], axis=-1)
            test_output = model.model_layer3_embed(hparams=hparams, X=test_context, topic=test_topic_vec, thetasize=theta_size)

            test_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=test_context[:, 1:], logits=test_output['logits'][:, :-1]))
            test_loss_all = test_loss

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

        new_vars=[all_vars[2]]
        pretrain_vars = [v for v in all_vars if v not in new_vars]


        if args.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        else:
            exit('Bad optimizer:', args.optimizer)

        if args.accumulate_gradients > 1:
            if args.memory_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=opt,
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss_all)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            if args.memory_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(loss_all, train_vars)
            else:
                opt_grads = tf.gradients(loss_all, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            os.path.join(checkpoint_path, args.run_name))

        sess.run(tf.global_variables_initializer())

        topic_vars = [v for v in tf.trainable_variables() if 'Topic' in v.name]
        topic_saver = tf.train.Saver(
            var_list=topic_vars,
            max_to_keep=20)
        topic_ckpt = phi_save_check + '/pretrain_gbn_model-' + str(args.phi_num)
        topic_saver.restore(sess, topic_ckpt)

        saver = tf.train.Saver(
            var_list=pretrain_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        if args.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(checkpoint_path, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('./models', args.model_name))
        elif args.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('./models', args.model_name))
        else:
            ckpt = tf.train.latest_checkpoint(args.restore_from)
        print('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)

        print('Loading dataset...')
        Len_word, Len_character, Len_subword = load_raw_dataset(enc, args.test_raw_dataset, args.combine)
        print('Len_word: ', Len_word)
        print('Len_character: ', Len_character)
        print('Len_subword: ', Len_subword)

        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_chunks, seed=1)
            val_batches_rgbn = [[val_data_sampler.sample_rgbn(args.valid_sequence_len, args.sent_J) for _ in range(args.val_batch_size)]
                           for _ in range(args.val_batch_count)]

        if args.test_every > 0:
            test_data_sampler = Sampler(test_chunks, seed=1)
            test_batches_rgbn = test_data_sampler.sample_all_rgbn(args.valid_sequence_len, args.sent_J)
        val_ppl_recoder = []
        test_ppl_recoder = []
        counter = 1
        counter_path = os.path.join(checkpoint_path, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(checkpoint_path, args.run_name))
            print(
                'Saving',
                os.path.join(checkpoint_path, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(checkpoint_path, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def validation():
            print('Calculating validation loss...')
            losses = []
            for batch in tqdm.tqdm(val_batches_rgbn):
                val_batch_gbn = bow_sequence_gbn(batch, len(enc.decoder))
                losses.append(sess.run(val_loss, feed_dict={val_context: np.array(batch)[:,-1,:], gbn3.input_x:val_batch_gbn}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            print(
                '[{counter} | {time:2.2f}] validation loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))
            print('val perplexity:',np.exp(v_val_loss))
            val_ppl_recoder.append(np.exp(v_val_loss))
            pickle.dump(val_ppl_recoder, open(os.path.join(checkpoint_path, args.run_name, 'val_ppl_recoder.pckl') , 'wb'))
            loss_eval.append(v_val_loss)


        def test():
            print('Calculating test loss...')
            losses = []
            ppl = []
            for batch in tqdm.tqdm(test_batches_rgbn):
                test_batch_gbn = bow_sequence_test_gbn(batch, len(enc.decoder))
                losses.append(sess.run(test_loss, feed_dict={test_context: np.reshape(np.array(batch)[-1,:],[1,-1]), gbn3.input_x:test_batch_gbn}))

            v_test_loss = np.mean(losses)
            test_ppl_recoder.append(np.exp(v_test_loss))
            pickle.dump(test_ppl_recoder, open(os.path.join(checkpoint_path, args.run_name, 'test_ppl_recoder.pckl') , 'wb'))

            print(
                '[{counter} | {time:2.2f}] test loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_test_loss))
            print('test perplexity:',np.exp(v_test_loss))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        loss_eval = []

        try:
            while True:
                if counter % args.save_every == 0:
                    save()
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    validation()
                if args.test_every > 0 and (counter % args.test_every == 0 or counter == 1):
                    test()

                if args.accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        sess.run(
                            opt_compute, feed_dict={context: sample_batch()})
                    (v_loss, v_summary) = sess.run((opt_apply, summaries))
                else:
                    train_batch = sample_batch_rgbn()
                    train_batch_gbn = bow_sequence_gbn(train_batch, len(enc.decoder))
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summaries),
                        feed_dict={context: np.array(train_batch)[:,-1,:], gbn3.input_x:train_batch_gbn})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter += 1


        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    main()
