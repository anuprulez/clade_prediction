import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import random
from random import choices
import logging
import tensorflow as tf
import h5py
import copy
import glob


import neural_network
import utils

PATH_KMER_F_DICT = "data/ncov_global/kmer_f_word_dictionaries.json"
PRE_TR_GEN_ENC_WEIGHTS = "data/generated_files/pretr_generator_encoder_weights.h5"
PRE_TR_GEN_DEC_WEIGHTS = "data/generated_files/pretr_generator_decoder_weights.h5"
PRETRAIN_GEN_ENC_MODEL = "data/generated_files/pretrain_gen_encoder"
PRETRAIN_GEN_DEC_MODEL = "data/generated_files/pretrain_gen_decoder"

DISC_WEIGHTS = "data/generated_files/disc_weights.h5"
DISC_PAR_ENC_WEIGHTS = "data/generated_files/disc_par_enc_weights.h5"
DISC_GEN_ENC_WEIGHTS = "data/generated_files/disc_gen_enc_weights.h5"

GEN_ENC_WEIGHTS = "data/generated_files/generator_encoder_weights.h5"
GEN_DEC_WEIGHTS = "data/generated_files/generator_decoder_weights.h5"

TRAIN_GEN_ENC_MODEL = "data/generated_files/gen_enc_model"
TRAIN_GEN_DEC_MODEL = "data/generated_files/gen_dec_model"


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

n_disc_step = 6
n_gen_step = 3
unrolled_steps = 0
test_log_step = 100
teacher_forcing_ratio = 0.0
disc_clip_norm = 5.0
gen_clip_norm = 5.0
pretrain_clip_norm = 1.0



def wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.multiply(y_true, y_pred))


def discriminator_loss(real_output, fake_output):
    #real_loss = -tf.math.reduce_mean(real_output)
    #fake_loss = tf.math.reduce_mean(fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss, fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    #return -tf.math.reduce_mean(fake_output)


def get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc_model, disc_gen_enc_model, size_stateful, pos_size, pos_variations, pos_variations_count, step):

    generated_logits, encoder, decoder, gen_t_loss = utils.loop_encode_decode_stateful(seq_len, batch_size, vocab_size, unrolled_x, unrolled_y, encoder, decoder, enc_units, teacher_forcing_ratio, True, size_stateful, pos_size, pos_variations, pos_variations_count, step)

    variation_score = utils.get_sequence_variation_percentage(unrolled_x, generated_logits)
    print("Generation variation score: {}".format(str(variation_score)))

    gen_tokens = tf.argmax(generated_logits, axis=-1)
    print("True output")
    print(unrolled_y[:batch_size, :])
    print()
    print("Gen output")
    print(gen_tokens[:batch_size, :])
    print()


    _, real_enc_f_x, real_enc_b_x = disc_par_enc_model(unrolled_x, training=True)
    real_state_x = real_enc_f_x + real_enc_b_x
    #print(real_state_x.shape, real_enc_f_x.shape, real_enc_b_x.shape)
    # unrelated real X
    _, unrelated_real_state_f_x, unrelated_real_state_b_x = disc_par_enc_model(un_X, training=True)
    unrelated_real_state_x = unrelated_real_state_f_x + unrelated_real_state_b_x
    # encode true child sequences for discriminator
    # reformat real output to one-hot encoding
    one_hot_real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)
    #real_state_y = disc_gen_enc_model(one_hot_real_y, training=True)

    _, real_enc_f_y, real_enc_b_y = disc_gen_enc_model(one_hot_real_y, training=True)
    real_state_y = real_enc_f_y + real_enc_b_y
    #print(real_state_y.shape, real_enc_f_y.shape, real_enc_b_y.shape)
    # unrelated real y
    one_hot_unrelated_y = tf.one_hot(un_y, depth=generated_logits.shape[-1], axis=-1)
    _, unrelated_real_state_f_y, unrelated_real_state_b_y = disc_gen_enc_model(one_hot_unrelated_y, training=True)
    unrelated_real_state_y = unrelated_real_state_f_y + unrelated_real_state_b_y

    # encode generated child sequences for discriminator
    #fake_state_y = disc_gen_enc_model(generated_logits, training=True)
    _, fake_enc_f_y, fake_enc_b_y = disc_gen_enc_model(generated_logits, training=True)
    fake_state_y = fake_enc_f_y + fake_enc_b_y
    #print(fake_state_y.shape, fake_enc_f_y.shape, fake_enc_b_y.shape)
    return real_state_x, real_state_y, fake_state_y, unrelated_real_state_x, unrelated_real_state_y, encoder, decoder, disc_par_enc_model, disc_gen_enc_model, gen_t_loss


def d_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful, pos_size, pos_variations, pos_variations_count, step):
    print("Applying gradient update on discriminator...")
    with tf.GradientTape() as disc_tape:
        real_x, real_y, fake_y, unreal_x, unreal_y, _, _, disc_par_enc, disc_gen_enc, _ = get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, size_stateful, pos_size, pos_variations, pos_variations_count, step)
        # discriminate pairs of true parent and true child sequences
        real_output = discriminator([real_x, real_y], training=True)
        # discriminate pairs of true parent and generated child sequences
        fake_output = discriminator([real_x, fake_y], training=True)
        # discriminate pairs of true parent and random sequences
        unreal_output = discriminator([unreal_x, unreal_y], training=True)
        # halve the fake outpus and combine them to keep the final size same as the real output
        t1 = fake_output[:int(batch_size/2.0)]
        t2 = unreal_output[:int(batch_size/2.0)]
        combined_fake_output = tf.concat([t1, t2], 0)
        # compute discriminator loss
        disc_real_loss, disc_fake_loss = discriminator_loss(real_output, combined_fake_output)
        total_disc_loss = disc_real_loss + disc_fake_loss
    # update discriminator's parameters
    print()
    disc_trainable_vars = discriminator.trainable_variables + disc_gen_enc.trainable_variables + disc_par_enc.trainable_variables
    gradients_of_discriminator = disc_tape.gradient(total_disc_loss, disc_trainable_vars)
    #print("Train Disc gradient norm before clipping: ", [tf.norm(gd) for gd in gradients_of_discriminator])
    gradients_of_discriminator = [(tf.clip_by_norm(grad, clip_norm=disc_clip_norm)) for grad in gradients_of_discriminator]
    #print("Train Disc gradient norm after clipping: ", [tf.norm(gd) for gd in gradients_of_discriminator])
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_trainable_vars))
    return encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, disc_real_loss, disc_fake_loss, total_disc_loss


def g_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful, pos_size, pos_variations, pos_variations_count, step):
    print("Applying gradient update on generator...")
    with tf.GradientTape() as gen_tape:
        real_x, _, fake_y, _, _, encoder, decoder, _, _, gen_true_loss = get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, size_stateful, pos_size, pos_variations, pos_variations_count, step)
        # discriminate pairs of true parent and generated child sequences
        fake_output = discriminator([real_x, fake_y], training=True)
        gen_fake_loss = generator_loss(fake_output)
        total_gen_loss = gen_true_loss + gen_fake_loss
    print()
    # get all trainable vars for generator
    gen_trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients_of_generator = gen_tape.gradient(total_gen_loss, gen_trainable_vars)
    #gradients_norm = [tf.norm(gd) for gd in gradients_of_generator]
    #print("Train Gen gradient norm: {}".format(str(tf.reduce_mean(gradients_norm).numpy())))
    #print("Train Gen gradient norm before clipping: ", [tf.norm(gd) for gd in gradients_of_generator])
    gradients_of_generator = [(tf.clip_by_norm(grad, clip_norm=gen_clip_norm)) for grad in gradients_of_generator]
    #print("Train Gen gradient norm after clipping: ", [tf.norm(gd) for gd in gradients_of_generator])
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_trainable_vars))
    return encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, gen_true_loss, gen_fake_loss, total_gen_loss


def sample_true_x_y(batch_size, X_train, y_train, cluster_indices):
    '''rand_batch_indices = np.random.randint(0, X_train.shape[0], batch_size)
    x_batch_train = X_train[rand_batch_indices]
    y_batch_train = y_train[rand_batch_indices]
    unrolled_x = utils.convert_to_array(x_batch_train)
    unrolled_y = utils.convert_to_array(y_batch_train)'''
    # sample_true_x_y(batch_size, X_train, y_train, cluster_indices)'''
    cluster_keys = list(cluster_indices.keys())
    cluster_keys = list(np.unique(cluster_keys))
    random.shuffle(cluster_keys)
    if len(cluster_keys) >= batch_size:
        rand_keys = random.sample(cluster_keys, batch_size)
    else:
        rand_keys = np.array(choices(cluster_keys, k=batch_size))
    rand_batch_indices = list()
    #print(rand_keys)
    for key in rand_keys:
        rows_indices = cluster_indices[key]
        random.shuffle(rows_indices)
        rand_batch_indices.append(rows_indices[0])
    x_batch_train = X_train[rand_batch_indices]
    y_batch_train = y_train[rand_batch_indices]
    unrolled_x = utils.convert_to_array(x_batch_train)
    unrolled_y = utils.convert_to_array(y_batch_train)

    return unrolled_x, unrolled_y


def redraw_unique(x, y):
    x_str = utils.convert_to_string_list(x)
    y_str = utils.convert_to_string_list(y)
    u_x = list(set(x_str))
    u_y = list(set(y_str))
    print(len(u_x), len(u_y))
    return u_x, u_y


def get_mut_size(mut_dist):
    pos_size = dict()
    for mut in mut_dist:
        size = mut_dist[mut]
        pos = mut.split(">")[1]
        if pos not in pos_size:
            pos_size[pos] = 0
        pos_size[pos] += len(size)
    return pos_size


def get_text_data():
    tr_file = glob.glob("data/ncov_global/data_from_colab/train/*.csv")
    te_file = glob.glob("data/ncov_global/data_from_colab/test/*.csv")

    path_train_x = "data/ncov_global/data_from_colab/train/train_x.csv"
    path_train_y = "data/ncov_global/data_from_colab/train/train_y.csv"
    path_test_x = "data/ncov_global/data_from_colab/test/test_x.csv"
    path_test_y = "data/ncov_global/data_from_colab/test/test_y.csv"

    train_x = pd.read_csv(path_train_x, sep=",", header=None)
    train_y = pd.read_csv(path_train_y, sep=",", header=None)
    test_x = pd.read_csv(path_test_x, sep=",", header=None)
    test_y = pd.read_csv(path_test_y, sep=",", header=None)

    train_x[0] = 0
    train_y[0] = 0
    test_x[0] = 0
    test_y[0] = 0
    print(train_x)
    print(train_y)
    return train_x, train_y, test_x, test_y


def pretrain_generator(inputs, epo_step, gen_encoder, gen_decoder, updated_lr, enc_units, vocab_size, n_batches, batch_size, pretr_parent_child_mut_indices, epochs, size_stateful, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict, pos_variations, pos_variations_count, cluster_indices):
  X_train, y_train, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches = inputs
  epo_avg_tr_gen_loss = list()
  epo_te_gen_loss = list()
  epo_tr_seq_var = list()
  epo_te_seq_var = list()
  batch_mut_distribution = dict()
  pos_size = dict()

  epo_pre_train_save_folder = "data/generated_files/pre_train/{}".format(str(epo_step+1))
  enc_pre_train_save_folder = "data/generated_files/pre_train/{}/enc".format(str(epo_step+1))
  dec_pre_train_save_folder = "data/generated_files/pre_train/{}/dec".format(str(epo_step+1))

  utils.create_dirs(epo_pre_train_save_folder)
  utils.create_dirs(enc_pre_train_save_folder)
  utils.create_dirs(dec_pre_train_save_folder)

  for step in range(n_batches):
      #updated_lr = utils.decayed_learning_rate(updated_lr, (epo_step + 1) * (step + 1))
      pretrain_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=updated_lr)
      unrolled_x, unrolled_y = sample_true_x_y(batch_size, X_train, y_train, cluster_indices)
      seq_len = unrolled_x.shape[1]
      with tf.GradientTape() as gen_tape:
          pred_logits, gen_encoder, gen_decoder, gen_loss = utils.loop_encode_decode_stateful(seq_len, batch_size, vocab_size, unrolled_x, unrolled_y, gen_encoder, gen_decoder, enc_units, teacher_forcing_ratio, True, size_stateful, pos_size, pos_variations, pos_variations_count, step)

          print("Training: true output seq")
          print(unrolled_y[:batch_size,], unrolled_y.shape)
          print()
          print(tf.argmax(pred_logits, axis=-1)[:batch_size, :], pred_logits.shape)

          # compute generated sequence variation
          variation_score = utils.get_sequence_variation_percentage(unrolled_x, pred_logits)
          print("Pretr: generation variation score: {}".format(str(variation_score)))
          epo_tr_seq_var.append(variation_score)

      print("Pretrain epoch {}/{}, batch {}/{}, gen true loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_batches), str(gen_loss.numpy())))
      print()
      gen_trainable_vars = gen_encoder.trainable_variables + gen_decoder.trainable_variables
      gradients_of_generator = gen_tape.gradient(gen_loss, gen_trainable_vars)
      gradients_of_generator = [(tf.clip_by_norm(grad, clip_norm=pretrain_clip_norm)) for grad in gradients_of_generator]
      pretrain_generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_trainable_vars))
      #print(pretrain_generator_optimizer.lr)

      # optimize pf discriminator
      if (step + 1) % test_log_step == 0 and step > 0:
          print("-------")
          print("Pretr: Prediction on test data at epoch {}/{}, batch {}/{}...".format(str(epo_step+1), str(epochs), str(step+1), str(n_batches)))
          print()
          gen_te_loss, gen_te_seq_var = utils.predict_sequence(epo_step, step, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, gen_encoder, gen_decoder, size_stateful, "pretrain", True)
          epo_te_gen_loss.append(gen_te_loss)
          epo_te_seq_var.append(gen_te_seq_var)
          print("-------")
          print()
      epo_avg_tr_gen_loss.append(gen_loss)
  # save model

  gen_encoder.save_weights(GEN_ENC_WEIGHTS)

  tf.keras.models.save_model(gen_encoder, PRETRAIN_GEN_ENC_MODEL)
  tf.keras.models.save_model(gen_decoder, PRETRAIN_GEN_DEC_MODEL)

  gen_encoder.save_weights(PRE_TR_GEN_ENC_WEIGHTS)
  gen_decoder.save_weights(PRE_TR_GEN_DEC_WEIGHTS)

  tf.keras.models.save_model(gen_encoder, enc_pre_train_save_folder)
  tf.keras.models.save_model(gen_decoder, dec_pre_train_save_folder)

  utils.save_as_json("data/generated_files/pretr_ave_batch_x_y_mut_epo_{}.json".format(str(epo_step)), batch_mut_distribution)
  return np.mean(epo_avg_tr_gen_loss), np.mean(epo_te_gen_loss), np.mean(epo_te_seq_var), np.mean(epo_tr_seq_var), gen_encoder, gen_decoder, updated_lr


def start_training_mut_balanced(inputs, epo_step, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, enc_units, vocab_size, n_train_batches, batch_size, parent_child_mut_indices, epochs, size_stateful, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict, pos_variations, pos_variations_count, train_cluster_indices_dict):
  """
  Training sequences balanced by mutation type
  """
  X_train, y_train, unrelated_X, unrelated_y, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches = inputs

  epo_avg_total_gen_loss = list()
  epo_ave_gen_true_loss = list()
  epo_avg_gen_fake_loss = list()

  epo_avg_total_disc_loss = list()
  epo_avg_disc_fake_loss = list()
  epo_avg_disc_real_loss = list()
  disc_real_loss = tf.constant(0)
  disc_fake_loss = tf.constant(0)
  total_disc_loss = tf.constant(0)
  gen_fake_loss = tf.constant(0)
  gen_true_loss = tf.constant(0)
  total_gen_loss = tf.constant(0)
  batch_mut_distribution = dict()

  epo_te_gen_loss = list()
  epo_te_seq_var = list()

  pos_size = dict() #get_mut_size(parent_child_mut_indices)

  mut_keys = list(parent_child_mut_indices.keys())

  epo_train_save_folder = "data/generated_files/gan_train/{}".format(str(epo_step+1))
  enc_train_save_folder = "data/generated_files/gan_train/{}/enc".format(str(epo_step+1))
  dec_train_save_folder = "data/generated_files/gan_train/{}/dec".format(str(epo_step+1))
  utils.create_dirs(epo_train_save_folder)
  utils.create_dirs(enc_train_save_folder)
  utils.create_dirs(dec_train_save_folder)

  for step in range(n_train_batches):
      #unrolled_x, unrolled_y, batch_mut_distribution = sample_true_x_y(parent_child_mut_indices, batch_size, X_train, y_train, batch_mut_distribution)
      unrolled_x, unrolled_y = sample_true_x_y(batch_size, X_train, y_train, train_cluster_indices_dict)
      un_X, un_y = utils.sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size)
      seq_len = unrolled_x.shape[1]
      disc_gen = step % n_disc_step
      if disc_gen in list(range(0, n_disc_step - n_gen_step)):
          # train discriminator
          _, _, disc_par_enc, disc_gen_enc, discriminator, disc_real_loss, disc_fake_loss, total_disc_loss = d_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful, pos_size, pos_variations, pos_variations_count, step)
          # share weights with generator's encoder
          disc_par_enc.load_weights(GEN_ENC_WEIGHTS)
          disc_gen_enc.load_weights(GEN_ENC_WEIGHTS)
          #disc_gen_enc.layers[1].set_weights(disc_par_enc.layers[1].get_weights())
          print("Training epoch {}/{}, batch {}/{}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_train_batches), str(disc_real_loss.numpy()), str(disc_fake_loss.numpy()), str(total_disc_loss.numpy())))
      else:
          # train generator with unrolled discriminator
          # save disc weights to reset after unrolling
          discriminator.save_weights(DISC_WEIGHTS)
          disc_par_enc.save_weights(DISC_PAR_ENC_WEIGHTS)
          disc_gen_enc.save_weights(DISC_GEN_ENC_WEIGHTS)
          print("Applying unrolled steps...")
          # unrolling steps
          for i in range(unrolled_steps):
              print("Unrolled step: {}/{}".format(str(i+1), str(unrolled_steps)))
              # sample data for unrolling
              #unroll_x, unroll_y, _ = sample_true_x_y(parent_child_mut_indices, batch_size, X_train, y_train, batch_mut_distribution)
              unroll_x, unroll_y = sample_true_x_y(batch_size, X_train, y_train, train_cluster_indices_dict)
              un_unroll_X, un_unroll_y = utils.sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size)
              # train discriminator
              _, _, disc_par_enc, disc_gen_enc, discriminator, d_r_l, d_f_l, d_t_l = d_loop(seq_len, batch_size, vocab_size, enc_units, unroll_x, unroll_y, un_unroll_X, un_unroll_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful, pos_size, pos_variations, pos_variations_count, step)
              print("Unrolled disc losses: real {}, fake {}, total {}".format(str(d_r_l.numpy()), str(d_f_l.numpy()), str(d_t_l.numpy())))
          # finish unrolling
          # train generator with unrolled discriminator
          encoder, decoder, _, _, _, gen_true_loss, gen_fake_loss, total_gen_loss = g_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful, pos_size, pos_variations, pos_variations_count, step)
          print("Training epoch {}/{}, batch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_train_batches), str(gen_true_loss.numpy()), str(gen_fake_loss.numpy()), str(total_gen_loss.numpy())))
          encoder.save_weights(GEN_ENC_WEIGHTS)
          # reset weights of discriminator, disc_par_enc and disc_gen_enc after unrolling
          discriminator.load_weights(DISC_WEIGHTS)
          disc_par_enc.load_weights(DISC_PAR_ENC_WEIGHTS)
          disc_gen_enc.load_weights(DISC_GEN_ENC_WEIGHTS)
      # intermediate prediction on test data while training
      if (step + 1) % test_log_step == 0 and step > 0:
          print("Training: prediction on test data...")
          with tf.device('/device:cpu:0'):
              _, _ = utils.predict_sequence(epo_step, step, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, encoder, decoder, size_stateful, "gan_train", True)
      
      print("Training epoch {}/{}, batch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_train_batches), str(gen_true_loss.numpy()), str(gen_fake_loss.numpy()), str(total_gen_loss.numpy()), str(disc_real_loss.numpy()), str(disc_fake_loss.numpy()), str(total_disc_loss.numpy())))
      # write off results
      epo_ave_gen_true_loss.append(gen_true_loss.numpy())
      epo_avg_gen_fake_loss.append(gen_fake_loss.numpy())
      epo_avg_total_gen_loss.append(total_gen_loss.numpy())
      epo_avg_disc_fake_loss.append(disc_fake_loss.numpy())
      epo_avg_disc_real_loss.append(disc_real_loss.numpy())
      epo_avg_total_disc_loss.append(total_disc_loss.numpy())
  # save model
  print("Training epoch {} finished, Saving model...".format(str(epo_step+1)))
  print()

  tf.keras.models.save_model(encoder, TRAIN_GEN_ENC_MODEL)
  tf.keras.models.save_model(decoder, TRAIN_GEN_DEC_MODEL)
 
  # save trained models per epoch
  tf.keras.models.save_model(encoder, enc_train_save_folder)
  tf.keras.models.save_model(decoder, dec_train_save_folder)

  encoder.save_weights(GEN_ENC_WEIGHTS)
  decoder.save_weights(GEN_DEC_WEIGHTS)
  utils.save_as_json("data/generated_files/ave_batch_x_y_mut_epo_{}.json".format(str(epo_step)), batch_mut_distribution)
  return np.mean(epo_ave_gen_true_loss), np.mean(epo_avg_gen_fake_loss), np.mean(epo_avg_total_gen_loss), np.mean(epo_avg_disc_real_loss), np.mean(epo_avg_disc_fake_loss), np.mean(epo_avg_total_disc_loss), np.mean(epo_te_gen_loss), np.mean(epo_te_seq_var), encoder, decoder
