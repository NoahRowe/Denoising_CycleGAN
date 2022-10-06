#!/usr/bin/env python3

###################################################################################################################
# IMPORTS
###################################################################################################################

from pathlib import Path
import time

import tensorflow as tf

from networks.generator import autoencoder
from networks.discriminator import convolutional_classifier

from utilities.dataset.load_files import get_file_lists
from utilities.dataset.make_dataset import make_dataset

from utilities.configure.configure_tensorflow import configure_tensorflow
configure_tensorflow(eager=False, gpu_index=0, allow_growth_gpu=False)


###################################################################################################################
# DEFINE DATA OBJECTS
###################################################################################################################

BATCH_SIZE = 128

data_directory = "/scratch/manderson/numpy/pulses/lib/detector_final/"

file_lists = get_file_lists(data_directory)
list_X_train, list_Y_train = file_lists['train']
list_X_val, list_Y_val = file_lists['val']

# Create the tensorflow datasets
print("Creating dataset objects... ", end="")
dataset_train = make_dataset(list_X_train, list_Y_train, batch_size=BATCH_SIZE, prefetch=150)
dataset_val = make_dataset(list_X_val, list_Y_val, batch_size=BATCH_SIZE, prefetch=150)
print("Done.")


###################################################################################################################
# DEFINE MODELS
###################################################################################################################

input_tuple = (4096,)

generator_c2n = autoencoder(input_tuple=input_tuple)
generator_n2c = autoencoder(input_tuple=input_tuple)

discriminator_c = convolutional_classifier(input_tuple=input_tuple)
discriminator_n = convolutional_classifier(input_tuple=input_tuple)


###################################################################################################################
# DEFINE LOSS FUNCTIONS
###################################################################################################################

LAMBDA = 10

binary_crossentropy_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False) # True or false?

def discriminator_loss(real_pulses, fake_pulses):
    clean_loss = binary_crossentropy_loss_fn(tf.ones_like(real_pulses), real_pulses)
    noisy_loss = binary_crossentropy_loss_fn(tf.zeros_like(fake_pulses), fake_pulses)
    loss = clean_loss + noisy_loss
    return loss * 0.5

def generator_loss(generated_predictions):
    loss = binary_crossentropy_loss_fn(tf.ones_like(generated_predictions), 
                                       generated_predictions)
    return loss


def cycle_consistency_loss(real_pulses, cycled_pulses):
    loss = tf.reduce_mean(tf.abs(real_pulses - cycled_pulses))
    return loss * LAMBDA

def identity_loss(real_pulses, same_pulses):
    loss = tf.reduce_mean(tf.abs(real_pulses - same_pulses))
    return loss * LAMBDA * 0.5


###################################################################################################################
# DEFINE OPTIMIZERS
###################################################################################################################

generator_n2c_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_c2n_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_n_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_c_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


###################################################################################################################
# DEFINE TRAINING CHECKPOINT
###################################################################################################################

checkpoint_path = "./model_checkpoints/train/"

ckpt = tf.train.Checkpoint(generator_c2n=generator_c2n,
                           generator_n2c=generator_n2c,
                           discriminator_c=discriminator_c,
                           discriminator_n=discriminator_n,
                           generator_n2c_optimizer=generator_n2c_optimizer,
                           generator_c2n_optimizer=generator_c2n_optimizer,
                           discriminator_c_optimizer=discriminator_c_optimizer,
                           discriminator_n_optimizer=discriminator_n_optimizer)


###################################################################################################################
# DEFINE TRAINING STEP
###################################################################################################################

# @tf.function
def train_step(real_clean, real_noisy):
    print("TRAINING STEP")
    with tf.GradientTape(persistent=True) as tape:
        
        ##########################################################################################################
        # Generate predictions
        ##########################################################################################################
        # Generator outputs
        fake_clean = generator_n2c(real_noisy)
        fake_noisy = generator_c2n(real_clean)
        
        # Used for cycle loss
        cycled_clean = generator_c2n(fake_clean)
        cycled_noisy = generator_n2c(fake_noisy)
        
        # Used for identity loss
        same_clean = generator_n2c(real_clean)
        same_noisy = generator_c2n(real_noisy)
        
        # Get discriminator output on real pulses
        disc_real_clean = discriminator_c(real_clean)
        disc_real_noisy = discriminator_n(real_noisy)
        
        # Get discriminator output on fake pulses
        disc_fake_clean = discriminator_c(fake_clean)
        disc_fake_noisy = discriminator_n(fake_noisy)
        
        ##########################################################################################################
        # Calculate losses
        ##########################################################################################################
        
        gen_n2c_loss = generator_loss(disc_fake_clean)
        gen_c2n_loss = generator_loss(disc_fake_noisy)
        
        clean_cycle_loss = cycle_consistency_loss(real_clean, cycled_clean)
        noisy_cycle_loss = cycle_consistency_loss(real_noisy, cycled_noisy)
        total_cycle_loss = clean_cycle_loss + noisy_cycle_loss
        
        n2c_identity_loss = identity_loss(real_clean, same_clean)
        c2n_identity_loss = identity_loss(real_noisy, same_noisy)
        
        total_gen_n2c_loss = gen_n2c_loss + total_cycle_loss + n2c_identity_loss
        total_gen_c2n_loss = gen_c2n_loss + total_cycle_loss + c2n_identity_loss
        
        disc_c_loss = discriminator_loss(disc_real_clean, disc_fake_clean)
        disc_n_loss = discriminator_loss(disc_real_noisy, disc_fake_noisy)
        
    ##############################################################################################################
    # Calculate gradients
    ##############################################################################################################
    # Generator gradients
    generator_n2c_gradients = tape.gradient(total_gen_n2c_loss, generator_n2c.trainable_variables)
    generator_c2n_gradients = tape.gradient(total_gen_c2n_loss, generator_c2n.trainable_variables)
    
    # Discriminator gradients
    discriminator_c_gradients = tape.gradient(disc_c_loss, discriminator_c.trainable_variables)
    discriminator_n_gradients = tape.gradient(disc_n_loss, discriminator_n.trainable_variables)
    
    ##############################################################################################################
    # Apply gradients
    ##############################################################################################################
    # Apply to generators
    generator_n2c_optimizer.apply_gradients(zip(generator_n2c_gradients, generator_n2c.trainable_variables))
    generator_c2n_optimizer.apply_gradients(zip(generator_c2n_gradients, generator_c2n.trainable_variables))
    
    # Apply to discriminators
    discriminator_c_optimizer.apply_gradients(zip(discriminator_c_gradients, discriminator_c.trainable_variables))
    discriminator_n_optimizer.apply_gradients(zip(discriminator_n_gradients, discriminator_n.trainable_variables))


###################################################################################################################
# TRAIN THE MODEL
###################################################################################################################

EPOCHS = 10
@tf.function
def train_function():

    for epoch in range(EPOCHS):
        start = time.time()

        n = 0
        for clean_pulses, noisy_pulses in dataset_train:
            train_step(real_clean=clean_pulses, real_noisy=noisy_pulses)
            if n % 1 == 0:
                print('.', end='')
            n += 1

        # Checkpoint
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = checkpoint_path + f"ckpt_{epoch}"
            ckpt.write(ckpt_save_path)
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

        print ('Time taken for epoch {} is {:.2f} sec\n'.format(epoch + 1, time.time()-start))
        
train_function()
