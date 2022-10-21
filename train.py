#!/usr/bin/env python3

###################################################################################################################
# IMPORTS
###################################################################################################################

from pathlib import Path
import time
import os
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

from networks.generator import autoencoder
from networks.discriminator import convolutional_classifier

from utilities.dataset.load_files import get_file_lists
from utilities.dataset.make_dataset import make_dataset


###################################################################################################################
# GPU PARAMETERS
###################################################################################################################

# Memory growth mode
gpu_index = 0

gpus = tf.config.list_physical_devices('GPU')

tf.config.set_visible_devices(gpus[gpu_index], 'GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Set memory mode
# Select gpu to use (old=1, new=0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.compat.v1.ConfigProto()

# if os.environ["CUDA_VISIBLE_DEVICES"]=="1":
#     config.gpu_options.per_process_gpu_memory_fraction = 0.30
# else: 
#     config.gpu_options.per_process_gpu_memory_fraction = 0.30
# set_session(tf.compat.v1.Session(config=config))


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
dataset_train = make_dataset(list_X_train, list_Y_train, batch_size=BATCH_SIZE)
dataset_val = make_dataset(list_X_val, list_Y_val, batch_size=BATCH_SIZE)
print("Done.")


###################################################################################################################
# DEFINE MODELS
###################################################################################################################

input_tuple = (4096,)

generator_c2n = autoencoder(input_tuple=input_tuple)
generator_n2c = autoencoder(input_tuple=input_tuple)

discriminator_c = convolutional_classifier(input_tuple=input_tuple)
discriminator_n = convolutional_classifier(input_tuple=input_tuple)

# generator_c2n.summary()
# generator_n2c.summary()


###################################################################################################################
# DEFINE LOSS FUNCTIONS
###################################################################################################################

LAMBDA = 10

binary_crossentropy_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False) # True or false?

@tf.function
def discriminator_loss(real_pulses, fake_pulses):
    clean_loss = binary_crossentropy_loss_fn(tf.ones_like(real_pulses), real_pulses)
    noisy_loss = binary_crossentropy_loss_fn(tf.zeros_like(fake_pulses), fake_pulses)
    loss = clean_loss + noisy_loss
    return loss * 0.5

@tf.function
def generator_loss(generated_predictions):
    loss = binary_crossentropy_loss_fn(tf.ones_like(generated_predictions), 
                                       generated_predictions)
    return loss

@tf.function
def cycle_consistency_loss(real_pulses, cycled_pulses):
    loss = tf.reduce_mean(tf.abs(real_pulses - cycled_pulses))
    return loss * LAMBDA

@tf.function
def identity_loss(real_pulses, same_pulses):
    loss = tf.reduce_mean(tf.abs(real_pulses - same_pulses))
    return loss * LAMBDA * 0.5


###################################################################################################################
# DEFINE OPTIMIZERS
###################################################################################################################

generator_n2c_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.8)
generator_c2n_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.8)

discriminator_n_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.8)
discriminator_c_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.8)


###################################################################################################################
# DEFINE TRAINING STEP
###################################################################################################################

@tf.function
def train_step(real_clean, real_noisy):

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
    
    losses = {
        "gen_n2c_loss": gen_n2c_loss,
        "gen_c2n_loss": gen_c2n_loss,
        "total_cycle_loss": total_cycle_loss,
        "n2c_identity_loss": n2c_identity_loss,
        "c2n_identity_loss": c2n_identity_loss,
        "total_gen_n2c_loss": total_gen_n2c_loss,
        "total_gen_c2n_loss": total_gen_c2n_loss,
        "disc_c_loss": disc_c_loss,
        "disc_n_loss": disc_n_loss
    }
    
    return losses


###################################################################################################################
# TRAIN THE MODEL
###################################################################################################################

epochs = 100
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    
    losses_dict = {
        "gen_n2c_loss": [],
        "gen_c2n_loss": [],
        "total_cycle_loss": [],
        "n2c_identity_loss": [],
        "c2n_identity_loss": [],
        "total_gen_n2c_loss": [],
        "total_gen_c2n_loss": [],
        "disc_c_loss": [],
        "disc_n_loss": [],
    }

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(dataset_train):
        temp_losses = train_step(x_batch_train, y_batch_train)
        
        for key in losses_dict.keys():
            losses_dict[key].append(temp_losses[key].numpy())
            
        # Log every 200 batches.
        if step % 300 == 0:
            print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))

    # Save models every epoch
    model_save_path = "./saved_models/"
    generator_n2c.save(model_save_path + f"generator_n2c_{epoch}")
    generator_c2n.save(model_save_path + f"generator_c2n_{epoch}")
    discriminator_c.save(model_save_path + f"discriminator_c_{epoch}")
    discriminator_n.save(model_save_path + f"discriminator_n_{epoch}")
    
    # Save the updated losses
    with open(model_save_path+f"losses_{epoch}.pkl", "wb") as f:
        pickle.dump(losses_dict, f)
    
    print("Time taken: %.2fs" % (time.time() - start_time))
    
    
    
    