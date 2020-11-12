import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import math
from pathlib import Path
from os import path
import FightPredictor_Common as fc
from Code.Settings import global_common as gc
from Code.Settings import manual_parameters as mp

class fight_predict_model:
    def __init__(self, fighter_input_shape, fight_input_shape, output_size, fighter_hidden_layers, fight_hidden_layers, middle_hidden_layers, last_hidden_layers, 
                weight_init, hidden_activation, output_activation, output_name, output_losses, losses_weight, metrics, lr_schedule, dropout_rate, l2_factor, model_name):
        self.fighter_input_shape = fighter_input_shape
        self.fight_input_shape = fight_input_shape
        self.output_size = output_size
        self.fighter_hidden_layers = fighter_hidden_layers
        self.fight_hidden_layers = fight_hidden_layers
        self.middle_hidden_layers = middle_hidden_layers
        self.last_hidden_layers = last_hidden_layers
        self.weight_init = weight_init
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.model_name = model_name
        self.output_name = [self.model_name + '_' + name for name in output_name]
        self.output_losses = {name_key: loss_func for name_key, loss_func in zip(self.output_name, output_losses)}
        self.losses_weight = {name_key: loss_weight for name_key, loss_weight in zip(self.output_name, losses_weight)}
        self.metrics = {name_key: metric for name_key, metric in zip(self.output_name, metrics)}
        self.lr_schedule = lr_schedule
        self.dropout_rate = dropout_rate
        self.l2_factor = l2_factor
        
        self.model = self.build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.model.compile(optimizer=self.optimizer, loss=self.output_losses, loss_weights=self.losses_weight, metrics=self.metrics)

    def mlp(self, hidden_layers_shape, input_shape=None, input_layer=None, name=''):
        assert input_shape != None or input_layer != None, "Must define input layer or it shape!"
        # Build input layer
        if input_layer == None and input_shape != None:
            input_layer = keras.Input(shape=input_shape)
        h = input_layer
        # Build hidden layer
        for i in range(len(hidden_layers_shape)):
            h = keras.layers.Dense(units=hidden_layers_shape[i], activation=self.hidden_activation, 
                                kernel_initializer=self.weight_init, kernel_regularizer=keras.regularizers.l2(self.l2_factor), 
                                name=self.model_name + '_' + name + "_dense_" + str(i))(h)
            h = keras.layers.BatchNormalization(name=self.model_name + '_' + name + "_batch_norm_" + str(i))(h)
            h = keras.layers.Dropout(self.dropout_rate, name=self.model_name + '_' + name + "_dropout_" + str(i))(h)          
        return h, input_layer

    def build_model(self):
        # Input MLP
        fighter_input_shape = (self.fighter_input_shape,)
        fighter_input_0 = keras.Input(shape=fighter_input_shape)
        fighter_input_1 = keras.Input(shape=fighter_input_shape)

        fighter_model_name = 'fighter'
        fight_model_name = 'fight'
        last_latent_layer_fighter, input_layer_fighter = self.mlp(self.fighter_hidden_layers, input_shape=fighter_input_shape, name=fighter_model_name)
        last_latent_layer_fight, input_layer_fight = self.mlp(self.fight_hidden_layers, input_shape=(self.fight_input_shape,), name=fight_model_name)
        fighter_input_model = keras.Model(inputs=input_layer_fighter, outputs=last_latent_layer_fighter, name=self.model_name + '_' + fighter_model_name)
        fight_input_model = keras.Model(inputs=input_layer_fight, outputs=last_latent_layer_fight, name=self.model_name + '_' + fight_model_name)

        fighter_output_0 = fighter_input_model(fighter_input_0)
        fighter_output_1 = fighter_input_model(fighter_input_1)
        concatenate_latent_layer = keras.layers.concatenate([fighter_output_0, fighter_output_1, fight_input_model.output], 
                                                        name=self.model_name + '_concatenate')

        # Middle MLP
        middle_latent_layer, _ = self.mlp(self.middle_hidden_layers, input_layer=concatenate_latent_layer, name='middle')

        # Output layer
        output_layers = []
        for i in range(len(self.output_size)):
            last_latent_layer, _ = self.mlp(self.last_hidden_layers[i], input_layer=middle_latent_layer, name='last_' + str(i))
            output_layers.append(keras.layers.Dense(self.output_size[i], self.output_activation[i], 
                                kernel_initializer=self.weight_init, kernel_regularizer=keras.regularizers.l2(self.l2_factor), name=self.output_name[i])(last_latent_layer))

        return keras.Model(inputs=[fighter_input_0, fighter_input_1, fight_input_model.input], outputs=output_layers, name=self.model_name)

def training_model(current_time):
    best_model_path = gc.LATEST_CHECKPOINT_DIR + "/best_saved_model"
    if path.exists(best_model_path):
        # Model
        model = fc.load_saved_model(best_model_path)
        model_name = model.name
        saved_model_dir = gc.LATEST_CHECKPOINT_DIR
        # Data
        train_features, train_labels = fc.load_data_for_model(model_name, gc.OUTPUT_NAME)
        steps_per_epoch = int(math.ceil((1 - gc.VALIDATION_SPLIT) * train_features[0].shape[0] / gc.BATCH_SIZE))
    else :
        model_name = "fight_predict_model-" + current_time
        saved_model_dir = "./saved_models/" + model_name
        Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
        # Data
        train_features, train_labels = fc.load_data_for_model(model_name, gc.OUTPUT_NAME)
        # Model
        steps_per_epoch = int(math.ceil((1 - gc.VALIDATION_SPLIT) * train_features[0].shape[0] / gc.BATCH_SIZE))
        fpm = create_model(train_features, train_labels, steps_per_epoch, model_name)
        model = fpm.model

        latest = tf.train.latest_checkpoint(gc.LATEST_CHECKPOINT_DIR)
        if latest is not None:
            model.load_weights(latest)

    #model.summary()
    keras.utils.plot_model(model, saved_model_dir + "/FPM_M_model_structure.png", show_shapes=True)

    # Training
    fc.fit_model(model, train_features, train_labels, steps_per_epoch, model_name, saved_model_dir, gc)

def create_model(inputs, labels, steps_per_epoch, model_name):
    output_size = [labels[name].shape[1] for name in labels]
    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(mp.learning_rate, decay_steps=steps_per_epoch * gc.MAX_EPOCH, decay_rate=mp.decay_rate)
    return fight_predict_model(inputs[0].shape[1], inputs[2].shape[1], output_size, mp.fighter_hidden_layers, mp.fight_hidden_layers, mp.middle_hidden_layers, mp.last_hidden_layers, 
                            mp.weight_init, mp.hidden_activation, gc.output_activation, gc.OUTPUT_NAME, gc.output_losses, gc.losses_weight, gc.metrics, lr_schedule, mp.dropout_rate, mp.l2_factor, model_name) 

if __name__ == "__main__":
    fc.execute_model_action(gc, training_model)