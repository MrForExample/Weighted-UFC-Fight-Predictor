import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

import pandas as pd
import numpy as np
import math
from pathlib import Path
from os import path
import FightPredictor_Common as fc
from Code.Settings import global_common as gc
from Code.Settings import tuner_parameters as tp

class fight_predict_hyper_model(kt.HyperModel):
    def __init__(self, fighter_input_shape, fight_input_shape, output_size, hidden_layers_range, 
                weight_initializer_choise, hidden_activation_choise, output_activation, output_name, output_losses, losses_weight, 
                metrics, learning_rates, decay_steps, dropout_rates, l2_factors, model_name):
        self.fighter_input_shape = fighter_input_shape
        self.fight_input_shape = fight_input_shape
        self.output_size = output_size
        self.hidden_layers_range = hidden_layers_range
        self.weight_initializer_choise = weight_initializer_choise
        self.hidden_activation_choise = hidden_activation_choise
        self.output_activation = output_activation
        self.model_name = model_name
        self.output_name = [self.model_name + '_' + name for name in output_name]
        self.output_losses = {name_key: loss_func for name_key, loss_func in zip(self.output_name, output_losses)}
        self.losses_weight = {name_key: loss_weight for name_key, loss_weight in zip(self.output_name, losses_weight)}
        self.metrics = {name_key: metric for name_key, metric in zip(self.output_name, metrics)}
        self.learning_rates = learning_rates
        self.decay_steps = decay_steps
        self.dropout_rates = dropout_rates
        self.l2_factors = l2_factors

    def mlp(self, hp, input_shape=None, input_layer=None, name=''):
        assert input_shape != None or input_layer != None, "Must define input layer or it shape!"
        # Build input layer
        if input_layer == None and input_shape != None:
            input_layer = keras.Input(shape=input_shape)
        h = input_layer
        # Build hidden layer
        layer_num = hp.Int(name + '_layer_num', self.hidden_layers_range['min_layer_num'], self.hidden_layers_range['max_layer_num'], step=self.hidden_layers_range['layer_step'])
        for i in range(layer_num):
            weight_init = hp.Choice(name + "_weight_init_" + str(i), self.weight_initializer_choise)
            hidden_activation = hp.Choice(name + "_hidden_activation_" + str(i), self.hidden_activation_choise)
            num_hidden = hp.Int(name + '_num_hidden_' + str(i), self.hidden_layers_range['min_unit_num'], self.hidden_layers_range['max_unit_num'], step=self.hidden_layers_range['unit_step'])
            l2_factor = hp.Float(name + '_l2_factor_' + str(1), self.l2_factors['min_l2'], self.l2_factors['max_l2'], default=self.l2_factors['l2_default'])
            h = keras.layers.Dense(units=num_hidden, activation=hidden_activation, 
                                kernel_initializer=weight_init, kernel_regularizer=keras.regularizers.l2(l2_factor), name=self.model_name + '_' + name + "_dense_" + str(i))(h)

            if hp.Choice(name + "_add_batch_norm_" + str(i), [True, False]):
                h = keras.layers.BatchNormalization(name=self.model_name + '_' + name + "_batch_norm_" + str(i))(h)

            dropout_rate = hp.Float(name + '_dropout_rate_' + str(i), self.dropout_rates['min_dr'], self.dropout_rates['max_dr'], step=self.dropout_rates['dr_step'], default=self.dropout_rates['dr_default']) 
            h = keras.layers.Dropout(dropout_rate, name=self.model_name + '_' + name + "_dropout_" + str(i))(h)

        return h, input_layer

    def build(self, hp):
        # Input MLP
        fighter_input_shape = (self.fighter_input_shape,)
        fighter_input_0 = keras.Input(shape=fighter_input_shape)
        fighter_input_1 = keras.Input(shape=fighter_input_shape)

        fighter_model_name = 'fighter'
        fight_model_name = 'fight'
        last_latent_layer_fighter, input_layer_fighter = self.mlp(hp, input_shape=fighter_input_shape, name=fighter_model_name)
        last_latent_layer_fight, input_layer_fight = self.mlp(hp, input_shape=(self.fight_input_shape), name=fight_model_name)
        fighter_input_model = keras.Model(inputs=input_layer_fighter, outputs=last_latent_layer_fighter, name=self.model_name + '_' + fighter_model_name)
        fight_input_model = keras.Model(inputs=input_layer_fight, outputs=last_latent_layer_fight, name=self.model_name + '_' + fight_model_name)

        fighter_output_0 = fighter_input_model(fighter_input_0)
        fighter_output_1 = fighter_input_model(fighter_input_1)
        concatenate_latent_layer = keras.layers.concatenate([fighter_output_0, fighter_output_1, fight_input_model.output], 
                                                            name=self.model_name + '_concatenate')

        # Middle MLP
        middle_latent_layer, _ = self.mlp(hp, input_layer=concatenate_latent_layer, name='middle')

        # Output layer
        output_layers = []
        for i in range(len(self.output_size)):
            last_latent_layer, _ = self.mlp(hp, input_layer=middle_latent_layer, name='last_' + str(i))

            weight_init = hp.Choice(self.output_name[i] + "_weight_init_" + str(i), self.weight_initializer_choise)
            l2_factor = hp.Float(self.output_name[i] + '_l2_factor_' + str(1), self.l2_factors['min_l2'], self.l2_factors['max_l2'], default=self.l2_factors['l2_default'])
            output_layers.append(keras.layers.Dense(self.output_size[i], self.output_activation[i], 
                                kernel_initializer=weight_init, kernel_regularizer=keras.regularizers.l2(l2_factor), name=self.output_name[i])(last_latent_layer))

        model = keras.Model(inputs=[fighter_input_0, fighter_input_1, fight_input_model.input], outputs=output_layers, name=self.model_name)

        lr = hp.Float('learning_rate', self.learning_rates['min_lr'], self.learning_rates['max_lr'], sampling='log')
        dr = hp.Int('decay_rate', self.learning_rates['min_decay_rate'], self.learning_rates['max_decay_rate'], step=self.learning_rates['decay_step'])
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(lr, decay_steps=self.decay_steps, decay_rate=dr)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss=self.output_losses, loss_weights=self.losses_weight, metrics=self.metrics)
        return model

def training_model(current_time):
    
    best_hypermodel_path = gc.LATEST_CHECKPOINT_DIR + "/best_hypermodel"
    if path.exists(best_hypermodel_path):
        # Model
        model = fc.load_saved_model(best_hypermodel_path)
        model_name = model.name
        saved_model_dir = gc.LATEST_CHECKPOINT_DIR
        # Data
        train_features, train_labels = fc.load_data_for_model(model_name, gc.OUTPUT_NAME)
        steps_per_epoch = int(math.ceil((1 - gc.VALIDATION_SPLIT) * train_features[0].shape[0] / gc.BATCH_SIZE))

        latest = tf.train.latest_checkpoint(gc.LATEST_CHECKPOINT_DIR)
        if latest is not None:
            model.load_weights(latest)
    else:
        model_name = "fight_predict_hyper_model-" + current_time
        saved_model_dir = "./saved_models/" + model_name
        Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
        # Data
        train_features, train_labels = fc.load_data_for_model(model_name, gc.OUTPUT_NAME)
        # Model
        steps_per_epoch = int(math.ceil((1 - gc.VALIDATION_SPLIT) * train_features[0].shape[0] / gc.BATCH_SIZE))
        fphm = create_model(train_features, train_labels, steps_per_epoch, model_name)

        tuner = kt.Hyperband(fphm , objective='val_loss', max_epochs=gc.MAX_TUNE_EPOCH, factor=3, hyperband_iterations=3, 
                            directory='tuner_dir', project_name=model_name)  
        
        tuner.search(train_features, train_labels, batch_size=gc.BATCH_SIZE, epochs=gc.MAX_TUNE_EPOCH, validation_split=gc.VALIDATION_SPLIT, verbose=0)
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        model = tuner.hypermodel.build(best_hps)

        model.save(best_hypermodel_path)

    #model.summary()
    keras.utils.plot_model(model, saved_model_dir + "/FPM_H_model_structure.png", show_shapes=True)

    # Training
    fc.fit_model(model, train_features, train_labels, steps_per_epoch, model_name, saved_model_dir, gc)

def create_model(inputs, labels, steps_per_epoch, model_name):
    output_size = [labels[name].shape[1] for name in labels]
    return fight_predict_hyper_model(inputs[0].shape[1], inputs[2].shape[1], output_size, tp.hidden_layers_range, 
                tp.weight_initializer_choise, tp.hidden_activation_choise, gc.output_activation, gc.OUTPUT_NAME, gc.output_losses, gc.losses_weight, 
                gc.metrics, tp.learning_rates_range, steps_per_epoch * gc.MAX_EPOCH, tp.dropout_rate_range, tp.l2_factor_range, model_name) 

if __name__ == "__main__":
    fc.execute_model_action(gc, training_model)

