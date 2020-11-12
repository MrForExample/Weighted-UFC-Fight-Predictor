import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

from pathlib import Path
import os
import FightPredictor_Common as fc
from Code.Settings import global_common as gc

# Bootstrap aggregating (Bagging) model
class fight_predict_ensemble_model:
    def __init__(self, all_models, output_name, output_losses, losses_weight, metrics, model_name='fight_predict_ensemble_model'):
        self.all_models = all_models
        self.model_name = model_name
        self.output_name = [self.model_name + '_' + name for name in output_name]
        self.output_losses = {name_key: loss_func for name_key, loss_func in zip(self.output_name, output_losses)}
        self.losses_weight = {name_key: loss_weight for name_key, loss_weight in zip(self.output_name, losses_weight)}
        self.metrics = {name_key: metric for name_key, metric in zip(self.output_name, metrics)}
        

        self.model = self.build_model()
        self.model.compile(loss=self.output_losses, loss_weights=self.losses_weight, metrics=self.metrics)

    def build_model(self):
        fighter_input_shape = (self.all_models[0].input[0].shape[1], )
        fight_input_shape = (self.all_models[0].input[2].shape[1], )

        fighter_input_0 = keras.Input(shape=fighter_input_shape)
        fighter_input_1 = keras.Input(shape=fighter_input_shape)
        fight_input = keras.Input(shape=fight_input_shape)

        models_output = []
        for model in self.all_models:
            models_output.append(model([fighter_input_0, fighter_input_1, fight_input]))

        output_layers = [keras.layers.Average(name=self.output_name[i])([output[i] for output in models_output]) for i in range(len(self.output_name))]
        try:
            ensemble_model = keras.Model(inputs=[fighter_input_0, fighter_input_1, fight_input], outputs=output_layers, name=self.model_name)
        except ValueError:
            for i_m in range(len(self.all_models)):
                model = self.all_models[i_m]
                model._name = 'model_' + str(i_m)
                for i_l, layer in enumerate(model.layers):
                    layer._name = model.name + '_layer_' + str(i_l)            
            ensemble_model = keras.Model(inputs=[fighter_input_0, fighter_input_1, fight_input], outputs=output_layers, name=self.model_name)
        
        return ensemble_model
    
def get_all_models():
    all_models = []
    models_root_dir = "./saved_models/"
    if os.path.exists(models_root_dir):
        all_models_dir_name = [name for name in os.listdir(models_root_dir)]
        for name in all_models_dir_name:
            model_dir = models_root_dir + name + "/best_saved_model"
            if os.path.exists(model_dir):
                model = fc.load_saved_model(model_dir)
                all_models.append(model)
    return all_models

def evaluate_and_save_ensemble_model():
    fpem = fight_predict_ensemble_model(get_all_models(), gc.OUTPUT_NAME, gc.output_losses, gc.losses_weight, gc.metrics)
    model = fpem.model
    #self.model.summary()
    keras.utils.plot_model(model, "./saved_models/FPM_E_model_structure.png", show_shapes=True)

    fc.evaluate_model(model, gc, fpem.model_name)

    saved_model_dir = "./saved_models/" + fpem.model_name
    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
    model.save(saved_model_dir + "/best_saved_model")

if __name__ == "__main__":
    fc.execute_model_action(gc, evaluate_and_save_ensemble_model)