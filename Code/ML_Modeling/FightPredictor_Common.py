import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.metrics import categorical_accuracy

import pandas as pd
import numpy as np
import datetime
import json
from pathlib import Path

def overall_accuracy(y_true, y_pred):
    y_overall_true = tf.map_fn(get_overall_odds, y_true)
    y_overall_pred = tf.map_fn(get_overall_odds, y_pred)
    return categorical_accuracy(y_overall_true, y_overall_pred)

def get_overall_odds(v):
    d_output_shape = v.get_shape()[0]
    odds = [v[0] + v[2] + v[4], v[1] + v[3] + v[5]]
    if d_output_shape > 6:
        odds_2 = 0
        for d_output_i in range(6, d_output_shape):
            odds_2 += v[d_output_i]
        odds.append(odds_2)
    return tf.convert_to_tensor(odds, dtype=tf.float32)

class OverallCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name='overall_accuracy', dtype=None):
        super(OverallCategoricalAccuracy, self).__init__(overall_accuracy, name, dtype=dtype)

def overall_force_pick_accuracy(y_true, y_pred):
    y_overall_true = tf.map_fn(get_over_force_pick_all_odds, y_true)
    y_overall_pred = tf.map_fn(get_over_force_pick_all_odds, y_pred)
    return categorical_accuracy(y_overall_true, y_overall_pred)

def get_over_force_pick_all_odds(v):
    odds = [v[0] + v[2] + v[4], v[1] + v[3] + v[5], 0]
    return tf.convert_to_tensor(odds, dtype=tf.float32)

class OverallForcePickCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name='overall_force_pick_accuracy', dtype=None):
        super(OverallForcePickCategoricalAccuracy, self).__init__(overall_force_pick_accuracy, name, dtype=dtype)

def get_masked_value(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return y_true, y_pred

def masked_binary_crossentropy(y_true, y_pred):
    y_true, y_pred = get_masked_value(y_true, y_pred)
    loss = K.binary_crossentropy(y_true, y_pred)
    return loss

def masked_mse_loss(y_true, y_pred):
    y_true, y_pred = get_masked_value(y_true, y_pred)
    loss = tf.reduce_mean(tf.square(y_true-y_pred))
    return loss

def masked_mae_loss(y_true, y_pred):
    y_true, y_pred = get_masked_value(y_true, y_pred)
    loss = tf.reduce_mean(tf.abs(y_true-y_pred)) 
    return loss

def masked_r2_loss(y_true, y_pred):
    y_true, y_pred = get_masked_value(y_true, y_pred)
    SS_res =  tf.reduce_mean(tf.square(y_true-y_pred)) 
    SS_tot =tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true))) 
    loss = SS_res/(SS_tot + K.epsilon())
    return loss

def masked_mae_accuracy(y_true, y_pred):
    y_true, y_pred = get_masked_value(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.abs(y_true-y_pred))
    return accuracy

def execute_model_action(gc, training_model):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if gc.model_action == 'train':
        training_model(current_time)
    else :
        model = load_saved_model(gc.model_load_path)
        if gc.model_action == 'evaluate':
            evaluate_model(model, gc, model.name)
        elif gc.model_action == 'predict':
            predict_fight(model)

def predict_fight(model):
    # Data
    fight_predict_df = pd.read_csv("./Data/FightNeedPredictData.csv", index_col=0)
    predict_features = split_features(fight_predict_df)
    # Prediction
    predictions = model.predict(predict_features)

    output_col_id = [r'_D_TAR_', r'_P_TAR_', r'_N_TAR_']
    dataset = pd.read_csv("./Data/FightAllTrainTestData.csv", index_col=0)

    All_output_names = []
    for i in range(len(output_col_id)):
        All_output_names += list(dataset.filter(regex=output_col_id[i]).columns)
    #print(All_output_names)
    predict_fight_num = len(fight_predict_df.index)
    predict_result_value = [[] for _ in range(predict_fight_num)]
    for predict in predictions:
        predict = predict.tolist()
        for i in range(len(predict)):
            predict_result_value[i] += predict[i]

    predict_result = {k: [] for k in All_output_names}
    for i in range(len(predict_result_value)):
        for j in range(len(All_output_names)):
            predict_result[All_output_names[j]].append(predict_result_value[i][j])

    # Save result
    result_log_dir = "./logs/predict/"
    Path(result_log_dir).mkdir(parents=True, exist_ok=True)
    with open(result_log_dir + model.name, "w") as outfile:  
        json.dump(predict_result, outfile, indent=4) 

def evaluate_model(model, gc, model_name):
    # Data
    test_features, test_labels = load_data_for_model(model.name, gc.OUTPUT_NAME, is_training=False)
    # Result
    result = model.evaluate(test_features, test_labels, batch_size=gc.BATCH_SIZE, verbose=1)
    result = dict(zip(model.metrics_names, result))
    print(result)
    # Save result
    evaluate_log_dir = "./logs/evaluate/"
    Path(evaluate_log_dir).mkdir(parents=True, exist_ok=True)
    with open(evaluate_log_dir + model_name, "w") as outfile:  
        json.dump(result, outfile, indent=4) 

def fit_model(model, train_features, train_labels, steps_per_epoch, model_name, saved_model_dir, gc):
    # Tensorboard fit log
    fit_log_dir = "./logs/fit/" + model_name
    tb_callback = keras.callbacks.TensorBoard(log_dir=fit_log_dir, histogram_freq=1)

    # Create a callback that saves the model's weights periodically, Include the epoch in the file name (uses `str.format`)
    checkpoint_path = saved_model_dir + "/cp-{epoch:04d}.ckpt"
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq=steps_per_epoch*gc.SAVE_FREQ)
    early_stopping = keras.callbacks.EarlyStopping(patience=gc.ES_PATIENCE)
    try:
        model.fit(train_features, train_labels, batch_size=gc.BATCH_SIZE, epochs=gc.MAX_EPOCH, validation_split=gc.VALIDATION_SPLIT, 
                            callbacks=[tb_callback, cp_callback, early_stopping], verbose=2)
    except KeyboardInterrupt:
        pass
    model.save(saved_model_dir + "/best_saved_model")

    evaluate_model(model, gc, model_name)

def load_saved_model(model_dir):
    return keras.models.load_model(model_dir, 
                            custom_objects={'masked_binary_crossentropy': masked_binary_crossentropy,
                                            'masked_r2_loss': masked_r2_loss,
                                            'OverallCategoricalAccuracy': OverallCategoricalAccuracy,
                                            'OverallForcePickCategoricalAccuracy': OverallForcePickCategoricalAccuracy,
                                            'masked_mae_accuracy': masked_mae_accuracy})

def load_data_for_model(model_name, output_name, is_training=True):
    if is_training:
        dataset_path = "./Data/FightAllTrainData.csv"
    else :
        dataset_path = "./Data/FightAllTestData.csv"
    
    dataset = pd.read_csv(dataset_path, index_col=0)
    dataset = dataset.sample(frac=1) 

    features = split_features(dataset)
    labels = split_labels(model_name, output_name, dataset)
    return features, labels

def split_features(dataset):
    all_features = dataset.filter(regex=r'^((?!_TAR_).)*$')
    input_col_rex = [r'_0', r'_1', r'^((?!_[01]).)*$']
    features = [all_features.filter(regex=rex).to_numpy(dtype=np.float) for rex in input_col_rex]
    return features

def split_labels(model_name, output_name, dataset):
    output_col_id = [r'_D_TAR_', r'_P_TAR_', r'_N_TAR_']
    #print([list(dataset.filter(regex=output_col_id[i]).columns) for i in range(3)])
    labels = {model_name + '_' + output_name[i]: dataset.filter(regex=output_col_id[i]).to_numpy(dtype=np.float) for i in range(len(output_name))}
    return labels