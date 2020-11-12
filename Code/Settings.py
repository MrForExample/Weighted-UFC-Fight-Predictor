from Code.ML_Modeling.FightPredictor_Common import masked_binary_crossentropy, masked_mse_loss, masked_r2_loss, OverallCategoricalAccuracy, OverallForcePickCategoricalAccuracy, masked_mae_accuracy
class global_common:
    model_load_name = "fight_predict_ensemble_model"

    # Set to load specific checkpoint for model
    LATEST_CHECKPOINT_DIR = "./saved_models/" + model_load_name
    MAX_EPOCH = 1000
    MAX_TUNE_EPOCH = 10
    BATCH_SIZE = 128
    OUTPUT_NAME = ['distribution_output', 'percentage_output', 'numerical_output']
    ES_PATIENCE = MAX_EPOCH
    SAVE_FREQ = 100
    VALIDATION_SPLIT = 0.2

    output_losses = ['categorical_crossentropy', masked_binary_crossentropy, masked_r2_loss]
    losses_weight = [1, 1, 1]
    #losses_weight = [1, 0.25, 0.25]
    metrics = [['accuracy', OverallCategoricalAccuracy(), OverallForcePickCategoricalAccuracy()], masked_mae_accuracy, masked_mae_accuracy]
    output_activation = ['softmax', 'sigmoid', 'relu']

    # train, evaluate, predict
    model_action = 'predict'
    # using in evaluate or predict to load the model
    model_load_path = "./saved_models/" + model_load_name + "/best_saved_model"

class manual_parameters:
    '''
    fighter_hidden_layers = [1024, 768, 512]
    fight_hidden_layers = [1024, 512]
    middle_hidden_layers = [2048, 1024]
    last_hidden_layers = [[1024, 768, 512], [512, 256], [512, 256]]
    weight_init = 'random_normal'
    hidden_activation = 'relu'
    learning_rate = 6e-3
    decay_rate = 10
    dropout_rate = 0.5
    l2_factor = 8e-5
    '''    
    '''
    fighter_hidden_layers = [1024, 768, 512]
    fight_hidden_layers = [768, 512]
    middle_hidden_layers = [2560, 2048, 1536]
    last_hidden_layers = [[512, 256], [512], [512]]
    weight_init = 'random_normal'
    hidden_activation = 'relu'
    learning_rate = 3e-3
    decay_rate = 6
    dropout_rate = 0.5
    l2_factor = 1e-4
    '''
    fighter_hidden_layers = [1024, 512, 256]
    fight_hidden_layers = [512, 256]
    middle_hidden_layers = [2048, 1536, 1024]
    last_hidden_layers = [[1024, 512, 256], [512, 256], [512, 256]]
    weight_init = 'glorot_normal'
    hidden_activation = 'relu'
    learning_rate = 2e-3
    decay_rate = 5
    dropout_rate = 0.5
    l2_factor = 1e-4
    
class tuner_parameters:
    hidden_layers_range = {'min_unit_num': 256, 'max_unit_num': 2048, 'min_layer_num': 1, 'max_layer_num': 3, 'unit_step': 256, 'layer_step': 1}
    learning_rates_range = {'min_lr': 1e-3, 'max_lr': 9e-3, 'min_decay_rate': 4, 'max_decay_rate': 20, 'decay_step': 4}
    dropout_rate_range = {'min_dr': 0, 'max_dr': 0.65, 'dr_step': 0.05, 'dr_default': 0.5}
    l2_factor_range = {'min_l2': 0, 'max_l2': 1e-2, 'l2_default': 1e-4}
    hidden_activation_choise = ['relu', 'elu', 'swish']
    weight_initializer_choise = ['glorot_normal', 'random_normal', 'random_uniform']

class build_settings:
    # Build features/labels for model to train or build features for model to predict
    is_build_train = False
    # Proportion of test dataset in entire dataset, using when split train/test dataset
    test_set_frac = 0.2
    # Chunks sample method
    divide_by_chunks = True 
    chunk_size = 100   
    # Fighter's gender in fight
    is_man = True
    fighter_0_ids = ['032cc3922d871c7f', '9e8f6c728eb01124']
    fighter_1_ids = ['6506c1d34da9c013', 'f4c49976c75c5ab2']
    # Fight weight class shorthand: STR, FLY, BAN, FEA, LIG, WEL, MID, LIGHEA, HEA
    fight_weights = ['LIG', 'LIG']
    fighter_0_ages=[]
    fighter_1_ages=[]
    fight_date=None