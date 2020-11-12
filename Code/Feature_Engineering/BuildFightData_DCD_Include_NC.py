import pandas as pd
import numpy as np
import calendar
import re
import datetime
import dateutil
from ast import literal_eval

from os import path
import pickle
from Code.Settings import build_settings as bs

def init_columns_name(last_words, base_columns):
    new_columns_name = []
    for col in base_columns:
        for i in range(2):
            new_columns_name.append(col + last_words + str(i))
    return new_columns_name

# Define some global average columns, columns that related to other fighter's physical attributes
state_data_columns = ['TKO_WIN_%', 'SUB_WIN_%', 'DEC_WIN_%']
state_side_columns = ['TKO_LOSS_%', 'SUB_LOSS_%', 'DEC_LOSS_%']
fighter_dynamic_columns = ['AGE', 'FIGHT_COUNT', 'FIGHT_MINUTE', 'REV_I', 'REV_P', 'DEC_DRAW_%', 'STANCE_ORT', 'STANCE_SOU', 'STANCE_SWI']
fighter_static_columns = ['REACH', 'HEIGHT']

fight_state_data_columns = init_columns_name('_', state_data_columns)
fight_state_side_columns = init_columns_name('_', state_side_columns)
fight_fighter_dynamic_columns = init_columns_name('_', fighter_dynamic_columns)
fight_fighter_static_columns = init_columns_name('_', fighter_static_columns)
fight_other_data_columns = ['MAX_FIGHT_TIME', 'WOMEN', 'STR', 'FLY', 'BAN', 'FEA', 'LIG', 'WEL', 'MID', 'LIGHEA', 'HEA']

fight_input_columns = fight_state_data_columns + fight_state_side_columns + fight_fighter_dynamic_columns + fight_fighter_static_columns + fight_other_data_columns

# Define weight value columns and they related average and per minute columns
need_weight_data_columns = ['KD', 'SIG_STR', 'TOTAL_STR', 'TD', 'SUB_ATT', 'CTRL', 'HEAD', 'BODY', 'LEG', 'DISTANCE', 'CLINCH', 'GROUND']
need_weight_side_columns = ['KD_DEF', 'SIG_DEF', 'TOTAL_DEF', 'TD_DEF', 'SUB_ATT_DEF', 'CTRL_DEF', 'HEAD_DEF', 'BODY_DEF', 'LEG_DEF', 'DISTANCE_DEF', 'CLINCH_DEF', 'GROUND_DEF']

fight_weighted_data_columns = init_columns_name('_W_', need_weight_data_columns)
fight_weighted_side_columns = init_columns_name('_W_', need_weight_side_columns)
fight_avg_data_columns = init_columns_name('_AVG_', need_weight_data_columns)
fight_avg_side_columns = init_columns_name('_AVG_', need_weight_side_columns)
fight_pm_data_columns = init_columns_name('_PM_', need_weight_data_columns)
fight_pm_side_columns = init_columns_name('_PM_', need_weight_side_columns)

fight_input_columns += fight_weighted_data_columns + fight_weighted_side_columns + fight_avg_data_columns + fight_avg_side_columns + fight_pm_data_columns + fight_pm_side_columns

init_value = dict.fromkeys(need_weight_data_columns + need_weight_side_columns, 0.5)
extra_init_value = {'KD': 0, 'KD_DEF': 1, 'SUB_ATT': 0, 'SUB_ATT_DEF': 1}
init_value.update(extra_init_value)

# Define training target columns
target_win_columns = ['TKO_WIN_%_D', 'SUB_WIN_%_D', 'DEC_WIN_%_D'] 
target_win_columns = init_columns_name('_TAR_', target_win_columns)
target_win_columns += ['DEC_DRAW_%_D_TAR_2']
target_n_raw_data_columns = ['KD_N', 'CTRL_N', 'REV_N', 'SUB_ATT_N']
target_n_data_columns = init_columns_name('_TAR_', target_n_raw_data_columns)
target_np_raw_data_columns = ['SIG_STR_N', 'SIG_STR_%_P', 'TOTAL_STR_N', 'TOTAL_STR_%_P', 'TD_N', 'TD_%_P', 
                        'HEAD_N', 'HEAD_%_P', 'BODY_N', 'BODY_%_P', 'LEG_N', 'LEG_%_P', 'DISTANCE_N', 'DISTANCE_%_P', 'CLINCH_N', 'CLINCH_%_P', 'GROUND_N', 'GROUND_%_P']
target_np_data_columns = init_columns_name('_TAR_', target_np_raw_data_columns)
target_np_data_columns += ['FIGHT_MINUTE_P_TAR_2']
fight_target_columns = target_win_columns + target_n_data_columns + target_np_data_columns

# Build training data frame
fight_train_columns = fight_input_columns + fight_target_columns
fight_train_df = pd.DataFrame(columns=fight_train_columns)
fight_train_df_path = "./Data/FightAllTrainTestData.csv"

# Load reformatted raw data frame
list_data_columns = ['KD', 'SIG_STR', 'SIG_STR_%', 'TOTAL_STR', 'TD', 'TD_%', 'SUB_ATT',
                    'REV', 'CTRL', 'HEAD', 'BODY', 'LEG', 'DISTANCE', 'CLINCH', 'GROUND']
list_data_columns = init_columns_name('_', list_data_columns)
list_data_columns = ['FIGHT_TIME', 'WIN', 'FIGHTER_ID', 'FIGHTER'] + list_data_columns

columns_converters = {}
for col in list_data_columns:
    columns_converters[col] = literal_eval

fight_reformat_df = pd.read_csv("./Data/FightReformatData.csv", converters=columns_converters, index_col=0)
fighter_reformat_df = pd.read_csv("./Data/FighterReformatData.csv", index_col=0)

fighter_build_info_path = "./Data/fighter_build_info.pickle"

def build_train_fight_data():
    print("Build train fight data\n")
    global fight_train_df
    # Store info for build weighted values: 
    # { fighter id: [ { last fight weighted values }, { all last fight numbers per minute }, { average values }, 
    # { value update count }, [ all fight count, all fight minute ] ] }
    fighter_build_info = {}
    # Fight data is ordered by date, top to buttom -> Nearest to farthest
    reverse_reformat_df = fight_reformat_df.iloc[::-1]
    for _, reformat_row in reverse_reformat_df.iterrows(): 
        new_fight_row = {}

        new_fight_row['MAX_FIGHT_TIME'] = reformat_row['FIGHT_TIME'][1]
        # Get fight weight class use one hot encoding
        fight_weight_class = reformat_row['WEIGHT_CLASS'].split(' ')
        new_fight_row['WOMEN'] = 0
        if fight_weight_class[0].upper() == 'WOMEN':
            new_fight_row['WOMEN'] = 1
            del fight_weight_class[0]
        fight_weight_class = ''.join([w.upper()[:3] for w in fight_weight_class])
        for w_col in fight_other_data_columns[2:]:
            new_fight_row[w_col] = int(w_col == fight_weight_class)

        for id_i in range(len(reformat_row['FIGHTER_ID'])):
            fighter_id = reformat_row['FIGHTER_ID'][id_i]
            # Init value for calculate weighted, average and per minute value
            if fighter_id not in fighter_build_info:
                fight_weighted_values = {}
                fight_num_per_minute = {}
                fight_avg_value = {}
                fight_value_count = {}
                for i in range(len(need_weight_data_columns)):
                    i_w = i * 2

                    fight_avg_value[fight_avg_data_columns[i_w][:-2]] = init_value[need_weight_data_columns[i]]
                    fight_avg_value[fight_avg_side_columns[i_w][:-2]] = init_value[need_weight_side_columns[i]]
                    fight_avg_value[fight_pm_data_columns[i_w][:-2]] = 0
                    fight_avg_value[fight_pm_side_columns[i_w][:-2]] = 0

                    fight_weighted_values[fight_weighted_data_columns[i_w][:-2]] = init_value[need_weight_data_columns[i]]
                    fight_weighted_values[fight_weighted_side_columns[i_w][:-2]] = init_value[need_weight_side_columns[i]]
                    fight_num_per_minute[fight_pm_data_columns[i_w][:-2]] = 0
                    fight_num_per_minute[fight_pm_side_columns[i_w][:-2]] = 0
                    fight_value_count[need_weight_data_columns[i]] = 0
                    fight_value_count[need_weight_side_columns[i]] = 0

                for i in range(len(state_data_columns)):
                    fight_avg_value[state_data_columns[i]] = 0
                    fight_avg_value[state_side_columns[i]] = 0

                fight_avg_value['DEC_DRAW_%'] = 0
                fight_avg_value['REV_I'] = 0
                fight_avg_value['REV_P'] = 0
                fight_value_count['REV_I'] = 0
                fight_value_count['REV_P'] = 0
                fighter_build_info[fighter_id] = [fight_weighted_values, fight_num_per_minute, fight_avg_value, fight_value_count, [0, 0]]

            get_needed_fighter_data(new_fight_row, fighter_id, id_i, reformat_row['DATE'])

            new_fight_row['FIGHT_COUNT_' + str(id_i)] = fighter_build_info[fighter_id][4][0]
            new_fight_row['FIGHT_MINUTE_' + str(id_i)] = fighter_build_info[fighter_id][4][1]
            fighter_build_info[fighter_id][4][0] += 1
            fighter_build_info[fighter_id][4][1] += reformat_row['FIGHT_TIME'][0]

        # Calculate weighted value, they related average and per minute for both fighter, from all they past fight up to given fight
        for id_i in range(len(reformat_row['FIGHTER_ID'])):
            fighter_id = reformat_row['FIGHTER_ID'][id_i]
            other_id_i = reverse_01(id_i)
            other_fighter_id = reformat_row['FIGHTER_ID'][other_id_i]
            # 
            for i in range(len(need_weight_data_columns)):
                i_w = i * 2 + id_i
                other_i_w = i * 2 + other_id_i
                # Store value before update
                get_needed_fight_stats_data(new_fight_row, fighter_build_info, i_w, other_i_w, fighter_id, other_fighter_id)

                num_list = reformat_row[need_weight_data_columns[i] + '_' + str(id_i)]
                if len(num_list) > 2 and num_list[1] > 0:
                    fighter_build_info[fighter_id][3][need_weight_data_columns[i]] += 1
                    fighter_build_info[other_fighter_id][3][need_weight_side_columns[i]] += 1

                    last_weighted_value = fighter_build_info[fighter_id][0][fight_weighted_data_columns[i_w][:-2]]
                    last_side_weighted_value = fighter_build_info[other_fighter_id][0][fight_weighted_side_columns[i_w][:-2]]

                    now_num_per_minute = num_list[1] / reformat_row['FIGHT_TIME'][0]
                    # Accuracy: success_num / all_num
                    now_target_weighted_value = num_list[0] / num_list[1]
                    now_expect_weighted_value = (last_weighted_value + (1 - last_side_weighted_value)) / 2
                    max_update_value = now_target_weighted_value - now_expect_weighted_value

                    all_num_per_minute_i = fighter_build_info[fighter_id][1][fight_pm_data_columns[i_w][:-2]] + now_num_per_minute
                    all_num_per_minute_p = fighter_build_info[other_fighter_id][1][fight_pm_side_columns[i_w][:-2]] + now_num_per_minute
                    fighter_build_info[fighter_id][1][fight_pm_data_columns[i_w][:-2]] = all_num_per_minute_i
                    fighter_build_info[other_fighter_id][1][fight_pm_side_columns[i_w][:-2]] = all_num_per_minute_p
                    
                    change_percent_i = now_num_per_minute / all_num_per_minute_i
                    change_percent_p = now_num_per_minute / all_num_per_minute_p

                    new_weighted_value = (last_weighted_value + max_update_value) * change_percent_i + (1 - change_percent_i) * last_weighted_value
                    new_side_weighted_value = (last_side_weighted_value + max_update_value) * change_percent_p + (1 - change_percent_p) * last_side_weighted_value

                    fighter_build_info[fighter_id][0][fight_weighted_data_columns[i_w][:-2]] = new_weighted_value
                    fighter_build_info[other_fighter_id][0][fight_weighted_side_columns[i_w][:-2]] = new_side_weighted_value

                    # Calculate average values with respect to fight, e.g average strike and defend accuracy
                    now_num_percent = num_list[0] / num_list[1]
                    now_avg_percent = get_avg_value(now_num_percent, 
                                    fighter_build_info[fighter_id][2][fight_avg_data_columns[i_w][:-2]], 
                                    fighter_build_info[fighter_id][3][need_weight_data_columns[i]])
                    fighter_build_info[fighter_id][2][fight_avg_data_columns[i_w][:-2]] = now_avg_percent
                    now_avg_percent = get_avg_value(1 - now_num_percent, 
                                    fighter_build_info[other_fighter_id][2][fight_avg_side_columns[i_w][:-2]], 
                                    fighter_build_info[other_fighter_id][3][need_weight_side_columns[i]])
                    fighter_build_info[other_fighter_id][2][fight_avg_side_columns[i_w][:-2]] = now_avg_percent
                    # Calculate average number per minute with respect to fight, e.g average strike per minute
                    now_avg_num_per_minute = get_avg_value(now_num_per_minute, 
                                    fighter_build_info[fighter_id][2][fight_pm_data_columns[i_w][:-2]], 
                                    fighter_build_info[fighter_id][3][need_weight_data_columns[i]])
                    fighter_build_info[fighter_id][2][fight_pm_data_columns[i_w][:-2]] = now_avg_num_per_minute
                    now_avg_num_per_minute = get_avg_value(now_num_per_minute, 
                                    fighter_build_info[other_fighter_id][2][fight_pm_side_columns[i_w][:-2]], 
                                    fighter_build_info[other_fighter_id][3][need_weight_side_columns[i]])
                    fighter_build_info[other_fighter_id][2][fight_pm_side_columns[i_w][:-2]] = now_avg_num_per_minute

            # Calculate average reversal and been reversal number per fight
            new_fight_row['REV_I_' + str(id_i)] = fighter_build_info[fighter_id][2]['REV_I']
            new_fight_row['REV_P_' + str(other_id_i)] = fighter_build_info[other_fighter_id][2]['REV_P']

            rev_num_list = reformat_row['REV_' + str(id_i)]
            if len(rev_num_list) > 0:
                rev_num = rev_num_list[0]
                fighter_build_info[fighter_id][3]['REV_I'] += 1
                fighter_build_info[other_fighter_id][3]['REV_P'] += 1

                now_avg_rev_i = get_avg_value(rev_num, 
                                fighter_build_info[fighter_id][2]['REV_I'], 
                                fighter_build_info[fighter_id][3]['REV_I'])
                fighter_build_info[fighter_id][2]['REV_I'] = now_avg_rev_i
                now_avg_rev_p = get_avg_value(rev_num, 
                                fighter_build_info[other_fighter_id][2]['REV_P'], 
                                fighter_build_info[other_fighter_id][3]['REV_P'])
                fighter_build_info[other_fighter_id][2]['REV_P'] = now_avg_rev_p    

            # Get numerical and percentage target values
            for i in range(len(target_n_raw_data_columns)):
                i_w = i * 2 + id_i
                num_list = reformat_row[target_n_raw_data_columns[i][:-1] + str(id_i)]
                i_n = int(target_n_raw_data_columns[i][:-2] == 'SUB_ATT')
                if len(num_list) > i_n + 1:
                    new_fight_row[target_n_data_columns[i_w]] = num_list[i_n] / reformat_row['FIGHT_TIME'][0]

            for i in range(0, len(target_np_raw_data_columns), 2):
                i_w = i * 2 + id_i
                i_p = i_w + 2
                num_list = reformat_row[target_np_raw_data_columns[i][:-1] + str(id_i)]
                if len(num_list) > 2 and num_list[1] > 0:         
                    new_fight_row[target_np_data_columns[i_w]] = num_list[1] / reformat_row['FIGHT_TIME'][0]
                    new_fight_row[target_np_data_columns[i_p]] = num_list[0] / num_list[1]  

            if reformat_row['FIGHT_TIME'][1] >= reformat_row['FIGHT_TIME'][0]:    
                new_fight_row['FIGHT_MINUTE_P_TAR_2'] = reformat_row['FIGHT_TIME'][0] / reformat_row['FIGHT_TIME'][1]

        # Calculate win, tko, submission, decision average value for both fighter from all they past fight up to given fight (not include)
        id_i = 0
        other_id_i = 1
        fighter_id = reformat_row['FIGHTER_ID'][id_i]
        other_fighter_id = reformat_row['FIGHTER_ID'][other_id_i]

        is_win = reformat_row['WIN'][0] == 'W'
        fight_win_method = reformat_row['METHOD']
        all_win_method = ['TKO', 'SUB', 'DCW']
        for i in range(len(state_data_columns)):
            i_w = i * 2 + id_i
            other_i_w = i * 2 + other_id_i
            is_method_used = all_win_method[i] == fight_win_method
            # Store value before update
            get_needed_fight_state_data(new_fight_row, fighter_build_info, i, i_w, other_i_w, fighter_id, other_fighter_id)

            now_win_value = int(is_method_used and is_win)
            now_other_win_value = int(is_method_used and not is_win)
            now_win_percent = get_avg_value(now_win_value, 
                            fighter_build_info[fighter_id][2][state_data_columns[i]], 
                            fighter_build_info[fighter_id][4][0])
            now_loss_percent = get_avg_value(now_other_win_value, 
                            fighter_build_info[fighter_id][2][state_side_columns[i]], 
                            fighter_build_info[fighter_id][4][0])
            fighter_build_info[fighter_id][2][state_data_columns[i]] = now_win_percent
            fighter_build_info[fighter_id][2][state_side_columns[i]] = now_loss_percent

            now_other_win_percent = get_avg_value(now_other_win_value, 
                            fighter_build_info[other_fighter_id][2][state_data_columns[i]], 
                            fighter_build_info[other_fighter_id][4][0])
            now_other_loss_percent = get_avg_value(now_win_value, 
                            fighter_build_info[other_fighter_id][2][state_side_columns[i]], 
                            fighter_build_info[other_fighter_id][4][0])
            fighter_build_info[other_fighter_id][2][state_data_columns[i]] = now_other_win_percent   
            fighter_build_info[other_fighter_id][2][state_side_columns[i]] = now_other_loss_percent            

            new_fight_row[target_win_columns[i_w]] = now_win_value
            new_fight_row[target_win_columns[other_i_w]] = now_other_win_value
        # Calculate fighter's average draw and no contest percentage
        dcd_col = 'DEC_DRAW_%'
        dcd_col_value = int(fight_win_method in ['DCD', 'NC'])
        new_fight_row[dcd_col + '_' + str(id_i)] = fighter_build_info[fighter_id][2][dcd_col]
        new_fight_row[dcd_col + '_' + str(other_id_i)] = fighter_build_info[other_fighter_id][2][dcd_col]
        new_fight_row[dcd_col + '_D_TAR_2'] = dcd_col_value
        
        fighter_build_info[fighter_id][2][dcd_col] = get_avg_value(dcd_col_value, 
                                            fighter_build_info[fighter_id][2][dcd_col], 
                                            fighter_build_info[fighter_id][4][0])
        fighter_build_info[other_fighter_id][2][dcd_col] = get_avg_value(dcd_col_value, 
                                            fighter_build_info[other_fighter_id][2][dcd_col], 
                                            fighter_build_info[other_fighter_id][4][0])

        fight_train_df = fight_train_df.append(new_fight_row, ignore_index=True)
    
    with open(fighter_build_info_path, 'wb') as handle:
        pickle.dump(fighter_build_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fight_train_df.to_csv(fight_train_df_path)

def get_needed_fighter_data(new_row, fighter_id, id_i, fight_date, fighter_age=np.nan):
    fighter_reformat_row = fighter_reformat_df[fighter_reformat_df['FIGHTER_ID'] == fighter_id].squeeze()
    # Get fighter stance use one hot encoding
    for s in ['ORT', 'SOU', 'SWI']:
        new_row['STANCE_' + s + '_' + str(id_i)] = int(str(fighter_reformat_row['STANCE']) == s)
    # Get fighter reach and height for fight input value
    for i in range(len(fighter_static_columns)):
        i_w = i * 2 + id_i
        new_row[fight_fighter_static_columns[i_w]] = fighter_reformat_row[fighter_static_columns[i]]
    # Get fighter age according to date of born and fight date
    pattern = re.compile(r"[0-9\-]+")

    if np.isnan(fighter_age) and bool(pattern.match(str(fighter_reformat_row['DOB']))) and bool(pattern.match(str(fight_date))):
        dob = pd.to_datetime(fighter_reformat_row['DOB'])
        date = pd.to_datetime(fight_date)
        time_diff = dateutil.relativedelta.relativedelta(date, dob)
        fighter_age = time_diff.years + (time_diff.months + time_diff.days / 31) / 12
    new_row['AGE_' + str(id_i)] = fighter_age

def get_needed_fight_stats_data(new_row, fighter_build_info, i_w, other_i_w, fighter_id, other_fighter_id):
    # Store stats value for model input
    new_row[fight_weighted_data_columns[i_w]] = fighter_build_info[fighter_id][0][fight_weighted_data_columns[i_w][:-2]]
    new_row[fight_weighted_side_columns[other_i_w]] = fighter_build_info[other_fighter_id][0][fight_weighted_side_columns[i_w][:-2]]
    new_row[fight_avg_data_columns[i_w]] = fighter_build_info[fighter_id][2][fight_avg_data_columns[i_w][:-2]]
    new_row[fight_avg_side_columns[other_i_w]] = fighter_build_info[other_fighter_id][2][fight_avg_side_columns[i_w][:-2]]
    new_row[fight_pm_data_columns[i_w]] = fighter_build_info[fighter_id][2][fight_pm_data_columns[i_w][:-2]]
    new_row[fight_pm_side_columns[other_i_w]] = fighter_build_info[other_fighter_id][2][fight_pm_side_columns[i_w][:-2]]

def get_needed_fight_state_data(new_row, fighter_build_info, i, i_w, other_i_w, fighter_id, other_fighter_id):
    new_row[fight_state_data_columns[i_w]] = fighter_build_info[fighter_id][2][state_data_columns[i]]
    new_row[fight_state_side_columns[i_w]] = fighter_build_info[fighter_id][2][state_side_columns[i]]
    new_row[fight_state_data_columns[other_i_w]] = fighter_build_info[other_fighter_id][2][state_data_columns[i]]
    new_row[fight_state_side_columns[other_i_w]] = fighter_build_info[other_fighter_id][2][state_side_columns[i]]

def reverse_01(i):
    return (i + 1) % 2

def get_avg_value(new_value, old_avg_value, count):
    return old_avg_value + (new_value - old_avg_value) / count

def normalize_df_input(df):
    print("Normalize df input\n")
    fighter_data_columns = fighter_dynamic_columns[:5] + fighter_static_columns
    fight_fighter_data_columns = init_columns_name('_', fighter_data_columns)
    fight_need_normalize_columns = fight_fighter_data_columns + fight_pm_data_columns + fight_pm_side_columns + fight_other_data_columns[:1]

    # Normalize some columns of data frame
    normalize_parameters = {}
    parameters_path = './Data/normalize_parameters.pickle'
    if path.exists(parameters_path):
        with open(parameters_path, 'rb') as handle:
            normalize_parameters = pickle.load(handle)
    else :
        for n_col in fight_need_normalize_columns:
            parameters = []
            parameters.append(df[n_col].mean())
            parameters.append(df[n_col].max() - df[n_col].min())
            normalize_parameters[n_col] = parameters

        with open(parameters_path, 'wb') as handle:
            pickle.dump(normalize_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for row_index, train_row in df.iterrows(): 
        for n_col in normalize_parameters:
            df.loc[row_index, n_col] = mean_normalization(train_row[n_col], normalize_parameters[n_col][0], normalize_parameters[n_col][1])

    for n_col in normalize_parameters:
        df[n_col] = df[n_col].fillna(0)  

def mean_normalization(x, mean, val_range):
    return (x - mean) / val_range

# Uniform sample data in chunks, chunks is divided respect to time of fight
def df_sample():
    test_dataset = pd.DataFrame(columns=fight_train_df.columns)
    if bs.divide_by_chunks:
        print("Splite train/test dataset using chunks uniform sample")
        df_chunks = split_dataframe(fight_train_df, bs.chunk_size)
        for df_chunk in df_chunks:
            test_dataset = test_dataset.append(df_chunk.sample(frac=bs.test_set_frac), ignore_index=True)
    else :
        print("Splite train/test dataset using uniform sample")
        test_dataset = fight_train_df.sample(frac=bs.test_set_frac)
    train_dataset = fight_train_df.drop(test_dataset.index)
    return train_dataset, test_dataset

# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks
def split_dataframe(df, chunk_size = 100): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def build_and_process_train_fight_data():
    print("Build and process train fight data\n")
    global fight_train_df
    if path.exists(fight_train_df_path):
        fight_train_df = pd.read_csv(fight_train_df_path, index_col=0)
    else :
        build_train_fight_data()
    
    fight_train_df = double_and_switch_df_input(fight_train_df)
    normalize_df_input(fight_train_df)

    train_dataset, test_dataset = df_sample()
    train_dataset.to_csv("./Data/FightAllTrainData.csv")
    test_dataset.to_csv("./Data/FightAllTestData.csv")

def double_and_switch_df_input(dataset):
    print("Double and switch df input\n")
    # In order to train the model that treat two fighter's feature irrelevant to their input position, 
    # we double the input row and switch the all paired data: 0->1; 1->0
    mirror_dataset = pd.DataFrame(columns=dataset.columns)
    for _, train_row in dataset.iterrows(): 
        new_mirror_row = {}
        for i in range(len(train_row.index)):
            new_col = col = train_row.index[i]
            if col[-2:] == '_0':
                new_col = col[:-2] + '_1'
            elif col[-2:] == '_1':
                new_col = col[:-2] + '_0'
            new_mirror_row[new_col] = train_row[col]
        mirror_dataset = mirror_dataset.append(new_mirror_row, ignore_index=True)
    
    return dataset.append(mirror_dataset, ignore_index=True)

def build_and_process_predict_fight_data():
    print("Build and process predict fight data\n")
    if path.exists(fighter_build_info_path):
        with open(fighter_build_info_path, 'rb') as handle:
            fighter_build_info = pickle.load(handle)

        need_predict_df = pd.DataFrame(columns=fight_input_columns)

        fighter_id_pairs = list(zip(bs.fighter_0_ids, bs.fighter_1_ids))
        all_need_predict_num = len(fighter_id_pairs)
        fighter_0_ages = bs.fighter_0_ages + [np.nan] * max(all_need_predict_num - len(bs.fighter_0_ages), 0)
        fighter_1_ages = bs.fighter_1_ages + [np.nan] * max(all_need_predict_num - len(bs.fighter_1_ages), 0)
        fighter_age_pairs = list(zip(fighter_0_ages, fighter_1_ages))
        for fp_i in range(all_need_predict_num):
            new_need_predict_row = {}

            for id_i in range(2):
                fighter_id = fighter_id_pairs[fp_i][id_i]
                other_id_i = reverse_01(id_i)
                other_fighter_id = fighter_id_pairs[fp_i][other_id_i]

                new_need_predict_row['WOMEN'] = int(not bs.is_man)
                fight_weight_class = bs.fight_weights[fp_i]
                for w_col in fight_other_data_columns[2:]:
                    new_need_predict_row[w_col] = int(w_col == fight_weight_class)

                fight_date = bs.fight_date
                if fight_date is None:
                    fight_date = datetime.datetime.now().strftime("%Y-%m-%d")
                get_needed_fighter_data(new_need_predict_row, fighter_id, id_i, fight_date, fighter_age_pairs[fp_i][id_i])

                new_need_predict_row['FIGHT_COUNT_' + str(id_i)] = fighter_build_info[fighter_id][4][0]
                new_need_predict_row['FIGHT_MINUTE_' + str(id_i)] = fighter_build_info[fighter_id][4][1]

                for i in range(len(need_weight_data_columns)):
                    i_w = i * 2 + id_i
                    other_i_w = i * 2 + other_id_i
                    get_needed_fight_stats_data(new_need_predict_row, fighter_build_info, i_w, other_i_w, fighter_id, other_fighter_id)

                new_need_predict_row['REV_I_' + str(id_i)] = fighter_build_info[fighter_id][2]['REV_I']
                new_need_predict_row['REV_P_' + str(other_id_i)] = fighter_build_info[other_fighter_id][2]['REV_P']

            for i in range(len(state_data_columns)):
                i_w = i * 2 + id_i
                other_i_w = i * 2 + other_id_i
                get_needed_fight_state_data(new_need_predict_row, fighter_build_info, i, i_w, other_i_w, fighter_id, other_fighter_id)
            
            dcd_col = 'DEC_DRAW_%'
            new_need_predict_row[dcd_col + '_' + str(id_i)] = fighter_build_info[fighter_id][2][dcd_col]
            new_need_predict_row[dcd_col + '_' + str(other_id_i)] = fighter_build_info[other_fighter_id][2][dcd_col]

            need_predict_df = need_predict_df.append(new_need_predict_row, ignore_index=True)
            
        normalize_df_input(need_predict_df)
        
        # Add old need predict fight data if exist
        fight_need_predict_path = "./Data/FightNeedPredictData.csv"
        if path.exists(fight_need_predict_path):
            pre_need_predict_df = pd.read_csv(fight_need_predict_path, index_col=0)
            need_predict_df = need_predict_df.append(pre_need_predict_df, ignore_index=True)

        need_predict_df.to_csv(fight_need_predict_path)

if __name__ == "__main__":
    if bs.is_build_train:
        build_and_process_train_fight_data()
    else:
        build_and_process_predict_fight_data()