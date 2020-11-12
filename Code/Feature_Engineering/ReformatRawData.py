import pandas as pd
import numpy as np
import calendar
import re
from ast import literal_eval

def m_s_to_float_m(m, s):
    return round(float(m) + float(s) / 60, 3)

def inches_to_cm(inchs):
    return round(inchs * 2.54, 2)

def feet_to_cm(feet, inchs):
    return inches_to_cm(feet * 12 + inchs)

def reformat_date(raw_date):
    date = separate_data_by_rex(raw_date)
    if len(date) > 0:
        assert len(date) == 3, "Date format is not Y M D: {}".format(date)

        for i in range(3):
            if date[i].isalpha():
                date[i] = "{0:02d}".format(list(calendar.month_abbr).index(date[i][:3].lower().capitalize()))
                break

        year = max(date, key=len)
        assert len(year) == 4, "Year length is not 4: {}".format(year)

        date.insert(0, date.pop(date.index(year)))
        date = '-'.join(date)
    else :
        date = np.nan
    return date  

def separate_data_by_rex(raw_data, rex=r"[a-zA-Z0-9]+"):
    separate_data = re.findall(rex, raw_data)
    return separate_data

def separate_data_to_nums(raw_data):
    nums = [[], []]
    for i in range(len(raw_data)):
        nums[i % 2] += [int(num) for num in re.findall(r"[0-9]+", raw_data[i])]
    return nums    

def reformat_fighter_data():
    print("Reformat fighter data\n")
    fighter_data_columns = ['FIGHTER', 'DOB', 'HEIGHT', 'REACH', 'STANCE', 'FIGHTER_ID']
    fighter_raw_df = pd.read_csv("./Data/FighterRawData.csv", index_col=0)
    fighter_df = pd.DataFrame(columns=fighter_data_columns)

    for _, raw_row in fighter_raw_df.iterrows():
        new_fighter_row = {}
        for col in ['FIGHTER', 'FIGHTER_ID']:
            new_fighter_row[col] = raw_row[col]

        # reformat fighter date of born
        new_fighter_row['DOB'] = reformat_date(raw_row['DOB'])

        # reformat fighter height
        fighter_height = separate_data_by_rex(raw_row['HEIGHT'])
        if len(fighter_height) > 0:
            assert len(fighter_height) == 2, "Height format is not F'I: {}".format(fighter_height)
            fighter_height = feet_to_cm(int(fighter_height[0]), int(fighter_height[1]))
        else :
            fighter_height = np.nan

        # reformat fighter reach
        fighter_reach = separate_data_by_rex(raw_row['REACH'], rex=r"[a-zA-Z0-9\.]+")
        if len(fighter_reach) > 0:
            assert len(fighter_reach) == 1, "Reach format is not inchs: {}".format(fighter_reach)
            fighter_reach = inches_to_cm(float(fighter_reach[0]))
        else :
            fighter_reach = np.nan

        # The reach is usually very close to height
        if np.isnan(fighter_reach) and not np.isnan(fighter_height):
            fighter_reach = fighter_height
        elif np.isnan(fighter_height) and not np.isnan(fighter_reach):
            fighter_height = fighter_reach

        new_fighter_row['HEIGHT'] = fighter_height
        new_fighter_row['REACH'] = fighter_reach

        # reformat fighter stance, only three: Orthodox(ORT), Southpaw(SOU), Switch(SWI)
        fighter_stance = np.nan
        s_upper = str(raw_row['STANCE']).upper()
        if 'ORT' in s_upper:
            fighter_stance = 'ORT'
        elif 'SOU' in s_upper:
            fighter_stance = 'SOU'
        elif 'SWI' in s_upper or 'OPEN' in s_upper:
            fighter_stance = 'SWI'
        new_fighter_row['STANCE'] = fighter_stance

        fighter_df = fighter_df.append(new_fighter_row, ignore_index=True)

    fighter_df.to_csv("./Data/FighterReformatData.csv")

def reformat_fight_data():
    print("Reformat fight data\n")
    # load fight data, some cell convert to list
    fight_data_columns = ['DATE', 'WEIGHT_CLASS', 'METHOD', 'FIGHT_TIME', 'WIN', 'FIGHTER_ID', 'FIGHTER', 'KD_0', 'KD_1', 'SIG_STR_0', 'SIG_STR_1', 
                        'TOTAL_STR_0', 'TOTAL_STR_1', 'TD_0', 'TD_1', 'SUB_ATT_0', 'SUB_ATT_1', 'REV_0', 'REV_1', 'CTRL_0', 'CTRL_1', 
                        'HEAD_0', 'HEAD_1', 'BODY_0', 'BODY_1', 'LEG_0', 'LEG_1', 'DISTANCE_0', 'DISTANCE_1', 'CLINCH_0', 'CLINCH_1', 'GROUND_0', 'GROUND_1']

    list_data_columns = ['WIN', 'FIGHTER_ID', 'FIGHTER', 'KD', 'SIG_STR', 'SIG_STR_%', 'TOTAL_STR', 'TD', 'TD_%', 'SUB_ATT',
                        'REV', 'CTRL', 'HEAD', 'BODY', 'LEG', 'DISTANCE', 'CLINCH', 'GROUND']
    columns_converters = {}
    for col in list_data_columns:
        columns_converters[col] = literal_eval

    fight_raw_df = pd.read_csv("./Data/FightRawData.csv", converters=columns_converters, index_col=0)
    fight_df = pd.DataFrame(columns=fight_data_columns)

    for _, raw_row in fight_raw_df.iterrows():
        new_fight_row = {}
        for col in ['WIN', 'FIGHTER_ID', 'FIGHTER']:
            new_fight_row[col] = raw_row[col]

        # reformat fight date
        new_fight_row['DATE'] = reformat_date(raw_row['DATE'])
        #print(fight_date)
        
        # reformat weight class 
        weight_calss = []
        weight_calss_w = separate_data_by_rex(raw_row['WEIGHT_CLASS'])
        for w in weight_calss_w:
            if any(s in w.lower().capitalize() for s in ['Women', 'Light', 'weight']):
                weight_calss.append(w)
        if len(weight_calss) == 0:
            weight_calss.append("Open Weight")
        new_fight_row['WEIGHT_CLASS'] = ' '.join(weight_calss)

        # reformat result
        result_method = 'NC'
        method = separate_data_by_rex(raw_row['METHOD'])
        for m in method:
            m_upper = m.upper()
            if new_fight_row['WIN'][0] != new_fight_row['WIN'][1]:
                if 'KO' in m_upper:
                    result_method = 'TKO'
                elif 'DEC' in m_upper:
                    result_method = 'DCW'
                elif 'SUB' in m_upper:
                    result_method = 'SUB'
            elif new_fight_row['WIN'][0] == 'D':
                result_method = 'DCD'
        new_fight_row['METHOD'] = result_method

        # reformat fight time format: ROUND TIME TIME_FORMAT -> [final fight time, all round time]
        fight_time_format = separate_data_by_rex(raw_row['TIME_FORMAT'], r'\(([^\(\)]*)\)')
        all_round_time = 0
        final_fight_time = 0
        final_round_num = int(raw_row['ROUND'])
        if len(fight_time_format) > 0:
            fight_time_format = separate_data_by_rex(fight_time_format[0], rex=r"[0-9]+")
            for i_r in range(len(fight_time_format)):
                round_time = int(fight_time_format[i_r])
                all_round_time += round_time
                fight_time_format[i_r] = round_time
            fight_time_format.insert(0, all_round_time)

            for i_t in range(1, final_round_num):
                final_fight_time += fight_time_format[i_t]

        last_round_time = separate_data_by_rex(raw_row['TIME'], rex=r"[0-9]+")
        assert len(last_round_time) == 2, "Time format is not M:S : {}".format(last_round_time)
        final_fight_time = m_s_to_float_m(final_fight_time + float(last_round_time[0]), last_round_time[1])

        if final_fight_time > all_round_time:
            if final_fight_time > 15:
                all_round_time = max(final_fight_time, 25)
            else :
                all_round_time = max(final_fight_time, 15)
        new_fight_row['FIGHT_TIME'] = [final_fight_time, all_round_time]

        # reformat knockdown, submission attempt, Reversal/Sweep, strike and takedown numbers        
        strike_columns = ['KD', 'SUB_ATT', 'REV', 'CTRL', 'SIG_STR', 'TOTAL_STR', 'TD', 'HEAD', 'BODY', 'LEG', 'DISTANCE', 'CLINCH', 'GROUND']
        for i in range(len(strike_columns)):
            separated_data_list = separate_data_to_nums(raw_row[strike_columns[i]])
            for j in range(len(separated_data_list)):
                new_fight_row[strike_columns[i] + "_" + str(j % 2)] = separated_data_list[j]

        # reformat center control time: minute, second -> float(minute), round minute
        for col in ['CTRL_0', 'CTRL_1']:
            fighters_ctrl_times = []
            for i in range(1, len(new_fight_row[col]), 2):
                fighters_ctrl_times.append(m_s_to_float_m(new_fight_row[col][i-1], new_fight_row[col][i]))
                fighters_ctrl_times.append(fight_time_format[i // 2])
            new_fight_row[col] = fighters_ctrl_times

        # reformat submission: attempt number -> success or not,  attempt number
        for i in range(2):
            col = "SUB_ATT_" + str(i)
            fighter_subs = []
            is_sub_win = new_fight_row['METHOD'] == 'SUB' and new_fight_row['WIN'][i] == 'W'
            is_sub_round = True
            for j in range(len(new_fight_row[col])):
                sub_win = int(is_sub_win and is_sub_round)
                fighter_subs.append(sub_win)
                fighter_subs.append(new_fight_row[col][j])
                is_sub_round = final_round_num == j + 1
            new_fight_row[col] = fighter_subs

        # reformat knockdown: knockdown number -> knockdown number, significant strike land number
        for i in range(2):
            col = "KD_" + str(i)  
            fighter_kds = []
            for j in range(len(new_fight_row[col])):
                fighter_kds.append(new_fight_row[col][j]) 
                fighter_kds.append(new_fight_row['SIG_STR_' + str(i)][j * 2])
            new_fight_row[col] = fighter_kds

        fight_df = fight_df.append(new_fight_row, ignore_index=True)

    fight_df.to_csv("./Data/FightReformatData.csv")          

if __name__ == "__main__":
    reformat_fighter_data()
    reformat_fight_data()