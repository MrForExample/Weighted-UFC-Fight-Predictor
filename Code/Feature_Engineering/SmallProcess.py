import pandas as pd
import numpy as np

#csv_name = "raw_fighter_details_1500-1784"
#fighter_df = pd.read_csv("./Data/"+ csv_name +".csv", index_col=0)
'''
fighter_df = pd.read_csv("./Data/FighterRawData_New.csv", index_col=0)
manual_df = pd.read_csv("./Data/raw_fighter_details_Manual_Old.csv", index_col=0)

for index, row in fighter_df.iterrows():
    fighter_name = row['FIGHTER']
    num_of_name = len(fighter_df[fighter_df['FIGHTER'] == fighter_name])
    if num_of_name > 1 and fighter_name not in manual_df['FIGHTER'].values:
        print("{}: {}".format(fighter_name, num_of_name))
        manual_df = manual_df.append({'FIGHTER': fighter_name}, ignore_index=True)
'''
#fighter_df = pd.read_csv("./Data/FighterRawData.csv", index_col=0)

#fighter_df = fighter_df.replace(['null', '', np.nan, 'N/A'], '--')
#manual_df.to_csv("./Data/raw_fighter_details_Manual_Old.csv")
#fighter_df.to_csv("./Data/"+ csv_name +".csv")

manual_old_df = pd.read_csv("./Data/raw_fighter_details_Manual_Old.csv", index_col=0)
manual_df = pd.read_csv("./Data/raw_fighter_details_Manual.csv", index_col=0)

for index, row in manual_df.iterrows():
    if row['FIGHTER'] in manual_old_df['FIGHTER'].values:
        manual_df = manual_df.drop(index)

manual_df.to_csv("./Data/raw_fighter_details_Manual_New.csv")