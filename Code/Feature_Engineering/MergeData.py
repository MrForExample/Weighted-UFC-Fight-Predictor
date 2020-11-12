import pandas as pd
import numpy as np

if __name__ == "__main__":
    self_df = pd.read_csv("./Data/FighterRawData.csv", index_col=0)
    csv_name = "raw_fighter_details"
    other_df = pd.read_csv("./Data/"+ csv_name +".csv", index_col=0)

    manual_df = pd.read_csv("./Data/raw_fighter_details_Manual.csv", index_col=0)

    self_df = self_df.replace(['null', '', np.nan], '--')
    other_df = other_df.replace(['null', '', np.nan], '--')

    self_column = ['HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB']
    other_column = ['Height', 'Weight', 'Reach', 'Stance', 'DOB']
    #other_column = ['HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB']
    miss_df = self_df.drop_duplicates(subset=['FIGHTER'], keep=False)
    miss_df = miss_df[miss_df.values == '--'].drop_duplicates()
    #miss_df = self_df[self_df.values == '--'].drop_duplicates(subset=['FIGHTER'], keep=False)
    for index, row in miss_df.iterrows():
        fighter_name = row['FIGHTER']
        num_of_name = len(other_df[other_df['FIGHTER'] == fighter_name])
        if num_of_name == 1:
            other_row = other_df[other_df['FIGHTER'] == fighter_name].squeeze()
            for i in range(5):
                #print("{}: {}-{}".format(fighter_name, row[self_column[i]], other_row[other_column[i]]))
                if row[self_column[i]] == '--' and not other_row[other_column[i]] == '--':
                    self_df.loc[index, self_column[i]] = other_row[other_column[i]]
                    print("Change fighter {}'s {} to {}".format(fighter_name, self_column[i], other_row[other_column[i]]))
        elif num_of_name > 1 and fighter_name not in manual_df['FIGHTER'].values:
            print("{} with many name".format(fighter_name))
            manual_df = manual_df.append({'FIGHTER': fighter_name}, ignore_index=True)
