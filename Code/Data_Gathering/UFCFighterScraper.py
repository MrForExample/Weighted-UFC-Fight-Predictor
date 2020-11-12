import scrapy
import pandas as pd
import string
from os import path

fighter_data_columns = ['FIGHTER', 'RECORD', 'HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB', 
                        'SLpM', 'Str_Acc', 'SApM', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg', 'FIGHTER_ID']
all_fighter_df = pd.DataFrame(columns=fighter_data_columns)

fighter_raw_df_path = './Data/FighterRawData.csv'

pre_fighter_df = None
if path.exists(fighter_raw_df_path):
    pre_fighter_df = pd.read_csv(fighter_raw_df_path, index_col=0)

class UFCFighterSpider(scrapy.Spider):

    name = 'ufc_fighter_spider'
    start_urls = ["http://www.ufcstats.com/statistics/fighters?char=" + char + "&page=all" for char in list(string.ascii_lowercase)]
    #start_urls = ["http://www.ufcstats.com/statistics/fighters?char=z&page=all"]

    all_fighter_data = []

    def parse(self, response): 
        info_tr_rows = response.css("tr.b-statistics__table-row")
        for row in info_tr_rows:
            link = row.css("a::attr(href)").get()
            if link != None:
                fighter_id = link.split("/")[-1]
                if pre_fighter_df is None or fighter_id not in pre_fighter_df['FIGHTER_ID'].values:
                    new_fighter_data = {}
                    new_fighter_data['FIGHTER_ID'] = fighter_id
                    self.all_fighter_data.append(new_fighter_data)
                    yield response.follow(link, callback=self.parse_fighter, cb_kwargs=dict(fighter_data=new_fighter_data))

    def parse_fighter(self, response, fighter_data):
        fighter_data['FIGHTER'] = response.css("span.b-content__title-highlight::text").get().strip()
        fighter_data['RECORD'] = response.css("span.b-content__title-record::text").get().strip()
        info_li_items = response.css("li.b-list__box-list-item")
        for i in range(len(info_li_items)):
            if i != 9:
                fighter_data[fighter_data_columns[i + (i < 9) + 1]] = info_li_items[i].xpath("text()").extract()[1].strip()

    def closed(self, reason):
        global all_fighter_df
        for fighter_data in self.all_fighter_data:
            all_fighter_df = all_fighter_df.append(fighter_data, ignore_index=True)
        
        if pre_fighter_df is not None:
            all_fighter_df = all_fighter_df.append(pre_fighter_df, ignore_index=True)        
        all_fighter_df.to_csv(fighter_raw_df_path)