import scrapy
import pandas as pd
from os import path

start_url = "http://www.ufcstats.com/statistics/events/completed?page=all"
#start_url = "http://www.ufcstats.com/statistics/events/completed?page=1"

detail_data_colums = ['EVENT_ID', 'DATE', 'LOCATION', 'WEIGHT_CLASS', 'METHOD', 'ROUND', 'TIME', 'TIME_FORMAT', 'REFEREE', 'DETAILS', 'WIN', 'FIGHTER_ID', 'FIGHTER']
total_data_columns = ['KD', 'SIG_STR', 'SIG_STR_%', 'TOTAL_STR', 'TD', 'TD_%', 'SUB_ATT', 'REV', 'CTRL']
sig_str_data_columns = ['HEAD', 'BODY', 'LEG', 'DISTANCE', 'CLINCH', 'GROUND']
fight_data_columns = detail_data_colums + total_data_columns + sig_str_data_columns

all_fight_df = pd.DataFrame(columns=fight_data_columns)

fight_raw_df_path = './Data/FightRawData.csv'

pre_fight_df = None
if path.exists(fight_raw_df_path):
    pre_fight_df = pd.read_csv(fight_raw_df_path, index_col=0)
    #pre_fight_df = pre_fight_df.loc[:, ~pre_fight_df.columns.str.contains('^Unnamed')]

class UFCFightSpider(scrapy.Spider):

    name = 'ufc_fight_spider'
    start_urls = [start_url]

    all_event_data = []

    def parse(self, response):
        all_events_link = self.get_link(response, "tr.b-statistics__table-row", "a::attr(href)")

        all_events_link = self.remove_repeat_event(all_events_link)

        print("Total {} of events don't store yet".format(len(all_events_link)))

        for link in all_events_link:
            new_event_fight_data = []
            self.all_event_data.append(new_event_fight_data)
            yield response.follow(link, callback=self.parse_event, cb_kwargs=dict(event_fight_data=new_event_fight_data))

        '''
        new_event_fight_data = []
        self.all_event_data.append(new_event_fight_data)
        yield response.follow(all_events_link[0], callback=self.parse_event, cb_kwargs=dict(event_fight_data=new_event_fight_data))
        '''
            
    def parse_event(self, response, event_fight_data):
        event_id = response.url.split("/")[-1]

        event_fights_link = self.get_link(response, "tr.b-fight-details__table-row", "a.b-flag::attr(href)")
        main_event_name = response.css("span.b-content__title-highlight::text").get().strip()
        date_loclation = response.css("li.b-list__box-list-item::text").getall()
        date = date_loclation[1].strip()
        location = date_loclation[3].strip()
        fights_num = len(event_fights_link)
        print("Total {} of fights at date {}, location {}, event {}".format(fights_num, date, location, main_event_name))

        for i in range(fights_num):
            new_fight_data = self.create_fight_dict()
            new_fight_data['EVENT_ID'] = event_id
            new_fight_data['DATE'] = date
            new_fight_data['LOCATION'] = location
            event_fight_data.append(new_fight_data)
            yield response.follow(event_fights_link[i], callback=self.parse_fight, cb_kwargs=dict(fight_data=new_fight_data))

    def parse_fight(self, response, fight_data):
        fight_data['WEIGHT_CLASS'] = response.css("div.b-fight-details__fight-head i::text").getall()[-1].strip()

        win = response.css("i.b-fight-details__person-status::text").getall()
        fight_data['WIN'] = [win[0].strip(), win[1].strip()]

        info_fighter_items = response.css("h3.b-fight-details__person-name")
        fighter_id = info_fighter_items.css("a.b-fight-details__person-link::attr(href)").getall()
        fight_data['FIGHTER_ID'] = [fighter_id[0].split("/")[-1], fighter_id[1].split("/")[-1]]
        fighter_name = info_fighter_items.css("a.b-fight-details__person-link::text").getall()
        fight_data['FIGHTER'] = [fighter_name[0].strip(), fighter_name[1].strip()]

        fight_data['METHOD'] = response.css("i.b-fight-details__text-item_first i")[1].css("i::text").get().strip()
        info_i_items = response.css("i.b-fight-details__text-item::text").getall()
        fight_data['ROUND'] = info_i_items[1].strip()
        fight_data['TIME'] = info_i_items[3].strip()
        fight_data['TIME_FORMAT'] = info_i_items[5].strip()
        fight_data['REFEREE'] = response.css("i.b-fight-details__text-item span::text").get().strip()

        info_p_items = response.css("p.b-fight-details__text")[1]
        info_i_items = info_p_items.css("i.b-fight-details__text-item")
        if len(info_i_items) > 0:
            fight_data['DETAILS'] = ''
            for item in info_i_items:
                judge = item.css("span::text").get().strip()
                dec = item.xpath("text()").extract()[1].strip()
                fight_data['DETAILS'] += judge + ": " + dec
        else:
            fight_data['DETAILS'] = info_p_items.xpath("text()").extract()[1].strip()

        total_rows_items = []
        sig_str_rows_items = []
        info_tr_rows = response.css("tr.b-fight-details__table-row")
        for tr_row in info_tr_rows:
            info_p_items = tr_row.css("p.b-fight-details__table-text")
            item_num = len(info_p_items)
            if item_num == 20:
                total_rows_items.append(info_p_items)
            elif item_num == 18:
                sig_str_rows_items.append(info_p_items)

        for row_items in total_rows_items:
            for i in range(len(row_items)):
                if i > 1:
                    fight_data[total_data_columns[i//2-1]].append(row_items[i].xpath("text()").extract_first().strip())

        for row_items in sig_str_rows_items:
            for i in range(len(row_items)):
                if i > 5:
                    fight_data[sig_str_data_columns[i//2-3]].append(row_items[i].xpath("text()").extract_first().strip())

    def get_link(self, response, row_name, link_name):
        all_links = []
        rows = response.css(row_name)
        for row in rows:
            link = row.css(link_name).get()
            if link != None:
                all_links.append(link)
                #print("Event url: {}".format(link))
        return all_links

    def create_fight_dict(self):
        new_fight_data = {}
        for k in detail_data_colums:
            new_fight_data[k] = None
        for k in total_data_columns + sig_str_data_columns:
            new_fight_data[k] = []
        return new_fight_data

    def remove_repeat_event(self, all_events_link):
        if pre_fight_df is not None:
            for i in range(len(all_events_link)-1, -1, -1):
                event_id = all_events_link[i].split("/")[-1]
                if event_id in pre_fight_df['EVENT_ID'].values:
                    del all_events_link[i]
        return all_events_link

    def closed(self, reason):
        global all_fight_df
        for event_data in self.all_event_data:
            #print(event_data)
            all_fight_df = all_fight_df.append(event_data, ignore_index=True)
        
        if pre_fight_df is not None:
            all_fight_df = all_fight_df.append(pre_fight_df, ignore_index=True)
        all_fight_df.to_csv(fight_raw_df_path)

