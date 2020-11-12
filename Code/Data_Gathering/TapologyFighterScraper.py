import scrapy
import pandas as pd

self_df = pd.read_csv("../FighterRawData.csv", index_col=0)
miss_fighters_name = self_df[self_df.values == '--'].drop_duplicates(subset=['FIGHTER'])['FIGHTER'].values.tolist()
# Can't scrape all at once due to site spider restriction strategy
miss_fighters_name = miss_fighters_name[1500:]

start_url = "https://www.tapology.com/search?term="
#start_url = "https://www.tapology.com/search?term=Papy+Abedi"

fighter_data_columns = ['FIGHTER', 'HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB']

all_fighter_df = pd.DataFrame(columns=fighter_data_columns)

class TapologyFighterSpider(scrapy.Spider):

    name = 'tapology_fighter_spider'

    all_fighter_data = []

    custom_settings = {
        'CONCURRENT_REQUESTS': '1',
        'DOWNLOAD_DELAY': 1
    }
    
    def start_requests(self):
        url_fighters_name = []
        print(miss_fighters_name)
        print(len(miss_fighters_name))
        for name in miss_fighters_name:
            name_list = name.split(' ')
            if '' in name_list: 
                name_list.remove('')
                
            url_name = name_list[0]
            for i in range(1, len(name_list)):
                url_name += '+' + name_list[i]             

            url_fighters_name.append(url_name)

        start_urls = [start_url + url_name for url_name in url_fighters_name]
        #start_urls = [start_url]

        headers =  {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36',
            'Accept': 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate, sdch',
            'Accept-Language': 'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4',
        }

        for url in start_urls:
            yield scrapy.http.Request(url, headers=headers)     

    def parse(self, response):
        fighter_name = response.url.split('=')[-1].split('+')
        info_a_items = response.css("table.fcLeaderboard a")
        for item in info_a_items:
            fighter_name_on_site = item.xpath("text()").extract_first().strip().split(' ')
            if all(n in fighter_name_on_site for n in fighter_name):
                link = item.css("a::attr(href)").get()
                if link != None and link.split('/')[-2] == 'fighters':
                    new_fighter_data = {}
                    new_fighter_data['FIGHTER'] = ' '.join(fighter_name)
                    self.all_fighter_data.append(new_fighter_data)
                    yield response.follow(link, callback=self.parse_fighter, cb_kwargs=dict(fighter_data=new_fighter_data))

    def parse_fighter(self, response, fighter_data):
        info_li_items = response.css("div.details_two_columns ul.clearfix li")
        if len(info_li_items) > 0:
            info_span_items = info_li_items[4].css("span::text").getall()
            if len(info_span_items) > 0:
                fighter_data['DOB'] = info_span_items[-1].strip()

            info_span_items = info_li_items[6].css("span::text").getall()
            if len(info_span_items) > 0:
                fighter_data['WEIGHT'] = info_span_items[-1].strip()

            info_span_items = info_li_items[8].css("span::text").getall()
            if len(info_span_items) > 0:
                fighter_data['HEIGHT'] = info_span_items[0].strip().split(' ')[0]
                fighter_data['REACH'] = info_span_items[1].strip().split(' ')[0]

    def closed(self, reason):
        global all_fighter_df
        for fighter_data in self.all_fighter_data:
            #print(fighter_data)
            all_fighter_df = all_fighter_df.append(fighter_data, ignore_index=True)
        all_fighter_df.to_csv('../raw_fighter_details_1500-1784.csv')

