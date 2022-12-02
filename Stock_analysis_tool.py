from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import math
import csv
import sys
import collections
import re
import yfinance as yf
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import matplotlib.pyplot as plt
#collections.Callable = collections.abc.Callable

class stock:
    def __init__(self,keyword,plot = False, csvfile='stocks.csv', mode = "scraping_mode"):
        print('Processing for stock: ' + keyword + ' .....')
        self.symbol = ""
        self.avg_pe = 35
        self.market_base = 20000000
        self.fieldnames = ['Name','Symbol', 'Price', 'Market Cap', 'EPS Trailing', 'EPS Forward', 'Change', 'Change%', 'Prev Close', 'Volume', 'Year Low', 'Year High', 'P/E Trailing', 'P/E Forward', 'Dividend', 'Dividend Yield', 'Target High', 'Target Avg', 'Target Low']
        self.csvfile = csvfile
        if mode == "scraping_mode":
            self.__scrape_websites(keyword)   
             
        else:
            self.__parse_csv(keyword)
            
        self.yahoo = yf.Ticker(self.symbol) 
        self.hist = self.yahoo.history(period="1y")
        if(plot == True):
            self.get_plot()
        self.__stock_prediction()
        self.__get_ratings()
        self.__get_target_price()
        


    def __scrape_websites(self,keyword):
        self.__get_symbols(keyword)
        self.__update_params()

        
    def __parse_csv(self,keyword):
        with open(self.csvfile, 'r',newline='') as f_object:
            inputfile = csv.DictReader(f_object, fieldnames=self.fieldnames)
            for row in inputfile:
                if len(re.findall(keyword,row['Name']+row['Symbol'],re.IGNORECASE)) > 0:
                    self.name = row['Name']
                    self.symbol = row['Symbol']
                    self.sublink = '/symbol/' + self.symbol
                    self.Price  = float(row['Price'])
                    self.Market_Cap  = float(row['Market Cap'])
                    self.market_cap = str(self.Market_Cap) + 'M'
                    self.EPS_Trailing  = float(row['EPS Trailing'])
                    self.EPS_Forward  = float(row['EPS Forward'])
                    self.Change  = float(row['Change'])
                    self.Change_pc  = float(row['Change%'])
                    self.Prev_Close  = float(row['Prev Close'])
                    self.Volume  = int(row['Volume'])
                    self.Year_Low  = float(row['Year Low'])
                    self.Year_High  = float(row['Year High'])
                    self.PE_Trailing  = float(row['P/E Trailing'])
                    self.PE_Forward  = float(row['P/E Forward'])
                    self.Dividend  = float(row['Dividend'])
                    self.Dividend_Yield  = float(row['Dividend Yield'])
                    self.High = float(row['Target High'])
                    self.Avg = float(row['Target Avg'])
                    self.Low = float(row['Target Low'])
                    return
            print('Keyword Not Found')
            self.symbol = 'NIL'
                    
                

    def __write_data_to_csv(self,diction):
        with open(self.csvfile, 'a',newline='') as f_object:
            dictwriter_object = csv.DictWriter(f_object, fieldnames=self.fieldnames)
            dictwriter_object.writerow(diction)
            f_object.close()


    def __parse_params(self,key,line,out):
        parser = '\D+' + key + '</th\D+'
        
        if len(re.findall(parser,line)) >0:
            out = next(out).split('>')[1].split('<')[0]
            if(out != '-'):
                return (True,out)
            else:
                return (True,'0.00')
    
            
        else:
            return (False,None)

    def __get_marketcap(self,Y):
        if Y[-1] == 'B':
            self.Market_Cap = float(Y[:-1])*1000
        elif Y[-1] == 'M':
            self.Market_Cap = float(Y[:-1])
        else:
            self.Market_Cap = float(Y)/1000000
        return 

    def __get_symbols(self,keyword):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
        home = requests.get('https://www.slickcharts.com/nasdaq100',headers=headers)
        souphead = BeautifulSoup(home.content, 'html.parser')
        
        inp = open("out2.txt",'w')
        inp.write(str(souphead.prettify))
        inp.close()
        out2 = open("out2.txt",'r')
        

        x=""
        for line in out2:
            if len(re.findall(keyword,line,re.IGNORECASE)) > 0:
                if(re.search('td',line)):
                    self.name = line.split('>')[2].split('<')[0]
                    self.sublink = line.split('"')[1].split('"')[0]
                    self.symbol = self.sublink.split('/')[2] 
                    out2.close()   
                    
                    os.remove("out2.txt")
                    return
        print('Invalid Stock name')
    
    def __get_link_of(self):
        return 'https://www.slickcharts.com%s/'%self.sublink

    def __update_params(self):
        self.__get_target_prices()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
        content = requests.get(self.__get_link_of(),headers=headers)
        soup = BeautifulSoup(content.content, 'html.parser')


        into = open("out.txt",'w')
        into.write(str(soup.prettify))
        into.close()
        out = open("out.txt",'r')
        
        #self.writer.writeheader()

        for line in out:
            
            (X,Y) = self.__parse_params('Price',line,out)
            if X:
                self.Price = float(Y.replace(',',''))
                
            
            (X,Y) = self.__parse_params('Change',line,out)
            if X:
                self.Change = float(Y)
            
            (X,Y) = self.__parse_params('Change %',line,out)
            if X:
                self.Change_pc = float(Y[:-1])
    
            (X,Y) = self.__parse_params('Prev Close',line,out)
            if X:
                self.Prev_Close = float(Y.replace(',',''))
    
            (X,Y) = self.__parse_params('Volume',line,out)
            if X:
                self.Volume = int((Y.replace(",","")))
                
    
            (X,Y) = self.__parse_params('Year Low',line,out)
            if X:
                self.Year_Low = float(Y.replace(',',''))
    
            (X,Y) = self.__parse_params('Year High',line,out)
            if X:
                self.Year_High = float(Y.replace(',',''))
    
            (X,Y) = self.__parse_params('P/E \(Trailing\)',line,out)
            if X:
                self.PE_Trailing = float(Y)
    
            (X,Y) = self.__parse_params('P/E \(Forward\)',line,out)
            if X:
                self.PE_Forward = float(Y)
    
            (X,Y) = self.__parse_params('Dividend Yield',line,out)
            if X:
                self.Dividend_Yield = float(Y[:-1])
    
            (X,Y) = self.__parse_params('Dividend',line,out)
            if X:
                self.Dividend = float(Y)
    
            (X,Y) = self.__parse_params('Market Cap',line,out)
            if X:
                self.market_cap = Y
                self.__get_marketcap(Y)
                
    
            (X,Y) = self.__parse_params('EPS \(Trailing\)',line,out)
            if X:
                self.EPS_Trailing = float(Y)
    
            (X,Y) = self.__parse_params('EPS \(Forward\)',line,out)
            if X:
                self.EPS_Forward = float(Y)
        if self.PE_Trailing == 0:
            if self.PE_Forward == 0:
                self.PE_Trailing = self.avg_pe
                self.PE_Forward = self.avg_pe
            else:
                self.PE_Trailing = self.PE_Forward

        #print(Price,Volume,Market_Cap,Dividend,Dividend_Yield,EPS_Trailing,EPS_Forward,PE_Forward,PE_Trailing,Year_High,Year_Low,Change,Change_pc,Prev_Close)
        #print(symbol)
        
        self.__write_data_to_csv({'Name' : self.name, 'Symbol': self.symbol, 'Price': self.Price, 'Market Cap': self.Market_Cap , 'EPS Trailing': self.EPS_Trailing, 'EPS Forward': self.EPS_Forward, 'Change' : self.Change, 'Change%': self.Change_pc, 'Prev Close': self.Prev_Close, 'Volume': self.Volume, 'Year Low': self.Year_Low, 'Year High': self.Year_High, 'P/E Trailing': self.PE_Trailing, 'P/E Forward': self.PE_Trailing, 'Dividend': self.Dividend, 'Dividend Yield': self.Dividend_Yield, 'Target High': self.High, 'Target Avg': self.Avg, 'Target Low': self.Low})
        out.close()
        os.remove('out.txt')
        
    def __get_target_prices(self):
        print('Getting Target price for '+self.name)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
        link = 'https://www.webull.com/quote/nasdaq-%s/analysis'%self.symbol.lower()
        home = requests.get(link,headers=headers)
        souphead = BeautifulSoup(home.content, 'html.parser')
        
        inp = open("target.txt",'w')
        inp.write(str(souphead.prettify().encode("utf-8")))
        inp.close()
        out = open("target.txt",'r')
        for line in out:
           # print(line)
            lines = line.split('>')
            for i in range(len(lines)):
                
                if len(re.findall('\D+stock price target\D+',lines[i])) > 0:
                    sets = re.findall(r'\d+\.\d+', lines[i])
                    if len(sets) > 0:
                        self.Avg = float(sets[0])
                        self.High = float(sets[1])
                        self.Low = float(sets[2])
 
        out.close()  
        os.remove("target.txt")

    def __stock_prediction(self):

        ser = self.hist['Close'].squeeze()
        if len(ser)<252:
            val = ser[len(ser)*-1]
        else:
            val = ser[-251]
        #Stock prediction
        self.valuation = (self.Price * self.avg_pe)/self.PE_Trailing
        self.growth_factor = self.Price*self.EPS_Forward/self.EPS_Trailing
        self.Volatility_factor = (self.Year_High-self.Year_Low)/self.Year_Low
        self.Dividend_factor = self.Dividend
        self.Trust_factor = 1 + self.Market_Cap/self.market_base
        self.Trend_factor = (self.Price - val)/val

    def __get_target_price(self):    
        self.Value_price = (((self.valuation-self.Price))*(self.Volatility_factor * 0.5))  + self.Price
        
        self.Growth_price = self.Value_price +  ((self.growth_factor-self.Price))*(self.Volatility_factor) 
        self.Trend_price = self.Growth_price + self.Dividend_factor*4 + self.Trend_factor*self.Price*0.1
        self.Target_price = self.Trend_price * self.Trust_factor  

    def __get_ratings(self):
        self.value_rating = (self.valuation / self.Price) * 3.33
        self.growth_rating = self.growth_factor * 0.5
        if (self.growth_factor > 10):
            self.growth_rating = 10.0
        
        self.trend_rating = 5 + self.Trend_factor * 3.33
        if (self.Trend_factor > 10):
            self.trend_rating = 10.0
        self.trust_rating = math.sqrt(self.Trust_factor-1) * 100
        if (self.trust_rating > 10):
            self.trust_rating = 10.0
        
            

        self.dividend_rating = (self.Dividend_Yield**2)*10
        if self.dividend_rating > 10:
            self.dividend_rating = 10.0


    def __str__(self):
        print()
        print()
        print("----------------------   "+self.name+"   ----------------------")
        print()
        print("************* Financial Information *************")
        print("Symbol: "+self.symbol)
        print("Current Stock Price: $",str(self.Price))
        print("Market Cap: $",self.market_cap)
        print("P/E: ",str(self.PE_Trailing))
        print("Earnings Per Share(EPS): $",str(self.EPS_Trailing))
        print("Change: $",str(self.Change))
        print("Change%: ",str(self.Change_pc),"%")
        print("Year Low: $",str(self.Year_Low))
        print("Year High: $",str(self.Year_High))
        print("Trade Volume: ",str(self.Volume))
        print("Dividend: $",str(self.Dividend))
        print()
        print()


        print("----------------------    Stock Ratings      ------------------")
        print("Valuation Rating: "+ str(self.value_rating))
        print("Growth Rating: "+ str(self.growth_rating))
        print("Trend Rating: "+ str(self.trend_rating))
        print("Trust Rating: "+ str(self.trust_rating))
        print("Sharing Profits Rating: "+ str(self.dividend_rating))
        print()
        print()
        print("---------------------- Prediction Statistics ------------------")
        print("Valuation of the company: $"+str(self.valuation))
        print('Stock Price adjusting valuation: $' + str(self.Value_price))       
        print('Stock Price adjusting growth: $' + str(self.Growth_price))
        print('Stock Price adjusting trends: $' + str(self.Trend_price))
        print('Final Predicted Stock Price in a year: $' + str(self.Target_price)) 
        print("Growth percentage expected: "+str(((self.Target_price - self.Price)/self.Price)*100))
        print()
        print('Short term predicted Stock Price in a week: $' + str(self.Price*1.06)) 
        print()
        print()

        print("---------------------- Other Analysts Data ----------------------")
        print("Target Price High: $",str(self.High))
        print("Target Price Avg: $",str(self.Avg))
        print("Target Price Low: $",str(self.Low))
        print()
        print()
        return "" 

    def get_plot(self, *args):
        if len(args) == 0:
            span = '1y'
        else:
            span = args[0]

        hist = self.yahoo.history(period=span)
        
        ser = hist['Close'].squeeze()
        ser.plot()
        plt.ylabel('Dollars')
        plt.title(self.name + ' ' + self.symbol)
        plt.show()
    
    def get_name(self):
        return self.name

    def get_symbol(self):
        return self.symbol

    def get_target_prices(self):
        return self.High,self.Avg,self.Low,self.Target_price
    

    def get_market_trend(self):
        hist = self.yahoo.history(period='1y')
        
        ser = hist['Close'].squeeze()
        return ser


def get_all_symbols():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
    home = requests.get('https://www.slickcharts.com/nasdaq100',headers=headers)
    souphead = BeautifulSoup(home.content, 'html.parser')

    inp = open("stocklist.csv",'w',newline='')

    stocklist = list()
    NameList = list()
    cur = 'name'
    for tag in souphead.findAll('td'):
        for stock in tag.find_all('a'):
            if(cur == 'name'):   
                Name=str(stock.text)
                
                cur = 'sym'
            else:
                Symbol=str(stock.text)
                NameList.append({'Name': Name,'Symbol': Symbol})
                if(len(Symbol)>0):
                    stocklist.append({'Name': Name, 'Symbol' : Symbol, 'Link': ('https://www.slickcharts.com/symbol/'+Symbol+'/')})
                cur = 'name'
    
    writer = csv.DictWriter(inp,fieldnames=['Name','Symbol','Link'])
    writer.writeheader()
    writer.writerows(stocklist[:-2])
    inp.close()
    return NameList[:-2]


    

def scrape_function(keywordlist = [],plotlist = [],csv_file = 'stocks.csv'):
    keyword_list=['AAPL','MSFT','AMZN','GOOG','TSLA']
    for items in keywordlist:
        if items not in keyword_list:
            keyword_list.append(items)
    
    fieldnames = ['Name','Symbol', 'Price', 'Market Cap', 'EPS Trailing', 'EPS Forward', 'Change', 'Change%', 'Prev Close', 'Volume', 'Year Low', 'Year High', 'P/E Trailing', 'P/E Forward', 'Dividend', 'Dividend Yield', 'Target High', 'Target Avg', 'Target Low']
    with open(csv_file, 'w') as f_object:
        dictwriter_object = csv.DictWriter(f_object, fieldnames=fieldnames)
        dictwriter_object.writeheader()
        f_object.close()

    

    for keyword in keyword_list:
        Stock = stock(keyword = keyword)
        
        print(Stock)
    if len(plotlist) > 0:
        plt.ylabel('Dollars')
        plt.title('Stock Trends')
        thigh = []
        tlow = []
        tavg = []
        tpre = []
        names = []

    else:
        return
    
    for plot in plotlist:
        stk = stock(plot)
        plt.plot(stk.get_market_trend(),label = stk.get_name() + ' (' + stk.get_symbol() + ')')
        H,A,L,P = stk.get_target_prices()
        names.append(stk.get_name() + ' (' + stk.get_symbol() + ')')
        print(str(H) + ' ' + str(A)  +' ' + str(L) + ' ' + str(P))
        thigh.append(H)
        tlow.append(L)
        tavg.append(A)
        tpre.append(P)
        

    plt.legend()
    plt.show()    
        #stock(plot).get_plot()
    # set width of bar
    print(tlow)
    print(tavg)

    barWidth = 0.1
    fig = plt.subplots(figsize =(len(plotlist)*4, 8))
    br1 = np.arange(len(thigh))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    plt.bar(br1, thigh, color ='r', width = barWidth,
        edgecolor ='grey', label ='High')
    plt.bar(br2, tavg, color ='g', width = barWidth,
        edgecolor ='grey', label ='Avg')
    plt.bar(br3, tlow, color ='y', width = barWidth,
        edgecolor ='grey', label ='Low')
    plt.bar(br4, tpre, color ='b', width = barWidth,
        edgecolor ='grey', label ='Predicted')
    plt.xlabel('Stocks', fontweight ='bold', fontsize = 15)
    plt.ylabel('Price in Dollars', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(thigh))],
        names)
    plt.title('Prediction Comparison')
    plt.legend()
    plt.show()

def static_function(path='stocks.csv'):
    fieldnames = ['Name','Symbol', 'Price', 'Market Cap', 'EPS Trailing', 'EPS Forward', 'Change', 'Change%', 'Prev Close', 'Volume', 'Year Low', 'Year High', 'P/E Trailing', 'P/E Forward', 'Dividend', 'Dividend Yield', 'Target High', 'Target Avg', 'Target Low']
    with open(path, 'r') as f_object:
        inputfile = csv.DictReader(f_object, fieldnames=fieldnames)
        next(inputfile)
        for row in inputfile:
            Stock = stock(row['Symbol'],csvfile = path, mode="static_mode")
            print(Stock)
        f_object.close


def default_function():
    List = get_all_symbols()
    Names =[]
    Symbols = []
    for item in List:
        Names.append(item['Name'])
        Symbols.append(item['Symbol'])
    scrape_function(Names)

if __name__=='__main__':
    
    
    if len(sys.argv) ==1:
        default_function()

    elif sys.argv[1] == '--scrape':
        if len(sys.argv) > 2:
            keywords = list()
            plots = list()
            plotparam = False            
            for i in range(2,len(sys.argv)):
                if(sys.argv[i] == '--plot'):
                    plotparam = True
                    continue
                
                if(plotparam == False):
                    keywords.append(sys.argv[i])
                else:
                    plots.append(sys.argv[i])
                
            scrape_function(keywords,plots)
        else:
            scrape_function()
    
    elif  sys.argv[1] == '--static':
        if len(sys.argv) == 3:
            static_function(sys.argv[2])
        else:
            static_function()

#scrape_function()

