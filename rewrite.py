import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import neat
import csv
from binance.client import Client
import pickle
import math
import time

def get_binance_historical_data_1m(asset):
    API_KEY = 'xYfA667iJQon8BfQBb2jX4bVwzONGFGHi4r0a4g3qSRbUdGzpPHKksoK11sPMsub'
    SECRET_KEY = 's4lil2XCKjfSbt1MrxfTYR2HupL1SyUXqGcpzeDy9N11HMAHex5eH'

    client = Client(API_KEY, SECRET_KEY)
    btcusd_history = client.get_historical_klines(asset, Client.KLINE_INTERVAL_1MINUTE, "15 Feb, 2021")

    columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    btcusd_history_df = pd.DataFrame(btcusd_history, columns=columns)
    btcusd_history_df['time'] = pd.to_datetime(btcusd_history_df['time'], unit='ms')
    del btcusd_history_df['close_time']
    del btcusd_history_df['quote_asset_volume']
    del btcusd_history_df['number_of_trades']
    del btcusd_history_df['taker_buy_base_asset_volume']
    del btcusd_history_df['taker_buy_quote_asset_volume']
    del btcusd_history_df['ignore']
    cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    btcusd_history_df.columns = cols
    for col in cols:
        if col != 'Datetime': btcusd_history_df[col] = btcusd_history_df[col].astype(float)
    btcusd_history_df['EMA_100'] = btcusd_history_df['Close'].rolling(window=100).mean()
    btcusd_history_df['Lag_1'] = btcusd_history_df.Close - btcusd_history_df.Close.shift(1)
    btcusd_history_df = btcusd_history_df.dropna()
    df_yf = btcusd_history_df
    df_yf['Lag_1'] = df_yf['Close'] - df_yf['Close'].shift(1)
    df_yf['Lag_2'] = df_yf['Close'].shift(1) - df_yf['Close'].shift(2)
    df_yf['Lag_3'] = df_yf['Close'].shift(2) - df_yf['Close'].shift(3)
    df_yf['Lag_4'] = df_yf['Close'].shift(3) - df_yf['Close'].shift(4)
    df_yf['Lag_5'] = df_yf['Close'].shift(4) - df_yf['Close'].shift(5)

    df_yf['EMA_200'] = df_yf['Close'].rolling(window=200).mean()


    #lags

    df_yf = df_yf.dropna()
    #df_yf.to_csv('df_yf.csv')
    return df_yf

class DataFrame_AI_Getter:
    m_df = 0
    m_method = ''
    def __init__(self, asset, method):
        m_method = method
        if method == 'yf':
            self.m_df = yf.download(asset, interval='1m', period='7d')  # start='2021-12-01', end='2021-12-05')
        elif method == 'binance':
            API_KEY = 'xYfA667iJQon8BfQBb2jX4bVwzONGFGHi4r0a4g3qSRbUdGzpPHKksoK11sPMsub'
            SECRET_KEY = 's4lil2XCKjfSbt1MrxfTYR2HupL1SyUXqGcpzeDy9N11HMAHex5eH'

            client = Client(API_KEY, SECRET_KEY)
            btcusd_history = client.get_historical_klines(asset, Client.KLINE_INTERVAL_1MINUTE, "1 Feb, 2022")
            columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                           'number_of_trades',
                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            btcusd_history_df = pd.DataFrame(btcusd_history, columns=columns)
            btcusd_history_df['time'] = pd.to_datetime(btcusd_history_df['time'], unit='ms')
            del btcusd_history_df['close_time']
            del btcusd_history_df['quote_asset_volume']
            del btcusd_history_df['number_of_trades']
            del btcusd_history_df['taker_buy_base_asset_volume']
            del btcusd_history_df['taker_buy_quote_asset_volume']
            del btcusd_history_df['ignore']
            cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            btcusd_history_df.columns = cols
            for col in cols:
                if col != 'Datetime': btcusd_history_df[col] = btcusd_history_df[col].astype(float)
            self.m_df = btcusd_history_df
    # def __init__(self, filename):
    #     self.m_df = self.load_from_file(filename)
    #     m_method = 'load'

    def config_df(self):
        self.m_df = btcusd_history_df
        self.m_df['EMA_100'] = self.m_df['Close'].rolling(window=100).mean()
        self.m_df['Lag_1'] = self.m_df.Close - self.m_df.Close.shift(1)
        self.m_df = self.m_df.dropna()
        return self.m_df

    def load_from_file(self, filename):
        self.m_df = pd.read_csv(filename, delimiter='\t')
        return self.m_df

    def save_df_to_file(self, df, filename):
        df.to_csv(filename, sep='\t')

    def get_average_diff(self, df):
        df['Lag_1'] = df.Close - df.Close.shift(1)
        df = df.dropna()
        sum = 0.0
        for i in range(len(df)):
            sum += df.Lag_1.iloc[i]
        return sum / len(df)

class Trader:
    m_balance = 50
    m_deals = []
    m_train_df = 0
    # deal structure:
    # [x] - deal
    # [x][0] - open_price
    # [x][1] - eq
    # [x][2] - True if long, False if short
    # [x][3] - leverage
    # [x][4] - liquidation price
    # [x][5] - start_marj
    # [x][6] - balance before creating deal
    # [x][7] - start_index
    # [x][8] - deal_size in dollars
    # [x][9] - max_open
    # [x][10] - start_marj
    m_deals_counter = 0
    m_balance_changes = []
    m_net = 0
    m_trading_fee_multiplier = 0.99925
    m_logs = []
    m_is_trader_removed = False

    def __init__(self, net, start_balance = 50):
        self.m_balance = start_balance
        self.m_deals.clear()
        self.m_balance_changes.clear()
        self.m_logs.clear()
        self.m_logs.append(['Event', 'Type', 'Event_price', 'Deal_size', 'Max_open', 'Eq', 'Start_marj', 'Liquidation_price', 'Balance_before', 'Balance_after', 'Leverage', 'Event_index'])
        self.m_net = net

    def getDecision(self, input_net_data):
        return self.m_net.activate(input_net_data)

    def addLog(self, event, type, event_price, deal_size, max_open, eq, start_marj, liquidation_price, balance_before, balance_after, leverage, event_index):
        event_info = [ event, type, event_price, deal_size, max_open, eq, start_marj, liquidation_price, balance_before, balance_after, leverage, event_index ]
        self.m_logs.append(event_info)

    def saveLogs(self, filename):
        with open(filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for r in self.m_logs:
                writer.writerow(r)

    def getEq(self, start_price, start_marj, leverage):
        return (start_marj * leverage) / (start_price * 1.0116)

    # returns max possible count of coins to buy
    def getMaxOpen(self, balance, price, leverage):
        return (balance / price) * leverage

    # returns start price to create a deal
    def getStartMarj(self, start_price, eq, leverage):
        return eq * start_price * 1.0116 / leverage

    # returns liquidation_price,, this is a price when you loose all your money (all balance)
    def getLiquidationPrice(self, isLong, open_price, leverage, max_open, eq):
        if isLong:
            return open_price * (1 - (1 / leverage) * (max_open / eq)) * 1.0066
        return open_price * (1 + (1 / leverage) * (max_open / eq)) * 0.993524

    def addNewBalanceChanges(self):
        if len(self.m_balance_changes):
            if not self.m_balance == self.m_balance_changes[-1]:
                self.m_balance_changes.append(self.m_balance)
        else:
            self.m_balance_changes.append(self.m_balance)

    def addDeal(self, open_price, deal_size, direction, leverage, start_index):
        if(len(self.m_balance_changes)):
            if self.m_balance != self.m_balance_changes[-1]:
                self.m_balance_changes.append(self.m_balance)
        else:
            self.m_balance_changes.append(self.m_balance)

        eq = self.getEq(open_price, deal_size, leverage)
        start_marj = self.getStartMarj(open_price, eq, leverage)
        max_open = self.getMaxOpen(self.m_balance, open_price, leverage)
        liquidation_price = self.getLiquidationPrice(direction, open_price, leverage, max_open, eq)

        self.m_deals = [open_price, eq * self.m_trading_fee_multiplier,
                             direction, leverage, liquidation_price,
                             start_marj, self.m_balance, start_index,
                             deal_size, max_open]
        self.addLog('Create', direction, open_price, deal_size, max_open, eq * self.m_trading_fee_multiplier,
                    start_marj, liquidation_price, self.m_balance, self.m_balance - start_marj, leverage, start_index)
        self.m_balance -= start_marj

    def closeDeals(self, close_price, end_index):
        final_profit = 0
        balance_before = self.m_balance
        deal = self.m_deals
        if len(deal):
            if deal[2]:
                self.m_balance += deal[5]
                dif = (close_price / deal[0] - 1) * deal[3]  # final_local_balance / start_marj - 1
                final_profit = deal[5] * dif * self.m_trading_fee_multiplier
                self.m_balance += final_profit
            else:
                self.m_balance += deal[5]
                dif = ((deal[0] - close_price) / deal[0]) * deal[3]
                final_profit = deal[5] * dif * self.m_trading_fee_multiplier
                self.m_balance += final_profit

            self.addLog('Close', deal[2], close_price, deal[8], deal[9], deal[1],
                       deal[5], deal[4], balance_before, self.m_balance, deal[3],
                       end_index)
            self.m_deals_counter += 1
            self.addNewBalanceChanges()
        self.m_deals.clear()

        if self.m_balance <= 0: self.m_is_trader_removed = True
        return final_profit

    def plot_balance_changing(self):
        plt.figure(figsize=(20, 7))
        plt.plot(self.m_balance_changes)
        plt.show()

    def ne_decision_handler(self, decision, current_data, index, leverage, is_it_evol=False, genomes=[], trader_index=0):
        deal_size = self.m_balance * 0.5
        if self.m_balance <= 0:
            self.m_is_trader_removed = True
            return 0

        if len(self.m_deals):
            if self.m_deals[2] == True:
                if current_data.Low <= self.m_deals[4]:
                    self.m_is_trader_removed = True
                    return 0
            else:
                if current_data.High >= self.m_deals[4]:
                    self.m_is_trader_removed = True
                    return 0

        if self.m_is_trader_removed == False:
            if decision[0] > decision[1]:
                if len(self.m_deals) > 0 and self.m_deals[2] == False:
                    final_profit = self.closeDeals(current_data.High, index)
                    if is_it_evol: genomes[trader_index].fitness += final_profit ** 3
                    deal_size = self.m_balance * 0.5
                    self.addDeal(current_data.High, deal_size, True, leverage, index)
                    return 1
                elif len(self.m_deals) == 0:
                    self.addDeal(current_data.High, deal_size, True, leverage, index)
                    return 1
            elif decision[0] < decision[1]:
                if len(self.m_deals) > 0 and self.m_deals[2] == True:
                    final_profit = self.closeDeals(current_data.Low, index)
                    if is_it_evol: genomes[trader_index].fitness += final_profit ** 3
                    deal_size = self.m_balance * 0.5
                    self.addDeal(current_data.Low, deal_size, False, leverage, index)
                    return 1
                elif len(self.m_deals) == 0:
                    self.addDeal(current_data.Low, deal_size, False, leverage, index)
                    return 1

        if index % 1000 == 0 and self.m_deals_counter <= 0:
            self.m_is_trader_removed = True
            return 0

    def test_on_df(self, df, leverage):
        self.m_train_df = df
        self.m_balance_changes.clear()
        self.m_deals_counter = 0
        self.m_deals.clear()
        self.m_logs.clear()
        self.m_is_trader_removed = False
        self.m_logs.append(
            ['Event', 'Type', 'Event_price', 'Deal_size', 'Max_open', 'Eq', 'Start_marj', 'Liquidation_price',
             'Balance_before', 'Balance_after', 'Leverage', 'Event_index'])
        for i in range(1, len(df)):
            if self.m_is_trader_removed == False:
                net_input_data = [self.m_train_df.EMA_100.iloc[i], self.m_train_df.Lag_1.iloc[i],
                                  self.m_train_df.Volume.iloc[i], self.m_train_df.Close.iloc[i]]
                """net_input_data = [self.m_train_df.EMA_100.iloc[i], self.m_train_df.Lag_1.iloc[i],
                                  self.m_train_df.Volume.iloc[i], self.m_train_df.Close.iloc[i],
                                  self.m_train_df.EMA_5.iloc[i], self.m_train_df.EMA_20.iloc[i],
                                  self.m_train_df.EMA_50.iloc[i], self.m_train_df.EMA_150.iloc[i],
                                  self.m_train_df.EMA_200.iloc[i],
                                  self.m_train_df.EMAD_5_20.iloc[i], self.m_train_df.EMAD_5_50.iloc[i],
                                  self.m_train_df.EMAD_5_100.iloc[i], self.m_train_df.EMAD_5_150.iloc[i],
                                  self.m_train_df.EMAD_5_200.iloc[i],
                                  self.m_train_df.EMAD_20_50.iloc[i], self.m_train_df.EMAD_20_100.iloc[i],
                                  self.m_train_df.EMAD_20_150.iloc[i], self.m_train_df.EMAD_20_200.iloc[i],
                                  self.m_train_df.EMAD_50_100.iloc[i], self.m_train_df.EMAD_50_150.iloc[i],
                                  self.m_train_df.EMAD_50_200.iloc[i],
                                  self.m_train_df.EMAD_100_150.iloc[i], self.m_train_df.EMAD_100_200.iloc[i],
                                  self.m_train_df.EMAD_150_200.iloc[i],
                                  self.m_train_df.EMAD_5_20_LAG_1.iloc[i], self.m_train_df.EMAD_5_50_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_5_100_LAG_1.iloc[i], self.m_train_df.EMAD_5_150_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_5_200_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_20_50_LAG_1.iloc[i], self.m_train_df.EMAD_20_100_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_20_150_LAG_1.iloc[i], self.m_train_df.EMAD_20_200_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_50_100_LAG_1.iloc[i], self.m_train_df.EMAD_50_150_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_50_200_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_100_150_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_100_200_LAG_1.iloc[i],
                                  self.m_train_df.EMAD_150_200_LAG_1.iloc[i],
                                  ]"""
                decision = self.getDecision(net_input_data)
                self.ne_decision_handler(decision, self.m_train_df.iloc[i], i, leverage)
        self.plot_balance_changing()
        self.saveLogs('test_on_df_logs.csv')
        print("Balance: " + str(self.m_balance))
        print("Is_trader_removed: " + str(self.m_is_trader_removed))
        print("Deals_counter: " + str(self.m_deals_counter))


class NEAT_Trading:
    m_traders = []
    m_genomes = []
    m_config = 0
    m_train_df = []
    m_leverage = 0
    m_start_balance = 0

    def __init__(self, config_path, train_df, leverage, start_balance):
        self.m_traders.clear()
        self.m_genomes.clear()
        self.m_leverage = leverage
        self.m_config = config_path
        self.m_train_df = train_df
        self.m_start_balance = start_balance

    def removeTrader(self, trader, trader_index):
        trader.m_is_trader_removed = True
        self.m_genomes[trader_index].fitness *= -1

    def run(self, config_path, count_of_populations):
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        pop = neat.Population(config)
        pop.run(self.eval_genomes, count_of_populations)

    def ne_decision_handler(self, trader, decision, current_data, trader_index, df_index, is_it_evol):
        result_of_handling = trader.ne_decision_handler(decision, current_data, df_index, self.m_leverage, is_it_evol, self.m_genomes, trader_index)
        if result_of_handling == 0:
            self.removeTrader(trader, trader_index)

    def eval_genomes(self, genomes, config):
        self.m_traders.clear()
        self.m_genomes.clear()
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.m_traders.append(Trader(net, 50))
            genome.fitness = 0
            self.m_genomes.append(genome)

        for index, trader in enumerate(self.m_traders):
            trader.m_balance_changes.clear()
            trader.m_deals_counter = 0
            trader.m_deals.clear()
            trader.m_logs.clear()
            trader.m_logs.append(
                ['Event', 'Type', 'Event_price', 'Deal_size', 'Max_open', 'Eq', 'Start_marj', 'Liquidation_price',
                 'Balance_before', 'Balance_after', 'Leverage', 'Event_index'])
            prev_decisions = []
            for i in range(len(self.m_train_df)):
                if i > 0:
                    if trader.m_is_trader_removed == False:
                        net_input_data = [self.m_train_df.EMA_100.iloc[i], self.m_train_df.Lag_1.iloc[i],
                                          self.m_train_df.Volume.iloc[i], self.m_train_df.Close.iloc[i]]
                        decision = trader.getDecision(net_input_data)
                        if len(prev_decisions):
                            if decision != prev_decisions[-1]:
                                prev_decisions.append(decision)
                        else:
                            prev_decisions.append(decision)
                        self.ne_decision_handler(trader, decision, self.m_train_df.iloc[i], index, i, True)

            #print(prev_decisions)
            #if trader.m_balance > self.m_start_balance:
            if trader.m_balance > 50:
                #self.m_genomes[index].fitness += int(trader.m_balance - self.m_start_balance) ** 3
                print("********************************************************")
                print("Balance: " + str(trader.m_balance))
                print("Is_trader_removed: " + str(trader.m_is_trader_removed))
                print("Deals_counter: " + str(trader.m_deals_counter))
                trader.plot_balance_changing()
                trader.saveLogs('newLogggg.csv')
                with open('best_trader.pkl', 'wb') as outp:
                    pickle.dump(trader.m_net, outp, pickle.HIGHEST_PROTOCOL)
                time.sleep(60)




# LTC_df = yf.download('BNB-USD', interval='1m', period='7d')  # start='2021-12-01', end='2021-12-05')
# #LTC_df['EMA_100'] = LTC_df['Close'].ewm(span=100, adjust=False).mean()
# LTC_df['EMA_100'] = LTC_df['Close'].rolling(window=100).mean()
# LTC_df['Lag_1'] = LTC_df.Close - LTC_df.Close.shift(1)
# LTC_df = LTC_df.dropna()
# print(LTC_df)



# config_path = 'config.txt'
# ne = NEAT_Trading(config_path, LTC_df, 30, 50)
# print("HERE")
# ne.run(config_path, 99999)




#df = get_binance_historical_data_1m('BTCUSDT')
#df.to_csv('df_binance.csv')
#print(df)
# df_yf = yf.download('BTC-USD', interval='1m', period='7d')
# df_yf['Lag_1'] = df_yf['Close'] - df_yf['Close'].shift(1)
# df_yf['EMA_200'] = df_yf['Close'].rolling(window=200).mean()
# df_yf = df_yf.dropna()
#df_yf.to_csv('df_yf.csv')



df_yf_csv = pd.read_csv('df_yf.csv')
df_yf_csv = df_yf_csv.dropna()
# #print(df_yf_csv)
# df = pd.read_csv('df_binance.csv')

# print(df_yf_csv)
# print(df)


# for i in range(100):
#     print(str(df.iloc[i].Close) + ' ' + str(df.iloc[i].EMA_100))
#     print("**********************************")

df = pd.read_csv('df.csv')

best_net = 0
with open('best_trader.pkl', 'rb') as inp:
    best_net = pickle.load(inp)
#
trader = Trader(best_net)
trader.test_on_df(df_yf_csv, 30)