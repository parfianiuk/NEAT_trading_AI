import yfinance as yf
import pandas as pd
import numpy as np
import neat
import os
import sys
import math
import matplotlib.pyplot as plt
import csv
import time
from binance.spot import Spot as Client
import pickle

def get_binance_df(start, end):
    base_url = "https://api.binance.com"
    base_url_test = "https://testnet.binance.vision"
    spot_client = Client(base_url=base_url)
    btcusd_history = spot_client.klines("BTCUSDT", "1m", limit=1000)
    # btcusd_history = spot_client.klines("BTCUSDT", "1m", )
    # show as DataFrame
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


    #btcusd_history_df['EMA_100'] = btcusd_history_df['Close'].ewm(span=100, adjust=False).mean()
    btcusd_history_df.to_csv("TEST.csv", sep='\t')
    btcusd_history_df['EMA_100'] = btcusd_history_df['Close'].rolling(window=100).mean()
    btcusd_history_df['Lag_1'] = btcusd_history_df.Close - btcusd_history_df.Close.shift(1)
    btcusd_history_df = btcusd_history_df.dropna()
    #print(btcusd_history_df['Datetime'].iloc[0])
    btcusd_history_df.reset_index(drop=True)
    print(btcusd_history_df)

    btcusd_history_df.to_csv("TEST_1.csv", sep='\t')

    df = yf.download('BTC-USD', interval='1m', period='7d')  # start='2021-12-01', end='2021-12-05')
    df.reset_index(drop=True, inplace=True)
    print(df)

    df.to_csv("TEST_2.csv", sep='\t')

    return btcusd_history_df

class Trader:
    m_balance = 50
    m_rev_balance = 1000
    m_deals = []
    m_rev_deals = []
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
    # [x][10] - eq
    # [x][11] - start_marj
    m_deals_counter = 0
    m_rev_deals_counter = 0
    m_balances = []
    m_rev_balances = []
    m_net = 0
    m_trading_fee_multiplier = 0.99925
    m_deals_log = []
    m_detail_log = []
    m_is_trader_removed = False

    def __init__(self, net):
        self.m_balance = 50
        self.m_rev_balance = 1000
        self.m_deals.clear()
        self.m_rev_deals.clear()
        self.m_balances.clear()
        self.m_rev_balances.clear()
        self.m_deals_log.clear()
        self.m_deals_log.append(
            ['Type', 'Open_price', 'Close_price', 'Deal_size', 'Max_open', 'eq', 'start_marj', 'Liquidation_price',
             'Balance_before', 'Balance_after', 'Leverage', 'start_index', 'end_index'])
        self.m_net = net

    def getDecision(self, input_net_data):
        return self.m_net.activate(input_net_data)

    def addLog(self, deal, close_price, balance_before, balance_after, start_index, end_index, deal_size, max_open, eq,
               start_marj):
        deal_info = [deal[2], deal[0], close_price, deal_size, max_open, eq, start_marj, deal[4], balance_before,
                     balance_after, deal[3], start_index, end_index]
        self.m_deals_log.append(deal_info)

    def saveLogs(self, filename):
        with open(filename, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            for r in self.m_deals_log:
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

        # deals_counter changing

    # balances adding
    # balance changing
    # deals changing
    def addDeal(self, open_price, deal_size, direction, leverage, start_index):
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
        # [x][10] - eq
        # [x][11] - start_marj
        if len(self.m_balances):
            if not self.m_balance == self.m_balances[-1]:
                self.m_balances.append(self.m_balance)
        else:
            self.m_balances.append(self.m_balance)
        self.closeDeal(open_price, start_index)
        eq = self.getEq(open_price, deal_size, leverage)
        start_marj = self.getStartMarj(open_price, eq, leverage)
        max_open = self.getMaxOpen(self.m_balance, open_price, leverage)
        liquidation_price = self.getLiquidationPrice(direction, open_price, leverage, max_open, eq)
        self.m_deals = [open_price, eq * self.m_trading_fee_multiplier, direction, leverage, liquidation_price,
                        start_marj, self.m_balance, start_index, deal_size, max_open, eq, start_marj]
        self.m_balance -= start_marj
        self.m_deals_counter += 1

    # deals changing
    # balance changing
    # balances changing
    def closeDeal(self, close_price, end_index):
        final_profit = 0
        if len(self.m_deals):
            deal = self.m_deals
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
            self.addLog(deal, close_price, deal[6], self.m_balance, deal[7], end_index, deal[8], deal[9], deal[10],
                        deal[11])
            self.m_deals.clear()
        return final_profit

    def ne_decision_handler(self, decision, current_data, df_index, leverage):
        deal_size = self.m_balance * 0.50
        if self.m_balance < 0:
            self.m_is_trader_removed = True
            return 0

        if len(self.m_deals) > 0:
            if self.m_deals[2] == True:
                if current_data.Low < self.m_deals[4]:
                    self.m_is_trader_removed = True
                    return 0
            else:
                if current_data.High > self.m_deals[4]:
                    self.m_is_trader_removed = True
                    return 0

        if self.m_is_trader_removed == False:
            if decision[0] > decision[1]:
                if len(self.m_deals) > 0 and self.m_deals[2] == False:
                    self.closeDeal(current_data.Close, df_index)
                    deal_size = self.m_balance * 0.50
                    self.addDeal(current_data.Close, deal_size, True, leverage, df_index)
                    return 1
                elif len(self.m_deals) == 0:
                    self.addDeal(current_data.Close, deal_size, True, leverage, df_index)
            elif decision[0] < decision[1]:
                if len(self.m_deals) > 0 and self.m_deals[2] == True:
                    self.closeDeal(current_data.Close, df_index)
                    deal_size = self.m_balance * 0.50
                    self.addDeal(current_data.Close, deal_size, False, leverage, df_index)
                    return 1
                elif len(self.m_deals) == 0:
                    self.addDeal(current_data.Close, deal_size, False, leverage, df_index)

        if df_index % 1000 == 0 and self.m_deals_counter <= 0:
            self.m_is_trader_removed = True
            return 0

    def test_on_df(self, test_df, log_filename, leverage):
        start_balance = self.m_balance
        start_balances = self.m_balances
        start_deals_counter = self.m_deals_counter
        start_logs = self.m_deals_log
        start_is_trader_removed = self.m_is_trader_removed

        self.m_balance = 50
        self.m_balances.clear()
        self.m_deals.clear()
        self.m_deals_counter = 0
        self.m_deals_log.clear()
        self.m_deals_log.append(
            ['Type', 'Open_price', 'Close_price', 'Deal_size', 'Max_open', 'eq', 'start_marj', 'Liquidation_price',
             'Balance_before', 'Balance_after', 'Leverage', 'start_index', 'end_index'])
        self.m_is_trader_removed = False

        for i in range(len(test_df)):
            if i > 0:
                if self.m_is_trader_removed == False:
                    net_input_data = [test_df.EMA_100.iloc[i], test_df.Lag_1.iloc[i], test_df.Volume.iloc[i],
                                      test_df.Close.iloc[i]]
                    #df_net_input_data = [df.EMA_100.iloc[i], df.Lag_1.iloc[i], df.Volume.iloc[i],
                    #                  df.Close.iloc[i]]
                    #decision_df = self.getDecision(df_net_input_data)
                    decision = self.getDecision(net_input_data)
                    #print("Inp: " + str(net_input_data) + " Df_inp: " + str(df_net_input_data))
                    #print("Decision: " + str(decision) + " Decision_df: " + str(decision_df))
                    self.ne_decision_handler(decision, test_df.iloc[i], i, leverage)
        if self.m_is_trader_removed == False:
            self.closeDeal(test_df.Close.iloc[-1], len(test_df) - 1)
        else:
            self.m_balance = 0

        self.saveLogs(log_filename)

        final_results = [self.m_balance, self.m_deals_counter, self.m_is_trader_removed]

        self.plot_balance_changing(test_df)

        self.m_balance = start_balance
        self.m_balances = start_balances
        self.m_deals_counter = start_deals_counter
        self.m_deals_log = start_logs
        self.m_is_trader_removed = start_is_trader_removed

        return final_results

    def plot_balance_changing(self, frame):
        buy_dates = []
        sell_dates = []

        for i in range(len(self.m_deals_log)):
            if i > 0:
                if self.m_deals_log[i][0] == True:
                    buy_dates.append(self.m_deals_log[i][11])
                    sell_dates.append(self.m_deals_log[i][12])
                else:
                    buy_dates.append(self.m_deals_log[i][12])
                    sell_dates.append(self.m_deals_log[i][11])

        #plt.figure(figsize=(20, 7))
        #plt.scatter(frame.iloc[buy_dates].index, frame.iloc[buy_dates]['Close'], marker='^', c='g')
        #plt.scatter(frame.iloc[sell_dates].index, frame.iloc[sell_dates]['Close'], marker='v', c='r')
        #plt.plot(frame['Close'], alpha=0.7)

        pd.options.mode.chained_assignment = None
        fframe = []
        fframe = pd.DataFrame(self.m_balances, index=list(range(0, len(self.m_balances))))
        fframe['index'] = list(range(0, len(self.m_balances)))
        fframe.columns = {'balance', 'index'}


        plt.figure(figsize=(20, 7))
        plt.plot(fframe['balance'], alpha=0.7)
        plt.show()

class NEAT_Trading:
    m_traders = []
    m_genomes = []
    m_config = 0
    m_train_df = []
    m_leverage = []
    m_best_result = 1000

    def __init__(self, config_path, train_df, leverage):
        self.m_traders.clear()
        self.m_genomes.clear()
        self.m_leverage = leverage
        self.m_config = config_path
        self.m_train_df = train_df
        self.configuration_df(self.m_train_df)
        self.m_best_result = 1000

    def configuration_df(self, df):
        df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['Lag_1'] = df.Close - df.Close.shift(1)
        df = df.dropna()

    def removeTrader(self, trader, trader_index):
        trader.m_is_trader_removed = True
        self.m_genomes[trader_index].fitness = -9999

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

    def ne_decision_handler(self, trader, decision, current_data, trader_index, df_index):
        deal_size = trader.m_balance * 0.50
        if trader.m_balance < 0:
            self.removeTrader(trader, trader_index)
            return 0

        if len(trader.m_deals) > 0:
            if trader.m_deals[2] == True:
                if current_data.Low < trader.m_deals[4]:
                    self.removeTrader(trader, trader_index)
                    return 0
            else:
                if current_data.High > trader.m_deals[4]:
                    self.removeTrader(trader, trader_index)
                    return 0

        if trader.m_is_trader_removed == False:
            if decision[0] > decision[1]:
                if len(trader.m_deals) > 0 and trader.m_deals[2] == False:
                    self.m_genomes[trader_index].fitness += int(trader.closeDeal(current_data.Close, df_index))
                    trader.closeDeal(current_data.Close, df_index)
                    deal_size = trader.m_balance * 0.50
                    trader.addDeal(current_data.Close, deal_size, True, self.m_leverage, df_index)
                    return 1
                elif len(trader.m_deals) == 0:
                    trader.addDeal(current_data.Close, deal_size, True, self.m_leverage, df_index)
            elif decision[0] < decision[1]:
                if len(trader.m_deals) > 0 and trader.m_deals[2] == True:
                    trader.closeDeal(current_data.Close, df_index)
                    deal_size = trader.m_balance * 0.50
                    self.m_genomes[trader_index].fitness += int(trader.closeDeal(current_data.Close, df_index))
                    trader.addDeal(current_data.Close, deal_size, False, self.m_leverage, df_index)
                    return 1
                elif len(trader.m_deals) == 0:
                    trader.addDeal(current_data.Close, deal_size, False, self.m_leverage, df_index)

        if df_index % 1000 == 0 and trader.m_deals_counter <= 10:
            self.removeTrader(trader, trader_index)
            return 0

    def eval_genomes(self, genomes, config):
        self.m_traders.clear()
        self.m_genomes.clear()
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.m_traders.append(Trader(net))
            genome.fitness = 0
            self.m_genomes.append(genome)

        for index, trader in enumerate(self.m_traders):
            for i in range(len(self.m_train_df)):
                if i > 0:
                    if trader.m_is_trader_removed == False:
                        net_input_data = [self.m_train_df.EMA_100.iloc[i], self.m_train_df.Lag_1.iloc[i],
                                          self.m_train_df.Volume.iloc[i], self.m_train_df.Close.iloc[i]]
                        decision = trader.getDecision(net_input_data)
                        self.ne_decision_handler(trader, decision, self.m_train_df.iloc[i], index, i)

            if trader.m_is_trader_removed == False:
                trader.closeDeal(self.m_train_df.Close.iloc[-1], len(self.m_train_df) - 1)

        for index, trader in enumerate(self.m_traders):
            # self.m_genomes[index].fitness += int(trader.m_balance - 1000) ** 3
            if trader.m_balance > 0 and trader.m_is_trader_removed == False:
                if (trader.m_balance > 50):
                    if self.m_best_result < trader.m_balance:
                        self.m_best_result = trader.m_balance
                        trader.saveLogs('logs.csv')

                    print(trader.m_balance)
                    print(trader.m_deals_counter)
                    print(len(trader.m_balances))

                    pd.options.mode.chained_assignment = None
                    frame = []
                    frame = pd.DataFrame(trader.m_balances, index=list(range(0, len(trader.m_balances))))
                    frame['index'] = list(range(0, len(trader.m_balances)))
                    frame.columns = {'balance', 'index'}

                    plt.figure(figsize=(20, 7))
                    plt.plot(frame['balance'], alpha=0.7)
                    plt.show()

                    time.sleep(5)
                    user_decision = input("Name of AI: ")
                    with open(str(user_decision + '.pkl'), 'wb') as outp:
                        pickle.dump(trader.m_net, outp, pickle.HIGHEST_PROTOCOL)


                # print('********TEST********')
                # print('FRAMES')
                # for frame in frames:
                #     test_balance, test_deals_counter, test_is_trader_removed = trader.test_on_df(frame,
                #                                                                                  'Frame_logs.csv',
                #                                                                                  self.m_leverage)
                #     print('balance : ' + str(test_balance) + '\t deals_counter: ' + str(
                #         test_deals_counter) + '\t test_is_trader_removed: ' + str(test_is_trader_removed))
                #
                # btc_test_balance, btc_test_deals_counter, btc_test_is_trader_removed = trader.test_on_df(df,
                #                                                                                          'BTC_logs.csv',
                #                                                                                          self.m_leverage)
                # print('BTC balance : ' + str(btc_test_balance) + '\t deals_counter: ' + str(
                #     btc_test_deals_counter) + '\t btc_test_is_trader_removed: ' + str(btc_test_is_trader_removed))
                # ltc_test_balance, ltc_test_deals_counter, ltc_test_is_trader_removed = trader.test_on_df(LTC_df,
                #                                                                                          'LTC_logs.csv',
                #                                                                                          self.m_leverage)
                # print('LTC balance : ' + str(ltc_test_balance) + '\t deals_counter: ' + str(
                #     ltc_test_deals_counter) + '\t ltc_test_is_trader_removed: ' + str(ltc_test_is_trader_removed))
                # eth_test_balance, eth_test_deals_counter, eth_test_is_trader_removed = trader.test_on_df(ETH_df,
                #                                                                                          'ETH_logs.csv',
                #                                                                                          self.m_leverage)
                # print('ETH balance : ' + str(eth_test_balance) + '\t deals_counter: ' + str(
                #     eth_test_deals_counter) + '\t eth_test_is_trader_removed: ' + str(eth_test_is_trader_removed))
                # bnb_test_balance, bnb_test_deals_counter, bnb_test_is_trader_removed = trader.test_on_df(BNB_df,
                #                                                                                          'BNB_logs.csv',
                #                                                                                          self.m_leverage)
                # print('BNB balance : ' + str(bnb_test_balance) + '\t deals_counter: ' + str(
                #     bnb_test_deals_counter) + '\t bnb_test_is_trader_removed: ' + str(bnb_test_is_trader_removed))
                # print('********************')


                # if btc_test_balance > 1000 and ltc_test_balance > 1000 and eth_test_balance > 1000:
                #    time.sleep(60)

                # for i in range(len(self.m_deals_log)):
                #    frame.append()

                # time.sleep(60)

#df = get_binance_df(1644094800000, 1644181200000)
# df1 = get_binance_df()
# df2 = df.append(df1)
#print(df)
#print(df['Datetime'].iloc[999])
#print(df['Datetime'].iloc[1000])
#print(df['Datetime'].iloc[1001])
# print(df1)
# print(df2)
#frames = []

# config_path = 'config.txt'
# ne = NEAT_Trading(config_path, df, 30)
# ne.run(config_path, 99999)

# Access current Prices for your desired symbol

# def get_current_price(symbol):
#     ticker = yf.Ticker(symbol)
#     todays_data = ticker.history(period='1m')
#     return todays_data['Close'][0]
#
# print(get_current_price('BTC-USD'))

# with open('test.pkl', 'rb') as inp:
#     best_net = pickle.load(inp)
#
# LTC_df = yf.download('LTC-USD', interval='1m', period='7d')  # start='2021-12-01', end='2021-12-05')
# #LTC_df['EMA_100'] = LTC_df['Close'].ewm(span=100, adjust=False).mean()
# LTC_df['EMA_100'] = LTC_df['Close'].rolling(window=100).mean()
# LTC_df['Lag_1'] = LTC_df.Close - LTC_df.Close.shift(1)
# LTC_df = LTC_df.dropna()
#
#
# trader = Trader(best_net)
# balance, deals_counter, is_trader_removed = trader.test_on_df(df, "Saved_AI.csv", 30)
# print(balance)
# print(deals_counter)
# print(is_trader_removed)

trader = Trader(0)
trader.addDeal(1000, 25, True, 20, 1)
trader.closeDeal(1200, 5)
trader.saveLogs('new_logs.csv')
print("Deals_counter: " + str(trader.m_deals_counter))
print("Balances: " + str(trader.m_balances))

