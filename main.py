import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##

os.chdir(r"C:\Users\Arthur\Documents\GitHub\y-intercept")

##IMPORTING DATA
df0 = pd.read_csv(r'data.csv')
ticks = list(df0["ticker"].unique())

df = {}
for tick in ticks:
    df[tick] = df0.loc[df0['ticker'] == tick]

time = df[ticks[0]]["date"]
del df0

##

strat = {}
strat["VWAP"] = pd.DataFrame(columns=['ticker', "position"])
strat["TWAP"] = pd.DataFrame(columns=['ticker', "position"])
strat["REG"] = pd.DataFrame(columns=['ticker', "position"])

### PLOT
# for tick,y in df.items():
#     plt.plot(y["last"])
# plt.show()

##
ALLOC_FREQ = 30 #requency of reallocation
# from VWAP import *

def check_position(strat, ticker):
    strat["TWAP"]


##VWAP STRAT
def vwap(df):
    q = df['volume'].values       #traded volume
    p = df['last']          #traded price
    return df.assign(vwap=(p * q).cumsum() / q.cumsum())

def vwap_strat(df):
    l = len(df)
    vwap_df = vwap(df)

    #last trading day
    # tmp = vwap_df.iloc[-1]
    # tmp_past = vwap_df.iloc[-15]
    # diff =  tmp["last"] - tmp.vwap
    # ratio = 100*diff/tmp["last"]

    #signal to buy
    should_buy = vwap_df['last'][-1] > vwap_df["vwap"][-1] and vwap_df["vwap"][-1] > vwap_df["vwap"][-2]

    #signal to sell
    should_sell = vwap_df['last'][-1] < vwap_df["vwap"][-1] and vwap_df["vwap"][-1] < vwap_df["vwap"][-2]

    position = check_position(ticker)


    plt.title("VWAP vs price")
    plt.plot(vwap_df["vwap"])
    plt.plot(vwap_df["last"])
    plt.grid()
    # plt.show()


## TWAP STRAT
def twap(df, period):
    # tp = (df['low'] + df['close'] + df['high']).divide(3)
    p = df['last']
    return df.assign(twap=(p.rolling(period).sum().divide(period)))


def twap_strat(df, period = 15):
    twap_df = twap(df, period)
    # ...
    # ...

    plt.title("TWAP vs price")
    plt.plot(twap_df["twap"])
    plt.plot(twap_df["last"])
    plt.grid()
    # plt.show()

    return twap_df

# twap_strat(tmp2)




## MAIN

for i,t in enumerate(time) :

    # Frequency of reallocation
    if i % ALLOC_FREQ != 0 or i ==0:
        continue

    # Running strategy on each tick
    for tick in ticks[:3] :
        print(tick)
        ind = pd.Index(time).get_loc(t) #index of t
        tmp_dat = df[tick].iloc[:ind] #available data
        # print(tmp_dat["volume"])

        #benchmark
        s1 = vwap_strat(tmp_dat)

        s1_tmp = {'date': t, 'quantity': 1 , 'action': s1 }

        print(s1_tmp)
        # .append(s1_tmp, ignore_index = True)


        portfolio_s1[tick].append(s1_tmp, ignore_index=True)

        print(portfolio_s1[tick])

        # update
        # df_port = pd.DataFrame({'date' : [time], 'quantity':[12] })
        # portfolio[tick].append(df_port, ignore_index = True)


##
