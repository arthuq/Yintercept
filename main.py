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
for i,tick in enumerate(ticks):
    df[tick] = df0.loc[df0['ticker'] == tick]

time = df[ticks[0]]["date"]
del df0
##

strat = {}
strat["VWAP"] = {'ticker':ticks, 'position':[0]*len(ticks) }
strat["TWAP"] = {'ticker':ticks, 'position':[0]*len(ticks) }
strat["REG"] = {'ticker':ticks, 'position':[0]*len(ticks) }


### PLOT
# for tick,y in df.items():
#     plt.plot(y["last"])
# plt.show()

##
ALLOC_FREQ = 200 #requency of reallocation
CASH = 10e3
QUANTITY = 5

t_cash, x_cash = [], []

##
# from VWAP import *

def check_position(s, ticker):
    "Returns position from certain strategyt of certin ticker"
    return strat[s]["position"][strat[s]["ticker"].index(ticker)]


def market_order(t, tick, action, s):
    tmp = df[tick]
    global CASH

    if t not in tmp["date"].values :
        return None

    #get price of asset
    price = round(float( tmp.loc[tmp["date"] == t ]["last"] ), 4)
    actual_position = check_position(s, tick)

    if action == "long":
        qnt = int( min(CASH//price, QUANTITY) )
        strat[s]["position"][strat[s]["ticker"].index(tick)] += qnt
        CASH -= qnt*price

    elif action == "short" :
        qnt = int( min(actual_position, QUANTITY) )
        strat[s]["position"][strat[s]["ticker"].index(tick)] -= qnt
        CASH += qnt*price

##VWAP STRAT
def vwap(df):
    q = df['volume'].values         #traded volume
    p = df['last']                  #traded price
    return df.assign(vwap=(p * q).cumsum() / q.cumsum())


def vwap_strat(df, ticker):

    if len(df) <= ALLOC_FREQ :
        return "idle"

    vwap_df = vwap(df)
    position = check_position("VWAP", ticker)

    #signal to buy
    should_buy = vwap_df.iloc[-1]['last'] > vwap_df.iloc[-1]["vwap"] and vwap_df.iloc[-1]["vwap"] > vwap_df.iloc[-ALLOC_FREQ]["vwap"]

    #signal to sell
    should_sell = vwap_df.iloc[-1]['last'] < vwap_df.iloc[-1]["vwap"] and vwap_df.iloc[-1]["vwap"] < vwap_df.iloc[-ALLOC_FREQ]["vwap"]

    #Simple strategy : buy if signal to buy and no asset detained
    if position == 0 and should_buy :
        return "long"

    #Sell if asset is in portfolio
    elif position > 0 and should_sell :
        return "short"

    return "idle"


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
    for tick in ticks :

        if t not in df[tick]["date"].values:
            continue

        ind = pd.Index(time).get_loc(t)     #index of t
        tmp_dat = df[tick].iloc[:ind]       #available data

        #VWAP ------------------------------------------------
        s1 = vwap_strat(tmp_dat, tick)
        if s1 != "idle" :
            market_order(t, tick, s1, "VWAP")

    t_cash.append(i)
    x_cash.append(CASH)

            # print(s1, d)


print(strat["VWAP"]["position"])
print(CASH)
##
plt.scatter(t_cash, x_cash)
plt.show()