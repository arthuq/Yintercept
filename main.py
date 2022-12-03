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
QUANTITY = 1

# from VWAP import *

def check_position(s, ticker):
    "Returns position from certain strategyt of certin ticker"
    return strat[s]["position"][strat[s]["ticker"].index(ticker)]

def market_order(t, tick, action, s):
    tmp = df[tick]

    if t not in tmp["date"].values :
        return None

    price = round(float( tmp.loc[tmp["date"] == t ]["last"] ), 4)
    # print(t, tick, price)

    strat[s]["position"][strat[s]["ticker"].index(tick)] += QUANTITY

    # return price



##VWAP STRAT
def vwap(df):
    q = df['volume'].values         #traded volume
    p = df['last']                  #traded price
    return df.assign(vwap=(p * q).cumsum() / q.cumsum())



def vwap_strat(df, ticker):
    vwap_df = vwap(df)
    position = check_position("VWAP", ticker)

    #signal to buy
    should_buy = vwap_df.iloc[-1]['last'] > vwap_df.iloc[-1]["vwap"] and vwap_df.iloc[-1]["vwap"] > vwap_df.iloc[-ALLOC_FREQ]["vwap"]

    #signal to sell
    should_sell = vwap_df.iloc[-1]['last'] < vwap_df.iloc[-1]["vwap"] and vwap_df.iloc[-1]["vwap"] < vwap_df.iloc[-ALLOC_FREQ]["vwap"]

    if position == 0 and should_buy :
        return "long"

    elif position > 0 and should_sell :
        return "short"

    return "idle"

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
    for tick in ticks :


        if t not in df[tick]["date"].values:
            continue


        # print(tick)
        ind = pd.Index(time).get_loc(t) #index of t
        tmp_dat = df[tick].iloc[:ind] #available data
        # print(i, tick, "length", len(tmp_dat))
        # print(tmp_dat["volume"])

        #benchmark
        s1 = vwap_strat(tmp_dat, tick)

        if s1 != "idle" :
            market_order(t, tick, s1, "VWAP")
            print(s1, d)


        # s1_tmp = {'date': t, 'quantity': 1 , 'action': s1 }

        # print(s1_tmp)
        # .append(s1_tmp, ignore_index = True)


        # portfolio_s1[tick].append(s1_tmp, ignore_index=True)

        # print(portfolio_s1[tick])

        # update
        # df_port = pd.DataFrame({'date' : [time], 'quantity':[12] })
        # portfolio[tick].append(df_port, ignore_index = True)


##
