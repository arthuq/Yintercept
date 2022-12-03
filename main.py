import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Lasso


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

STRAT, CASH = {}, {}

strat_names = ["VWAP", "TWAP", "REG"]
for strategy in strat_names :
    STRAT[strategy] = {'ticker':ticks, 'position':[0]*len(ticks) }
    CASH[strategy] = 10e6

### PLOT
# for tick,y in df.items():
#     plt.plot(y["last"])
# plt.show()

##
ALLOC_FREQ = 300        #frequency of reallocation
QUANTITY = 5            #quantity on trades

#To keep track. temporary
t_cash = []
x_cash = {}
for strategy in strat_names :
    x_cash[strategy]= []

#Strategies file
# from functions import *

## FUNCTIONS

def check_position(s, ticker):
    "Returns position from certain strategyt of certin ticker"
    return STRAT[s]["position"][STRAT[s]["ticker"].index(ticker)]

def market_order(t, tick, action, s):
    tmp = df[tick]
    global CASH
    global QUANTITY
    # print(CASH)

    if t not in tmp["date"].values :
        return None

    #get price of asset
    price = round(float( tmp.loc[tmp["date"] == t ]["last"] ), 4)
    actual_position = check_position(s, tick)
    # cash = CASH[s]

    if action == "long":
        qnt = int( min(CASH[s] // price, QUANTITY) )
        STRAT[s]["position"][STRAT[s]["ticker"].index(tick)] += qnt
        CASH[s] -= qnt*price
        # print("long", qnt, CASH//price )

    elif action == "short" :
        qnt = int( min(actual_position, QUANTITY) )
        STRAT[s]["position"][STRAT[s]["ticker"].index(tick)] -= qnt
        CASH[s] += qnt*price
        # print("short", qnt)

##VWAP STRAT
def vwap(df):
    q = df['volume'].values         #traded volume
    p = df['last']                  #traded price
    return df.assign(vwap=(p * q).cumsum() / q.cumsum())

def vwap_strat(df, ticker, ALLOC_FREQ):
    # global ALLOC_FREQ

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


## REGRESSION STRAT



def lasso_strat(x_train, x_test, y, p) :
    # lasso = Lasso(alpha = 1.0)

    lasso = Lasso(alpha=0.015, fit_intercept=False, tol=0.001,
          max_iter=10e5, positive=True)

    lasso.fit(x_train,y)

    y_pred = lasso.predict([x_test])
    y_pred = round(y_pred[0], 4)

    # if y_pred > p :



    return y_pred



## MAIN

for i,t in enumerate(time) :

    # Frequency of reallocation
    if i % ALLOC_FREQ != 0 or i ==0:
        continue
    print(f"{i}/{len(time)}")

    #index of time
    ind = pd.Index(time).get_loc(t)

    # Running strategy on each tick
    for tick in ticks :
        if t not in df[tick]["date"].values:
            continue

        #available data
        tmp_dat = df[tick].iloc[:ind]

        #VWAP ------------------------------------------------
        s1 = vwap_strat(tmp_dat, tick, ALLOC_FREQ)
        if s1 != "idle" :
            market_order(t, tick, s1, "VWAP")


        # REG -----------------------------------------------

        #Inputs for Lasso
        y = tmp_dat.iloc[:-1]["last"]
        x_train, x_test  = pd.DataFrame(), []
        for tick2 in ticks:
            if tick2==tick : continue
            tmp = df[tick2].iloc[:ind]["last"].values
            if len(tmp) != len(y)+1 : continue

            x_train[tick2] = tmp[:-1]
            x_test.append(tmp[-1])
        del tmp

        #strat calculation
        p = tmp_dat.iloc[-1]["last"]
        s3 = lasso_strat(x_train, x_test, y, p)
        print(p, s3)


        #__________________________________
        del tmp_dat

    t_cash.append(i)
    for strategy in strat_names :
        x_cash[strategy].append(CASH[strategy])


print(STRAT["VWAP"]["position"])
print(CASH)
##
plt.title("Cash evolution")
for strategy in strat_names :
    plt.plot(t_cash, x_cash[strategy], label=strategy)

plt.legend()
plt.show()