import os
import time as ttime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
##
start_time = ttime.time()

##IMPORTING DATA
os.chdir(r"C:\Users\Arthur\Documents\GitHub\y-intercept")
df0 = pd.read_csv(r'data.csv')
ticks = list(df0["ticker"].unique())

df = {}
for i,tick in enumerate(ticks):
    df[tick] = df0.loc[df0['ticker'] == tick]

time = df[ticks[0]]["date"]
del df0

## SOME PARAMETERS

START_CASH = 10e6       #Initial invesment
ALLOC_FREQ = 30         #frequency of reallocation
QUANTITY = 5            #quantity on buy/sell

#Names of the differents strategies
strat_names = ["VWAP","LASSO", "LASSO_on_VWAP"]

#Initializing portfolio for each strategy
STRAT, CASH= {}, {}
for strategy in strat_names :
    STRAT[strategy] = {'ticker':ticks, 'position':[0]*len(ticks) }
    CASH[strategy] = START_CASH

### INITIAL PLOT
# for tick,y in df.items():
#     plt.plot(y["last"])
# plt.show()

## PORTFOLIO CONTINUOUS, KEEPING TRACK OF CHANGES
t_track, x_cash, x_assets = [0], {}, {}
for strategy in strat_names :
    x_cash[strategy], x_assets[strategy] = [START_CASH], [0]

## FUNCTIONS
def check_position(s, ticker):
    "Returns position from certain strategyt of certin ticker"
    return STRAT[s]["position"][STRAT[s]["ticker"].index(ticker)]

def market_order(t, tick, action, s):
    global CASH, QUANTITY
    tmp = df[tick]

    if t not in tmp["date"].values :
        return None

    #get price of asset
    price = round(float( tmp.loc[tmp["date"] == t ]["last"] ), 4)
    actual_position = check_position(s, tick)

    if action == "long":
        qnt = int( min(CASH[s] // price, QUANTITY) )
        STRAT[s]["position"][STRAT[s]["ticker"].index(tick)] += qnt
        CASH[s] -= qnt*price

    elif action == "short" :
        qnt = int( min(actual_position, QUANTITY) )
        STRAT[s]["position"][STRAT[s]["ticker"].index(tick)] -= qnt
        CASH[s] += qnt*price

##STRAT 1 : VWAP STRAT
def vwap(df):
    q = df['volume'].values         #traded volume
    p = df['last']                  #traded price
    return df.assign(vwap=(p * q).cumsum() / q.cumsum())

def vwap_single(spot, vol):
    return (spot * vol).cumsum() / vol.cumsum()

def vwap_strat(df, ticker):
    global ALLOC_FREQ

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

## STRAT 2 : TWAP STRAT (NOT FINISHED)
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


## STRAT 3 : REGRESSION STRAT
def lasso_strat(x, y, p, ticker, alpha=0.8, tol=1e-2, max_ite=10e4) :
    # start_time = ttime.time()

    #Lasso parameters
    lasso = Lasso(alpha=alpha, tol=tol, max_iter=max_ite)

    #Formatting data
    x_train = x.iloc[:-1]
    x_test = x.iloc[-1].values

    #Fitting the model
    lasso.fit(x_train,y)
    y_pred = round(lasso.predict([x_test])[0] , 4)
    # print(f" {round(ttime.time() - start_time,4)}")

    #Trend
    regressor = LinearRegression()
    xt = np.array(range(len(y)))
    regressor.fit(xt.reshape(-1, 1),np.array(y))
    slope = regressor.coef_[0]
    # print(slope)

    #Current position on ticker
    position = check_position("LASSO", ticker)

    #Undervalued asset
    should_buy = y_pred > p and slope > 0.05

    #Overvallued asset
    should_sell = y_pred < p and slope < -0.05

    #Simple strategy : buy if signal to buy and no asset detained
    if position == 0 and should_buy :
        return "long"

    #Sell if asset is in portfolio
    elif position > 0 and should_sell :
        return "short"

    return "idle"

## MAIN RUN

print("done importing.")
for ind,t in enumerate(time) :
    i=ind

    # Frequency of reallocation
    if ind % ALLOC_FREQ != 0 or ind ==0:
        continue
    # print(f"{i}/{len(time)}")
    print(f"{ind}/{len(time)}")

    #Construction spot df price, for LASSO
    tmp_spot, tmp_vwap = pd.DataFrame(), pd.DataFrame()
    lookback = max(30, ALLOC_FREQ)
    for tick in ticks:
        start = max(ind-lookback, 0)
        if t not in df[tick]["date"].values or ind>len(df[tick]):
            continue
        tmp_spot[tick] = df[tick]["last"].iloc[start:ind].values
        tmp_vwap[tick] = vwap_single(tmp_spot[tick], df[tick]["volume"].iloc[start:ind].values)

    # print(tmp_vwap)
    tmp_assets = { s:[0] for s in strat_names }

    # Running strategy on each tick
    for tick in ticks :
        if t not in df[tick]["date"].values or ind>len(df[tick]) :
            continue

        #available data
        tmp_dat = df[tick].iloc[:ind]

        #STRAT 1 : VWAP ------------------------------------------------
        s1 = vwap_strat(tmp_dat, tick)
        if s1 != "idle" :
            market_order(t, tick, s1, "VWAP")

        # STRAT 3 : REG -----------------------------------------------
        y = tmp_spot[tick].iloc[:-1]
        x = tmp_spot.drop(tick, axis=1) #.iloc[:-1]
        p = tmp_spot[tick].iloc[-1]

        s3 = lasso_strat(x, y, p, tick, 0.8, 1e-2, 10e4)
        if s3 != "idle" :
            market_order(t, tick, s3, "LASSO")
        # del tmp_dat

        # STRAT 4 : REG ON VWAP -----------------------------------------
        x = tmp_vwap.drop(tick, axis=1) #.iloc[:-1]
        s4 = lasso_strat(x, y, p, tick, 1, 1e-1, 10e2)
        if s4 != "idle" :
            market_order(t, tick, s4, "LASSO_on_VWAP")

        # SRTAT 5 : ALL SAME SIGNALS
        # if s1==s3==s4 and s1 != "idle":
            # market_order(t, tick, s1, "ALL")

        #Updating portfolio value after reallocation
        for s in strat_names:
            asset_qnt = STRAT[s]["position"][STRAT[s]["ticker"].index(tick)]
            tmp_assets[s] += asset_qnt*p
    del tmp_dat

    #Keeping track of allocations
    t_track.append(ind)
    for strategy in strat_names :
        x_cash[strategy].append(CASH[strategy])
        x_assets[strategy].append(tmp_assets[strategy])
    del tmp_assets

## PORTFOLIO REAL VALUE UPDATE

total_value={ s:[] for s in strat_names}
for s in strat_names:
    for c,a in zip(x_cash[s], x_assets[s]):
        total_value[s].append(float(c)+float(a) )

## RESULTS
def sign(x):
    if x>=0:
        return "+"
    return "-"

print(*[ f"PNL {i} : {round(j[-1], 2)} ({sign(j[-1]-START_CASH)}{round(abs(j[-1]-START_CASH), 2)})"  for i,j in total_value.items()], sep='\n')

print(f"Total time : {round(ttime.time() - start_time,4)}s")

## PLOT
plt.title("Portfolio evolution")
for strategy in strat_names :
    # plt.plot(t_track, x_cash[strategy], label="CASH_"+strategy)
    # plt.plot(t_track, x_assets[strategy], label="ASSETS_"+strategy, color="green" )
    plt.plot(t_track, total_value[strategy], label="TOT_"+strategy, linewidth=5 )
plt.grid()
plt.legend()
plt.show()