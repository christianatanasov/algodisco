#This module is to keep all of the functions used in each different notebook, so they can be reused and easily imported
import warnings
import datetime
import itertools
import numpy as np
import scipy as sp
import cvxpy as cp
import pandas as pd
import yfinance as yf
import seaborn as sns
import ipywidgets as widgets
import matplotlib.pyplot as plt
from functools import reduce

interval_dict = {'1mo' : 'M', '1d' : 'D', '5d' : 'D'}

def drawdown(r):
    comp_return_ts = (1 + r).cumprod()
    comp_return_max = comp_return_ts.cummax()
    return (comp_return_ts - comp_return_max) / comp_return_max

portfolio_return = lambda w, r: np.dot(w, (1 + r).prod())

def portfolio_vol(weights, covmat): # A bit unconventional with the loop, but find it easier to follow than the dot product short version
    vol = sum(weights ** 2 * returns.std() ** 2) 
    for x,y in list(itertools.combinations(range(len(weights)), 2)):
        vol += 2 * weights[x] * weights[y] * covmat.iloc[x,y]
    return np.sqrt(vol)

def annualize_returns(r, intervals, periods):
    return np.exp(np.log1p(r).sum()) ** (periods / intervals) - 1

def annualize_vol(r, intervals):
    return r.std() * np.sqrt(intervals)

def d_flat(i, r): # Flat yield curve here
    return (1 + r) ** -i

def d(r): # For r being interest rates for each period
    return (1 + pd.Series(r)).prod()

def pv(l, r): #Present value
    if isinstance(l.index[0], datetime.date):
        dates = (l.index - datetime.date.today()).days / 360
    else:
        dates = l.index
    return d_flat(dates, r)

def fr(a, l, r): #Funding ratio for given assets a and liabilities l and a flat yield curve at r
    return a / pv(l,r)

def fr_wrap(a, r): #Wrapper function to show funding ratio in a widget
    print(fr(a, df, r))
    
def gbm(s_0=100, mu=0.07, sigma=0.15, freq=12, years=50, n_scenarios=100):
    dt = 1 / freq
    n_steps = int(years * freq)
    r = np.random.normal((1 + mu) ** dt, sigma * np.sqrt(dt), (n_steps, n_scenarios))
    prices = s_0 * pd.DataFrame(r).cumprod()
    return prices
    
    
def get_yf_ts(ticker, period, interval):
    df = yf.Ticker(ticker).history(period=period, interval=interval)[['Close']].rename(columns={'Close' : ticker})
    df.index = df.index.to_period(interval_dict[interval])
    return df

def get_mcap(ticker, usd=True):
    info = yf.Ticker(ticker).info
    mcap, ccy = info['marketCap'], info['currency']
    if usd and ccy != 'USD': # If not a US stock, convert market cap in USD using previous close FX
        fx = yf.Ticker('{}USD=X'.format(ccy)).info['previousClose']
        return mcap * fx
    return mcap

def get_usd_mcap(ticker):
    info = yf.Ticker(ticker).info
    mcap, ccy = info['marketCap'], info['currency']
    if ccy != 'USD': # If not a US stock, convert market cap in USD using previous close FX
        fx = yf.Ticker('{}USD=X'.format(ccy)).info['previousClose']
        return mcap * fx
    return mcap

def get_dividends(ticker, start_year, end_year=datetime.date.today().year):
    df = yf.Ticker(ticker).get_dividends().to_frame()
    df.index = df.index.to_period('D').to_timestamp()
    df['YEAR'] = df.index.year
    return df[str(start_year) : str(end_year)].rename(columns={'Dividends' : '{}_DIV'.format(ticker)})

def get_div_return(div_df, ticker, rf):
    if type(rf.index) == pd.core.indexes.period.PeriodIndex:
        rf.index = rf.index.to_timestamp()
    div_df['DAYS_TO_EOY'] = (div_df['YEAR'].apply(lambda c: datetime.date(c, 12, 31)) - div_df.index.date).dt.days
    div_df = div_df.merge(rf, left_index=True, right_index=True)
    div_df['{}_DIV_EOY'.format(ticker)] = np.exp(div_df['^IRX'] * div_df['DAYS_TO_EOY'] / 365) * div_df['{}_DIV'.format(ticker)]
    return div_df

futures_fp = lambda S, rf, b, div, T_frac : S * np.exp((rf - b) * T_frac) - div

def d1(S, K, r, dt, sigma):
    return (np.log(S/K) + (r + np.power(sigma, 2) * .5 )* dt) / (sigma * np.sqrt(dt))

def d2(S, K, r, dt, sigma):
    return d1(S, K, r, dt, sigma) - sigma * np.sqrt(dt)

def call(S, K, r, dt, sigma):
    return S * sp.stats.norm.cdf(d1(S, K, r, dt, sigma)) - K * np.exp(-r * dt) * sp.stats.norm.cdf(d2(S, K, r, dt, sigma))

def iv(C, S, K, r, dt):
    for sigma in np.linspace(0.01, 3, 500):
        if (C - call(S, K, r, dt, sigma)) < 1e-3:
            return sigma