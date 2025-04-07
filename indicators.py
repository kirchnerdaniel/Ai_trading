import ta
import pandas as pd
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume


def calculate_rsi(close_prices, period=14):
    df = pd.DataFrame(close_prices,columns = ['Close'])
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], period).rsi()

    rsi_values = df['RSI'].dropna().tolist()
    return rsi_values

def calculate_adx(close_prices, high_prices, low_prices, period=14):
    df = pd.DataFrame({
        'High': high_prices,
        'Low' : low_prices,
        'Close': close_prices
    })

    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], period).adx()

    adx_values = df['ADX'].dropna().tolist()
    return adx_values

def calculate_atr(close_prices, high_prices, low_prices, period=14):
    df = pd.DataFrame({
        'High': high_prices,
        'Low' : low_prices,
        'Close': close_prices
    })

    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], period).average_true_range()
    
    atr_values = df['ATR'].dropna().tolist()
    return atr_values

def calculate_vwap(close_prices, high_prices, low_prices, volume, period=14):
    df = pd.DataFrame({
        'High': high_prices,
        'Low' : low_prices,
        'Close': close_prices,
        'Volume': volume
    })

    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        volume=df['Volume'],
        window=period
    ).volume_weighted_average_price()

    vwap_values = df['VWAP'].dropna().tolist()

    return vwap_values

def calculate_ma(close_prices, period=14):
    df = pd.DataFrame(close_prices,columns = ['Close'])
    df['MA'] = ta.trend.SMAIndicator(df['Close'], period).sma_indicator()

    ma_values = df['MA'].dropna().tolist()

    return ma_values

def calculate_average_volumen(volume, period=14):
    df = pd.DataFrame(volume,columns = ['Volume'])
    df['AV'] = ta.trend.SMAIndicator(df['Volume'], period).sma_indicator()

    av_values = df['AV'].dropna().tolist()

    return av_values

def calculate_ichimoku_cloud(high_prices, low_prices, period1=9, period2=26):
    
    df = pd.DataFrame({
        'High': high_prices,
        'Low': low_prices,
    })
    df['Tenkan-sen'] = (df['High'].rolling(window=period1).max() + df['Low'].rolling(window=period1).min()) / 2
    df['Kijun-sen'] = (df['High'].rolling(window=period2).max() + df['Low'].rolling(window=period2).min()) / 2
    
    ichimoku_components = {
        'Tenkan-sen': df['Tenkan-sen'].dropna().tolist(),
        'Kijun-sen': df['Kijun-sen'].dropna().tolist(),
    }
    
    return ichimoku_components

def calculate_stochastic(close_prices, high_prices, low_prices, period=14):
    df = pd.DataFrame({
        'High': high_prices,
        'Low' : low_prices,
        'Close': close_prices
    })
    df['Stochastic'] = ta.momentum.StochasticOscillator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'],
        window=period).stoch()

    stochastic_values = df['Stochastic'].dropna().tolist()
    return stochastic_values

def calculate_williams_r(close_prices, high_prices, low_prices, period=14):

    df = pd.DataFrame({
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    })
    
    williams_r_indicator = ta.momentum.WilliamsRIndicator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'],
        lbp=period  
    )
    df['Williams %R'] = williams_r_indicator.williams_r()
    
    williams_r_values = df['Williams %R'].dropna().tolist()
    return williams_r_values

def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    
    min_length = max(fast_period, slow_period, signal_period)
    if len(close_prices) < min_length:
        raise ValueError(f"close_prices must be at least {min_length} elements long.")
    
    df = pd.DataFrame({'Close': close_prices})
    
    macd_indicator = ta.trend.MACD(
        close=df['Close'], 
        window_slow=slow_period, 
        window_fast=fast_period, 
        window_sign=signal_period
    )
    
    df['MACD'] = macd_indicator.macd() 
    df['Signal'] = macd_indicator.macd_signal()
    df['Histogram'] = macd_indicator.macd_diff()
    
    macd_components = {
        'macd_values': df['MACD'].dropna().tolist(),
        'signal_values': df['Signal'].dropna().tolist(),
        'histogram_values': df['Histogram'].dropna().tolist(),
    }
    
    return macd_components


def calculate_cmf(close_prices, high_prices, low_prices, volumes, period=20):
   
    df = pd.DataFrame({
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })

    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        volume=df['Volume'], 
        window=period
    ).chaikin_money_flow()

    cmf_values = df['CMF'].dropna().tolist()
    return cmf_values



def calculate_cci(close_prices, high_prices, low_prices, period=20):
    df = pd.DataFrame({
        'High': high_prices,
        'Low' : low_prices,
        'Close': close_prices
    })

    df['CCI'] = ta.trend.cci(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=period, 
        constant=0.015
    )

    cci_values = df['CCI'].dropna().tolist()
    return cci_values


def calculate_oscp(close_prices):
    df = pd.DataFrame(close_prices,columns = ['Close'])
    df['Long MA'] = ta.trend.SMAIndicator(df['Close'], 10).sma_indicator()
    df['Short MA'] = ta.trend.SMAIndicator(df['Close'], 5).sma_indicator()

    long_ma_values = df['Long MA'].dropna().tolist()
    short_ma_values = df['Short MA'].dropna().tolist()

    min_length = min(len(long_ma_values), len(short_ma_values))
    long_ma_values = long_ma_values[-min_length:]
    short_ma_values = short_ma_values[-min_length:]

    OSCP = []
    for i in range(min_length):
        OSCP.append((short_ma_values[i] - long_ma_values[i]) / short_ma_values[i])

    return OSCP


def calculate_ad(close_prices, high_prices, low_prices, volumes):
    df = pd.DataFrame({
        'High': high_prices,
        'Low' : low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    df['A/D'] = ta.volume.acc_dist_index(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        volume=df['Volume']
    )

    ad_values = df['A/D'].tolist()
    return ad_values

def calculate_roc(close_prices, period=12):
    df = pd.DataFrame({'Close': close_prices})
    
    df['ROC'] = ta.momentum.roc(close=df['Close'], window=period)
    
    roc_values = df['ROC'].dropna().tolist()
    return roc_values

def calculate_momentum(close_prices):
    
    momentum_values = []
    for i in range(4, len(close_prices)):
        momentum_values.append((close_prices[i] - close_prices[i-4])/close_prices[i-4])

    return momentum_values

def calculate_momentum_old(close_prices):
    
    momentum_values = []
    for i in range(4, len(close_prices)):
        momentum_values.append(close_prices[i] - close_prices[i-4])

    return momentum_values