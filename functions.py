
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import matplotlib.pyplot as plt
#import MetaTrader5 as mt5


#def historicos(path, login, password, server, start_date, end_date, save_name, symbol):
#    connection = mt5.initialize(path=path, login=login, password=password, server=server)
#    tuplas = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M12, start_date, end_date)
#    df = pd.DataFrame(tuplas)
#    df['time'] = pd.to_datetime(df['time'], unit='s')
#    df.to_excel("Historicos.xlsx")
#    return df


def moving_average(path, login, password, server, start_date, end_date, save_name, symbol):
    #df = historicos(path, login, password, server, start_date, end_date, save_name, symbol)
    df = pd.read_excel('Historicos.xlsx')
    df['moving_average_short'] = df['close'].rolling(window=20).mean()
    df['moving_average_long'] = df['close'].rolling(window=50).mean()

    plt.figure()
    plt.plot(df.close)
    plt.plot(df.moving_average_short)
    plt.plot(df.moving_average_long)
    #plt.show()
    return df


def boolinger_bands(path, login, password, server, start_date, end_date, save_name, symbol):
    #df = historicos(path, login, password, server, start_date, end_date, save_name, symbol)
    df = pd.read_excel('Historicos.xlsx')
    df['moving_average'] = df['close'].rolling(window=20).mean()
    df['standar_deviation'] = df['close'].rolling(window=20).std()
    df['boolinger_upper'] = df['moving_average'] + (df['standar_deviation'] * 2)
    df['boolinger_lower'] = df['moving_average'] - (df['standar_deviation'] * 2)

    plt.figure()
    plt.plot(df.close)
    plt.plot(df.moving_average)
    plt.plot(df.boolinger_upper)
    plt.plot(df.boolinger_lower)
    #plt.show()
    return df


def trading(path, login, password, server, start_date, end_date, save_name, symbol):
    capital = 100000
    ma_df = moving_average(path, login, password, server, start_date, end_date, save_name, symbol)
    bb_df = boolinger_bands(path, login, password, server, start_date, end_date, save_name, symbol)

    final = ma_df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']].copy()

    for t in range(1, len(ma_df)):
        # For Moving Average
        if (ma_df.loc[t-1, 'moving_average_short'] < ma_df.loc[t-1, 'moving_average_long'] and
                ma_df.loc[t, 'moving_average_short'] > ma_df.loc[t, 'moving_average_long']):
            final.loc[t, 'Operation'] = 'Comprar'
        elif (ma_df.loc[t-1, 'moving_average_short'] > ma_df.loc[t-1, 'moving_average_long'] and
                ma_df.loc[t, 'moving_average_short'] < ma_df.loc[t, 'moving_average_long']):
            final.loc[t, 'Operation'] = 'Vender'

        # For boolinger Bands
        if (bb_df.loc[t-1, 'close'] > bb_df.loc[t-1, 'boolinger_lower'] and
                bb_df.loc[t, 'close'] < bb_df.loc[t, 'boolinger_lower']):
            final.loc[t, 'Operation'] = 'Comprar'
        elif (bb_df.loc[t-1, 'close'] < bb_df.loc[t-1, 'boolinger_upper'] and
                bb_df.loc[t, 'close'] > bb_df.loc[t, 'boolinger_upper']):
            final.loc[t, 'Operation'] = 'Vender'

    earnings_df = final.dropna(subset=['Operation'])
    earnings_df.reset_index(drop=True, inplace=True)

    earnings_df['ganancia'] = earnings_df['close'] - earnings_df['close'].shift()

    num_acciones = 10000

    earnings_df['ganancia'] = earnings_df['ganancia'] * num_acciones
    earnings_df.loc[0, 'ganancia'] = capital
    earnings_df['Capital'] = earnings_df['ganancia'].cumsum()

    final['Operation'].fillna("-", inplace=True)
    final = final.merge(earnings_df[['time', 'Capital']], left_on="time", right_on="time", how="left")
    final['Capital'].fillna(method='ffill', inplace=True)
    final['Capital'].fillna(capital, inplace=True)

    final.to_excel('prueba.xlsx')
    return final
