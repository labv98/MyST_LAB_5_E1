
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
import numpy as np
import datetime
from datetime import date
import time
import plotly.graph_objects as go
import plotly.express as px
import pyswarms as ps
# import MetaTrader5 as mt5


def historicos(path, login, password, server, start_date_backtest, end_date_backtest,
               start_date_prueba, end_date_prueba, symbol):
    connection = mt5.initialize(path=path, login=login, password=password, server=server)
    # Download backtest data
    df_backtest = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_date_backtest, end_date_backtest))
    df_backtest['time'] = pd.to_datetime(df_backtest['time'], unit='s')
    df_backtest.to_excel("Historicos_backtest.xlsx")
    # Download prueba data
    df_prueba = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_date_prueba, end_date_prueba))
    df_prueba['time'] = pd.to_datetime(df_prueba['time'], unit='s')
    df_prueba.to_excel("Historicos_prueba.xlsx")
    return df_backtest, df_prueba

def cost_function(variable):
    num_acciones = variable.item(0)
    pp_ma = variable.item(1)
    pg_ma = pp_ma
    w_bb = int(variable.item(2))
    sd_bb = variable.item(3)
    print(num_acciones, pp_ma, pg_ma, w_bb, sd_bb)

    df = pd.read_excel('Historicos_backtest.xlsx', index_col=0)
    ma_df = df.copy()
    bb_df = df.copy()
    ### MOVING AVERAGE
    ma_df['moving_average_short'] = ma_df['close'].rolling(window=20).mean()
    ma_df['moving_average_long'] = ma_df['close'].rolling(window=50).mean()
    # Obtener las columnas necesarias
    final_ma = ma_df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']].copy()
    final_bb = bb_df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']].copy()
    final_bb['Operation'] = np.nan
    ## BOLLINGER BANDS
    bb_df['moving_average'] = bb_df['close'].rolling(window=w_bb).mean()
    bb_df['standar_deviation'] = bb_df['close'].rolling(window=w_bb).std()
    bb_df['boolinger_upper'] = bb_df['moving_average'] + (bb_df['standar_deviation'] * sd_bb)
    bb_df['boolinger_lower'] = bb_df['moving_average'] - (bb_df['standar_deviation'] * sd_bb)

    ### Constantes
    Capital = 100000
    hoy = pd.Timestamp(datetime.datetime.today())
    save_time_venta = ma_df['time'][0] - datetime.timedelta(hours=2)
    save_time_compra = ma_df['time'][0] - datetime.timedelta(hours=2)

    for t in range(1, len(ma_df)):
        # For Moving Average
        ## Compras
        if (ma_df.loc[t - 1, 'moving_average_short'] < ma_df.loc[t - 1, 'moving_average_long'] and
                ma_df.loc[t, 'moving_average_short'] > ma_df.loc[t, 'moving_average_long'] and
                ma_df.loc[t, 'close'] * num_acciones <= Capital):
            final_ma.loc[t, 'Operation'] = 'Comprar'
            final_ma.loc[t, 'exposure'] = ma_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_ma.loc[t, 'Stoploss'] = max(ma_df.loc[t, 'close'] - ((Capital * 0.1) / num_acciones),
                                              ma_df.loc[t, 'close'] * (1 - pp_ma))
            final_ma.loc[t, 'Takeprofit'] = ma_df.loc[t, 'close'] * (1 + pg_ma)
            # Actualización Capital
            Capital = Capital - (ma_df.loc[t, 'close'] * num_acciones)
            # Cerrando posición
            ## Precio Stoploss
            close_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] <= final_ma.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_ma.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] >= final_ma.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_ma.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] <= final_ma.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_ma.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] >= final_ma.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_ma.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_ma.loc[t, 'CloseOp'] = np.where(final_ma.loc[t, 'Date_SL'] > final_ma.loc[t, 'Date_TP'],
                                                  final_ma.loc[t, 'TP'],
                                                  final_ma.loc[t, 'SL'])
            # Profit
            final_ma.loc[t, 'Profit'] = (final_ma.loc[t, 'CloseOp'] - final_ma.loc[t, 'close']) * num_acciones if \
                final_ma.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_ma.loc[t, 'CloseOp'] * num_acciones)
        ## Ventas
        elif (ma_df.loc[t - 1, 'moving_average_short'] > ma_df.loc[t - 1, 'moving_average_long'] and
              ma_df.loc[t, 'moving_average_short'] < ma_df.loc[t, 'moving_average_long'] and
              ma_df.loc[t, 'close'] * num_acciones <= Capital):
            final_ma.loc[t, 'Operation'] = 'Vender'
            final_ma.loc[t, 'exposure'] = ma_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_ma.loc[t, 'Stoploss'] = min(ma_df.loc[t, 'close'] + ((Capital * 0.1) / num_acciones),
                                              ma_df.loc[t, 'close'] * (1 + pp_ma))
            final_ma.loc[t, 'Takeprofit'] = ma_df.loc[t, 'close'] * (1 - pg_ma)
            # Actualización Capital
            Capital = Capital - ma_df.loc[t, 'close'] * num_acciones
            # Cerrando posición
            ## Precio Stoploss
            close_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] >= final_ma.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_ma.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] <= final_ma.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_ma.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] >= final_ma.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_ma.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] <= final_ma.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_ma.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_ma.loc[t, 'CloseOp'] = np.where(final_ma.loc[t, 'Date_SL'] > final_ma.loc[t, 'Date_TP'],
                                                  final_ma.loc[t, 'TP'],
                                                  final_ma.loc[t, 'SL'])
            # Profit
            final_ma.loc[t, 'Profit'] = (final_ma.loc[t, 'close'] - final_ma.loc[t, 'CloseOp']) * num_acciones if \
                final_ma.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_ma.loc[t, 'CloseOp'] * num_acciones)

        # For Boolinger Bands
        ## Compras
        if (bb_df.loc[t - 1, 'close'] > bb_df.loc[t - 1, 'boolinger_lower'] and
                bb_df.loc[t, 'close'] < bb_df.loc[t, 'boolinger_lower'] and
                bb_df.loc[t, 'close'] * num_acciones <= Capital and
                bb_df.loc[t, 'time'] - save_time_compra > datetime.timedelta(hours=2)):

            save_time_compra = bb_df.loc[t, 'time']

            final_bb.loc[t, 'Operation'] = 'Comprar'

            final_bb.loc[t, 'exposure'] = bb_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_bb.loc[t, 'Stoploss'] = max(bb_df.loc[t, 'close'] - ((Capital * 0.1) / num_acciones),
                                              bb_df.loc[t, 'close'] - (
                                                          bb_df.loc[t, 'moving_average'] - bb_df.loc[t, 'close']))
            final_bb.loc[t, 'Takeprofit'] = bb_df.loc[t, 'moving_average']
            # Actualización Capital
            Capital = Capital - bb_df.loc[t, 'close'] * num_acciones
            # Cerrando posición
            ## Precio Stoploss
            close_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] <= final_bb.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_bb.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] >= final_bb.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_bb.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] <= final_bb.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_bb.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] >= final_bb.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_bb.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_bb.loc[t, 'CloseOp'] = np.where(final_bb.loc[t, 'Date_SL'] > final_bb.loc[t, 'Date_TP'],
                                                  final_bb.loc[t, 'TP'],
                                                  final_bb.loc[t, 'SL'])
            # Profit
            final_bb.loc[t, 'Profit'] = (final_bb.loc[t, 'CloseOp'] - final_bb.loc[t, 'close']) * num_acciones if \
                final_bb.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_bb.loc[t, 'CloseOp'] * num_acciones)
        ## Ventas
        elif (bb_df.loc[t - 1, 'close'] < bb_df.loc[t - 1, 'boolinger_upper'] and
              bb_df.loc[t, 'close'] > bb_df.loc[t, 'boolinger_upper'] and
              bb_df.loc[t, 'close'] * num_acciones <= Capital and
              bb_df.loc[t, 'time'] - save_time_venta > datetime.timedelta(hours=2)):

            save_time_venta = bb_df.loc[t, 'time']

            final_bb.loc[t, 'Operation'] = 'Vender'
            final_bb.loc[t, 'exposure'] = bb_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_bb.loc[t, 'Stoploss'] = min(bb_df.loc[t, 'close'] + ((Capital * 0.1) / num_acciones),
                                              bb_df.loc[t, 'close'] + (
                                                          bb_df.loc[t, 'close'] - bb_df.loc[t, 'moving_average']))
            final_bb.loc[t, 'Takeprofit'] = bb_df.loc[t, 'moving_average']
            # Actualización Capital
            Capital = Capital - bb_df.loc[t, 'close'] * num_acciones
            # Cerrando posición
            ## Precio Stoploss
            close_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] >= final_bb.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_bb.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] <= final_bb.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_bb.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] >= final_bb.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_bb.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] <= final_bb.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_bb.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_bb.loc[t, 'CloseOp'] = np.where(final_bb.loc[t, 'Date_SL'] > final_bb.loc[t, 'Date_TP'],
                                                  final_bb.loc[t, 'TP'],
                                                  final_bb.loc[t, 'SL'])
            # Profit
            final_bb.loc[t, 'Profit'] = (final_bb.loc[t, 'close'] - final_bb.loc[t, 'CloseOp']) * num_acciones if \
                final_bb.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_bb.loc[t, 'CloseOp'] * num_acciones)

    # Earnings Moving Average
    earnings_df_ma = final_ma.dropna(subset=['Operation'])
    earnings_df_ma.reset_index(drop=True, inplace=True)
    # Earnings Boolinger Bands
    earnings_df_bb = final_bb.dropna(subset=['Operation'])
    earnings_df_bb.reset_index(drop=True, inplace=True)
    earnings_df_bb.rename(columns={'Operation': 'Operation_bb',
                                   'exposure': 'exposure_bb',
                                   'Stoploss': 'Stoploss_bb',
                                   'Takeprofit': 'Takeprofit_bb',
                                   'SL': 'SL_bb',
                                   'TP': 'TP_bb',
                                   'Date_SL': 'Date_SL_bb',
                                   'Date_TP': 'Date_TP_bb',
                                   'CloseOp': 'CloseOp_bb',
                                   'Profit': 'Profit_bb'}, inplace=True)
    # Merging Earnings
    df_ma_bb = df.merge(earnings_df_ma[['time', 'Operation', 'exposure',
                                        'Stoploss', 'Takeprofit', 'SL',
                                        'TP', 'Date_SL', 'Date_TP',
                                        'CloseOp', 'Profit']],
                        left_on='time', right_on='time', how='outer').merge(earnings_df_bb[['time',
                                                                                            'Operation_bb',
                                                                                            'exposure_bb',
                                                                                            'Stoploss_bb',
                                                                                            'Takeprofit_bb', 'SL_bb',
                                                                                            'TP_bb', 'Date_SL_bb',
                                                                                            'Date_TP_bb',
                                                                                            'CloseOp_bb', 'Profit_bb']],
                                                                            left_on='time', right_on='time',
                                                                            how='outer')
    df_ma_bb.fillna(0, inplace=True)
    df_ma_bb.sort_values(by='time', inplace=True, ignore_index=True)
    df_ma_bb = df_ma_bb[(df_ma_bb['Operation'] != 0) | (df_ma_bb['Operation_bb'] != 0)].reset_index(drop=True)
    df_ma_bb["Operation"] = df_ma_bb["Operation"].astype("str")
    df_ma_bb["Operation_bb"] = df_ma_bb["Operation_bb"].astype("str")
    df_ma_bb["Duplicates"] = np.where((((df_ma_bb['Operation'] == "Comprar") & (df_ma_bb['Operation_bb'] == "Vender")) | \
                                       ((df_ma_bb['Operation'] == "Vender") & (df_ma_bb['Operation_bb'] == "Comprar"))),
                                      1, 0)
    df_ma_bb = df_ma_bb[df_ma_bb['Duplicates'] == 0].copy()
    # Getting Final columns
    list_columns = ['exposure', 'Stoploss', 'Takeprofit', 'SL',
                    'TP', 'CloseOp', 'Profit']
    for i in list_columns:
        df_ma_bb[i + '_final'] = df_ma_bb[i] + df_ma_bb[i + '_bb']

    df_ma_bb['Date_SL_final'] = np.where(df_ma_bb['Date_SL'] == 0, df_ma_bb['Date_SL_bb'], df_ma_bb['Date_SL'])
    df_ma_bb['Date_TP_final'] = np.where(df_ma_bb['Date_TP'] == 0, df_ma_bb['Date_TP_bb'], df_ma_bb['Date_TP'])
    # Profit acum
    df_ma_bb['Profit_final_acum'] = df_ma_bb['Profit_final'].cumsum()
    # Sharpe Ratio
    ret = df_ma_bb['Profit_final_acum'].pct_change().dropna()
    mean_ret = np.mean(ret)
    rf = 0.05
    desvest = np.std(ret)
    sharpe_rat = (mean_ret - rf) / desvest
    print(sharpe_rat)
    return sharpe_rat * -1


def function(num_acciones, pp_ma, w_bb, sd_bb, data):

    pg_ma = pp_ma
    w_bb = int(w_bb)
    print(num_acciones, pp_ma, pg_ma, w_bb, sd_bb)

    df = pd.read_excel('Historicos_' + data + '.xlsx', index_col=0)
    ma_df = df.copy()
    bb_df = df.copy()
    ### MOVING AVERAGE
    ma_df['moving_average_short'] = ma_df['close'].rolling(window=20).mean()
    ma_df['moving_average_long'] = ma_df['close'].rolling(window=50).mean()
    # Obtener las columnas necesarias
    final_ma = ma_df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']].copy()
    final_bb = bb_df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']].copy()
    final_bb['Operation'] = np.nan
    ## BOLLINGER BANDS
    bb_df['moving_average'] = bb_df['close'].rolling(window=w_bb).mean()
    bb_df['standar_deviation'] = bb_df['close'].rolling(window=w_bb).std()
    bb_df['boolinger_upper'] = bb_df['moving_average'] + (bb_df['standar_deviation'] * sd_bb)
    bb_df['boolinger_lower'] = bb_df['moving_average'] - (bb_df['standar_deviation'] * sd_bb)

    ###Graficar

    #Moving Average 
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x= ma_df.time , y= ma_df.close , mode='lines', name='Close',
                        line=dict(color='gold', width=4)))

    fig1.add_trace(go.Scatter(x= ma_df.time, y= ma_df.moving_average_short , mode='lines', name='Short Moving Average',
                        line=dict(color='firebrick', width=3)))

    fig1.add_trace(go.Scatter(x= ma_df.time, y= ma_df.moving_average_long , mode='lines', name='Long Moving Average',
                        line=dict(color='forestgreen', width=3)))
    
    fig1.update_layout(title='Moving Average', xaxis_title='Date', yaxis_title='Values')

    #Bollinger Bands

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x= bb_df.time , y= bb_df.close , mode='lines', name='Close',
                        line=dict(color='gold', width=4)))

    fig2.add_trace(go.Scatter(x= bb_df.time, y= bb_df.boolinger_upper , mode='lines', name='Bollinger Upper',
                        line=dict(color='forestgreen', width=3)))
    
    fig2.add_trace(go.Scatter(x= bb_df.time, y= bb_df.boolinger_lower , mode='lines', name='Bollinger Close',
                        line=dict(color='olive', width=3)))
    
    fig2.update_layout(title='Bollinger Bands', xaxis_title ='Date', yaxis_title='Values')


    ### Constantes
    Capital = 100000
    hoy = pd.Timestamp(datetime.datetime.today())
    save_time_venta = ma_df['time'][0] - datetime.timedelta(hours=2)
    save_time_compra = ma_df['time'][0] - datetime.timedelta(hours=2)

    for t in range(1, len(ma_df)):
        # For Moving Average
        ## Compras
        if (ma_df.loc[t - 1, 'moving_average_short'] < ma_df.loc[t - 1, 'moving_average_long'] and
                ma_df.loc[t, 'moving_average_short'] > ma_df.loc[t, 'moving_average_long'] and
                ma_df.loc[t, 'close'] * num_acciones <= Capital):
            final_ma.loc[t, 'Operation'] = 'Comprar'
            final_ma.loc[t, 'exposure'] = ma_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_ma.loc[t, 'Stoploss'] = max(ma_df.loc[t, 'close'] - ((Capital * 0.1) / num_acciones),
                                              ma_df.loc[t, 'close'] * (1 - pp_ma))
            final_ma.loc[t, 'Takeprofit'] = ma_df.loc[t, 'close'] * (1 + pg_ma)
            # Actualización Capital
            Capital = Capital - (ma_df.loc[t, 'close'] * num_acciones)
            # Cerrando posición
            ## Precio Stoploss
            close_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] <= final_ma.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_ma.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] >= final_ma.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_ma.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] <= final_ma.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_ma.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] >= final_ma.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_ma.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_ma.loc[t, 'CloseOp'] = np.where(final_ma.loc[t, 'Date_SL'] > final_ma.loc[t, 'Date_TP'],
                                                  final_ma.loc[t, 'TP'],
                                                  final_ma.loc[t, 'SL'])
            # Profit
            final_ma.loc[t, 'Profit'] = (final_ma.loc[t, 'CloseOp'] - final_ma.loc[t, 'close']) * num_acciones if \
                final_ma.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_ma.loc[t, 'CloseOp'] * num_acciones)
        ## Ventas
        elif (ma_df.loc[t - 1, 'moving_average_short'] > ma_df.loc[t - 1, 'moving_average_long'] and
              ma_df.loc[t, 'moving_average_short'] < ma_df.loc[t, 'moving_average_long'] and
              ma_df.loc[t, 'close'] * num_acciones <= Capital):
            final_ma.loc[t, 'Operation'] = 'Vender'
            final_ma.loc[t, 'exposure'] = ma_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_ma.loc[t, 'Stoploss'] = min(ma_df.loc[t, 'close'] + ((Capital * 0.1) / num_acciones),
                                              ma_df.loc[t, 'close'] * (1 + pp_ma))
            final_ma.loc[t, 'Takeprofit'] = ma_df.loc[t, 'close'] * (1 - pg_ma)
            # Actualización Capital
            Capital = Capital - ma_df.loc[t, 'close'] * num_acciones
            # Cerrando posición
            ## Precio Stoploss
            close_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] >= final_ma.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_ma.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                             (ma_df['close'] <= final_ma.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_ma.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] >= final_ma.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_ma.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = ma_df[(ma_df['time'] >= final_ma.loc[t, 'time']) &
                            (ma_df['close'] <= final_ma.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_ma.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_ma.loc[t, 'CloseOp'] = np.where(final_ma.loc[t, 'Date_SL'] > final_ma.loc[t, 'Date_TP'],
                                                  final_ma.loc[t, 'TP'],
                                                  final_ma.loc[t, 'SL'])
            # Profit
            final_ma.loc[t, 'Profit'] = (final_ma.loc[t, 'close'] - final_ma.loc[t, 'CloseOp']) * num_acciones if \
                final_ma.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_ma.loc[t, 'CloseOp'] * num_acciones)

        # For Boolinger Bands
        ## Compras
        if (bb_df.loc[t - 1, 'close'] > bb_df.loc[t - 1, 'boolinger_lower'] and
                bb_df.loc[t, 'close'] < bb_df.loc[t, 'boolinger_lower'] and
                bb_df.loc[t, 'close'] * num_acciones <= Capital and
                bb_df.loc[t, 'time'] - save_time_compra > datetime.timedelta(hours=2)):

            save_time_compra = bb_df.loc[t, 'time']

            final_bb.loc[t, 'Operation'] = 'Comprar'

            final_bb.loc[t, 'exposure'] = bb_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_bb.loc[t, 'Stoploss'] = max(bb_df.loc[t, 'close'] - ((Capital * 0.1) / num_acciones),
                                              bb_df.loc[t, 'close'] - (
                                                          bb_df.loc[t, 'moving_average'] - bb_df.loc[t, 'close']))
            final_bb.loc[t, 'Takeprofit'] = bb_df.loc[t, 'moving_average']
            # Actualización Capital
            Capital = Capital - bb_df.loc[t, 'close'] * num_acciones
            # Cerrando posición
            ## Precio Stoploss
            close_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] <= final_bb.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_bb.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] >= final_bb.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_bb.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] <= final_bb.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_bb.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] >= final_bb.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_bb.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_bb.loc[t, 'CloseOp'] = np.where(final_bb.loc[t, 'Date_SL'] > final_bb.loc[t, 'Date_TP'],
                                                  final_bb.loc[t, 'TP'],
                                                  final_bb.loc[t, 'SL'])
            # Profit
            final_bb.loc[t, 'Profit'] = (final_bb.loc[t, 'CloseOp'] - final_bb.loc[t, 'close']) * num_acciones if \
                final_bb.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_bb.loc[t, 'CloseOp'] * num_acciones)
        ## Ventas
        elif (bb_df.loc[t - 1, 'close'] < bb_df.loc[t - 1, 'boolinger_upper'] and
              bb_df.loc[t, 'close'] > bb_df.loc[t, 'boolinger_upper'] and
              bb_df.loc[t, 'close'] * num_acciones <= Capital and
              bb_df.loc[t, 'time'] - save_time_venta > datetime.timedelta(hours=2)):

            save_time_venta = bb_df.loc[t, 'time']

            final_bb.loc[t, 'Operation'] = 'Vender'
            final_bb.loc[t, 'exposure'] = bb_df.loc[t, 'close'] * num_acciones
            # Cálculo stoploss & takeprofit
            final_bb.loc[t, 'Stoploss'] = min(bb_df.loc[t, 'close'] + ((Capital * 0.1) / num_acciones),
                                              bb_df.loc[t, 'close'] + (
                                                          bb_df.loc[t, 'close'] - bb_df.loc[t, 'moving_average']))
            final_bb.loc[t, 'Takeprofit'] = bb_df.loc[t, 'moving_average']
            # Actualización Capital
            Capital = Capital - bb_df.loc[t, 'close'] * num_acciones
            # Cerrando posición
            ## Precio Stoploss
            close_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] >= final_bb.loc[t, 'Stoploss'])].close.to_list()
            close_sl.append([0])
            final_bb.loc[t, 'SL'] = close_sl[0]
            ## Precio Takeprofit
            close_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                             (bb_df['close'] <= final_bb.loc[t, 'Takeprofit'])].close.to_list()
            close_tp.append([0])
            final_bb.loc[t, 'TP'] = close_tp[0]
            ## Time Stoploss
            time_sl = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] >= final_bb.loc[t, 'Stoploss'])].time.to_list()
            time_sl.append([hoy])
            final_bb.loc[t, 'Date_SL'] = time_sl[0]
            ## Time Takeprofit
            time_tp = bb_df[(bb_df['time'] >= final_bb.loc[t, 'time']) &
                            (bb_df['close'] <= final_bb.loc[t, 'Takeprofit'])].time.to_list()
            time_tp.append([hoy])
            final_bb.loc[t, 'Date_TP'] = time_tp[0]
            # Operación cerrada
            final_bb.loc[t, 'CloseOp'] = np.where(final_bb.loc[t, 'Date_SL'] > final_bb.loc[t, 'Date_TP'],
                                                  final_bb.loc[t, 'TP'],
                                                  final_bb.loc[t, 'SL'])
            # Profit
            final_bb.loc[t, 'Profit'] = (final_bb.loc[t, 'close'] - final_bb.loc[t, 'CloseOp']) * num_acciones if \
                final_bb.loc[t, 'CloseOp'] != 0 else 0
            # Actualización Capital después de cerrar posición
            Capital = Capital + (final_bb.loc[t, 'CloseOp'] * num_acciones)

    # Earnings Moving Average
    earnings_df_ma = final_ma.dropna(subset=['Operation'])
    earnings_df_ma.reset_index(drop=True, inplace=True)
    # Earnings Boolinger Bands
    earnings_df_bb = final_bb.dropna(subset=['Operation'])
    earnings_df_bb.reset_index(drop=True, inplace=True)
    earnings_df_bb.rename(columns={'Operation': 'Operation_bb',
                                   'exposure': 'exposure_bb',
                                   'Stoploss': 'Stoploss_bb',
                                   'Takeprofit': 'Takeprofit_bb',
                                   'SL': 'SL_bb',
                                   'TP': 'TP_bb',
                                   'Date_SL': 'Date_SL_bb',
                                   'Date_TP': 'Date_TP_bb',
                                   'CloseOp': 'CloseOp_bb',
                                   'Profit': 'Profit_bb'}, inplace=True)
    # Merging Earnings
    df_ma_bb = df.merge(earnings_df_ma[['time', 'Operation', 'exposure',
                                        'Stoploss', 'Takeprofit', 'SL',
                                        'TP', 'Date_SL', 'Date_TP',
                                        'CloseOp', 'Profit']],
                        left_on='time', right_on='time', how='outer').merge(earnings_df_bb[['time',
                                                                                            'Operation_bb',
                                                                                            'exposure_bb',
                                                                                            'Stoploss_bb',
                                                                                            'Takeprofit_bb', 'SL_bb',
                                                                                            'TP_bb', 'Date_SL_bb',
                                                                                            'Date_TP_bb',
                                                                                            'CloseOp_bb', 'Profit_bb']],
                                                                            left_on='time', right_on='time',
                                                                            how='outer')
    df_ma_bb.fillna(0, inplace=True)
    df_ma_bb.sort_values(by='time', inplace=True, ignore_index=True)
    df_ma_bb = df_ma_bb[(df_ma_bb['Operation'] != 0) | (df_ma_bb['Operation_bb'] != 0)].reset_index(drop=True)
    df_ma_bb["Operation"] = df_ma_bb["Operation"].astype("str")
    df_ma_bb["Operation_bb"] = df_ma_bb["Operation_bb"].astype("str")
    df_ma_bb["Duplicates"] = np.where((((df_ma_bb['Operation'] == "Comprar") & (df_ma_bb['Operation_bb'] == "Vender")) | \
                                       ((df_ma_bb['Operation'] == "Vender") & (df_ma_bb['Operation_bb'] == "Comprar"))),
                                      1, 0)
    df_ma_bb = df_ma_bb[df_ma_bb['Duplicates'] == 0].copy()
    # Getting Final columns
    list_columns = ['exposure', 'Stoploss', 'Takeprofit', 'SL',
                    'TP', 'CloseOp', 'Profit']
    for i in list_columns:
        df_ma_bb[i + '_final'] = df_ma_bb[i] + df_ma_bb[i + '_bb']

    df_ma_bb['Date_SL_final'] = np.where(df_ma_bb['Date_SL'] == 0, df_ma_bb['Date_SL_bb'], df_ma_bb['Date_SL'])
    df_ma_bb['Date_TP_final'] = np.where(df_ma_bb['Date_TP'] == 0, df_ma_bb['Date_TP_bb'], df_ma_bb['Date_TP'])
    # Profit acum
    df_ma_bb['Profit_final_acum'] = df_ma_bb['Profit_final'].cumsum()

    # Sharpe Ratio
    ret = df_ma_bb['Profit_final_acum'].pct_change().dropna()
    mean_ret = np.mean(ret)
    rf = 0.05
    desvest = np.std(ret)
    sharpe_rat = (mean_ret - rf) / desvest

    # Drawdown
    drawdown_cap = df_ma_bb['Profit_final'].min()

    # Drawup
    drawup_cap = df_ma_bb['Profit_final'].max()

    est_mad = {'Métrica' : ['Sharpe', 'Drawdown','Drawup'],
               '' : ['Cantidad','DrawDown $ (capital)','DrawUp $ (capital)'],
               'Valor' : [sharpe_rat, drawdown_cap, drawup_cap],
               'Descripción' : ['Sharpe Ratio Fórmula Original',
                               'Máxima pérdida flotante registrada',
                                'Máxima ganancia flotante registrada']
              }
    # Crear DataFrame
    df_est_mad = pd.DataFrame(est_mad)

    # Arreglar la tabla

    
    df_ma_bb['Operation_final']= np.where(df_ma_bb['Operation']=='0',df_ma_bb['Operation_bb'],df_ma_bb['Operation'])

    df_ma_bb_profit= df_ma_bb[['time','Operation_final','Profit_final','Profit_final_acum']]

    return sharpe_rat, df_est_mad, df_ma_bb, df_ma_bb_profit, fig1.show(), fig2.show()

def resultados_heuristicos():
    resultados = []
    for n_a in range(1000, 10000, 1000):
        for p_ma in np.arange(0.01, 0.05, 0.01):
            for w_b in range(40, 100, 10):
                for d_b in range(1, 4, 1):
                    resultados.append(cost_function(n_a, p_ma, w_b, d_b))
                    print(n_a, p_ma, w_b, d_b)
                    print(resultados[-1])
    
    # Plot result heuristic
    fig = go.Figure([go.Scatter( y= resultados,
                            line = dict(color='teal', width=3))])
    fig.update_layout(title='Heuristico',
                    xaxis_title='Iteration',
                    yaxis_title='Values')

    return resultados,fig.show()

def five_visualizations(df_ma_bb):
    
    #Precio
    fig_precio = go.Figure()
    fig_precio.add_trace(go.Scatter(x= df_ma_bb['time'], y= df_ma_bb['close'],
                        mode='lines',
                        name='Close moving average',
                        line=dict(color='royalblue', width=3)))

    fig_precio.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # EXPOSURE
    fig_exposure = go.Figure([go.Scatter(x= df_ma_bb['time'], y= df_ma_bb['exposure_final'],
                            line = dict(color='teal', width=3))])
    fig_exposure.update_layout(title='Exposure',
                    xaxis_title='Date',
                    yaxis_title='$')
    
    # Profit acum
    fig_profacum = go.Figure([go.Scatter(x= df_ma_bb['time'], y= df_ma_bb['Profit_final_acum'],
                            line = dict(color='indigo', width=3))])
    fig_profacum.update_layout(title='Profit Acumulado',
                    xaxis_title='Date',
                    yaxis_title='$')
    
    ########
    ventas_ma = df_ma_bb[df_ma_bb['Operation']=='Vender']
    ventas_bb = df_ma_bb[df_ma_bb['Operation_bb']=='Vender']

    ######

    fig_SLMA = go.Figure()
    fig_SLMA.add_trace(go.Scatter(x= ventas_ma['time'], y= ventas_ma['close'],
                        mode='lines',
                        name='Close',
                        line=dict(color='gold', width=4)))

    fig_SLMA.add_trace(go.Scatter(x= ventas_ma['time'], y= ventas_ma['Stoploss'],
                        mode='lines', name='Stoploss',
                        line=dict(color='firebrick', width=3)))

    fig_SLMA.add_trace(go.Scatter(x= ventas_ma['time'], y= ventas_ma['Takeprofit'],
                        mode='lines', name='Takeprofit',
                        line=dict(color='forestgreen', width=3)))
    # Edit the layout
    fig_SLMA.update_layout(title='Close vs Stoploss',
                    xaxis_title='Date',
                    yaxis_title='Values')
    
    #####
    fig_SLBB = go.Figure()
    fig_SLBB.add_trace(go.Scatter(x= ventas_bb['time'], y= ventas_bb['close'],
                        mode='lines',
                        name='Close',
                        line=dict(color='gold', width=4)))

    fig_SLBB.add_trace(go.Scatter(x= ventas_bb['time'], y= ventas_bb['Stoploss_bb'],
                        mode='lines', name='Stoploss',
                        line=dict(color='firebrick', width=3)))

    fig_SLBB.add_trace(go.Scatter(x= ventas_bb['time'], y= ventas_bb['Takeprofit_bb'],
                        mode='lines', name='Takeprofit',
                        line=dict(color='forestgreen', width=3)))
    # Edit the layout
    fig_SLBB.update_layout(title='Close vs Stoploss',
                    xaxis_title='Date',
                    yaxis_title='Values')

    compras_ma = df_ma_bb[df_ma_bb['Operation']=='Comprar']
    compras_bb = df_ma_bb[df_ma_bb['Operation_bb']=='Comprar']

    fig_TPMA = go.Figure()
    fig_TPMA.add_trace(go.Scatter(x=compras_ma['time'], y=compras_ma['close'],
                        mode='lines',
                        name='Close',
                        line=dict(color='olivedrab', width=4)))

    fig_TPMA.add_trace(go.Scatter(x=compras_ma['time'], y=compras_ma['Takeprofit'],
                        mode='lines', name='Takeprofit',
                        line=dict(color='darkorange', width=3)))

    fig_TPMA.add_trace(go.Scatter(x=compras_ma['time'], y=compras_ma['Stoploss'],
                        mode='lines', name='Stoploss',
                        line=dict(color='cadetblue', width=3)))

    # Edit the layout
    fig_TPMA.update_layout(title='Close vs Takeprofit',
                    xaxis_title='Date',
                    yaxis_title='Values')
    
    #####
    fig_TPBB = go.Figure()
    fig_TPBB.add_trace(go.Scatter(x=compras_bb['time'], y=compras_bb['close'],
                        mode='lines',
                        name='Close',
                        line=dict(color='olivedrab', width=4)))

    fig_TPBB.add_trace(go.Scatter(x=compras_bb['time'], y=compras_bb['Takeprofit_bb'],
                        mode='lines', name='Takeprofit',
                        line=dict(color='darkorange', width=3)))

    fig_TPBB.add_trace(go.Scatter(x=compras_bb['time'], y=compras_bb['Stoploss_bb'],
                        mode='lines', name='Stoploss',
                        line=dict(color='cadetblue', width=3)))

    # Edit the layout
    fig_TPBB.update_layout(title='Close vs Takeprofit',
                    xaxis_title='Date',
                    yaxis_title='Values')


    return fig_precio.show(), fig_exposure.show(), fig_profacum.show(), fig_SLMA.show(), fig_TPBB.show()
    

def PSO_Optimization(constraints = (np.array([100, 0.01, 20, 0.05]),np.array([5000, 0.05, 100, 4])),
                     dimensions_num = 4,  iterations = 10):

    # Time of execution
    start_time = time.time()
    # Set-up hyperparameters
    options = {'c1': 2, 'c2': 2.2, 'w':0.73}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions_num, options=options, bounds=constraints)

    # Perform optimization
    cost, pos = optimizer.optimize(cost_function, iters=iterations)

    # Graficar

    fig = go.Figure([go.Scatter( y= np.array(optimizer.cost_history)*-1,
                            line = dict(color='deepskyblue', width=3))])
    fig.update_layout(title='Optimizer',
                    xaxis_title='Iterations',
                    yaxis_title='Sharpe')
    
    final_time = time.time() - start_time

    return -cost, pos, final_time , fig.show()




