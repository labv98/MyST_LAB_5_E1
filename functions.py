
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
import MetaTrader5 as mt5


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


def moving_average(path, login, password, server, start_date, end_date, save_name, symbol):
    #df = historicos(path, login, password, server, start_date, end_date, save_name, symbol)
    df = pd.read_excel('Historicos_backtest.xlsx')
    df['moving_average_short'] = df['close'].rolling(window=20).mean()
    df['moving_average_long'] = df['close'].rolling(window=50).mean()

    plt.figure()
    plt.plot(df.close)
    plt.plot(df.moving_average_short)
    plt.plot(df.moving_average_long)
    plt.show()
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
    plt.show()
    return df


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
                bb_df.loc[t, 'close'] * num_acciones <= Capital):  # and
            # bb_df.loc[t, 'time'] - save_time_compra > datetime.timedelta(hours=2)):

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
              bb_df.loc[t, 'close'] * num_acciones <= Capital):  # and
            # bb_df.loc[t, 'time'] - save_time_venta > datetime.timedelta(hours=2)):

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
    # print(variable)
    # num_acciones = variable.item(0)
    # pp_ma = variable.item(1)
    pg_ma = pp_ma
    w_bb = int(w_bb)
    # w_bb = int(variable.item(2))
    # sd_bb = variable.item(3)
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

    return - sharpe_rat



