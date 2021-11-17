
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
import MetaTrader5 as mt5


def historicos(path, login, password, server, start_date, end_date, save_name, symbol):
    connection = mt5.initialize(path=path, login=login, password=password, server=server)

    tuplas = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.TIMEFRAME_H12)

    df = pd.DataFrame(tuplas)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['time_msc'] = pd.to_datetime(df['time'], unit='ms')

    #
    # reportpath = path.abspath('ReportesDeals_MT5/') + "/"
    # save_name = mt5.account_info().name
    # df.to_excel(reportpath + "Deals_" + save_name + ".xlsx")

    return df
