
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import data as dt
from datetime import datetime
import MetaTrader5 as mt5
from functions import historicos, moving_average, boolinger_bands, trading

# -- TEST 1 : 
usuario = 'Daniel Garcia'
symbol = 'EURUSD'
meta_path = 'C:\Program Files\MetaTrader 5 Terminal\\terminal64.exe'
login_count = 5400339 #'Chelsi': 5400342 #'Daniel': 5400339
password_count = '2qeDQrhu' #'Chelsi': 'XN1xho9d' #'Daniel': '2qeDQrhu'
server_name = 'FxPro-MT5'
start_date_backtest = datetime(year=2018, month=1, day=1)
end_date_backtest = datetime(year=2018, month=12, day=31)
start_date_prueba = datetime(year=2019, month=1, day=1)
end_date_prueba = datetime(year=2019, month=12, day=31)

datos_train, datos_prueba = historicos(meta_path, login_count, password_count, server_name,
                                       start_date_backtest, end_date_backtest, start_date_prueba,
                                       end_date_prueba, symbol)
#
# ma = moving_average(meta_path, login_count, password_count, server_name,
#                      start_date_train, end_date_train, usuario, symbol)
#
# bb = boolinger_bands(meta_path, login_count, password_count, server_name,
#                      start_date_train, end_date_train, usuario, symbol)

#resultado = trading(meta_path, login_count, password_count, server_name,
#                    start_date_train, end_date_train, usuario, symbol)


print(datos_train, datos_prueba)
# print(ma)
# print(bb)
#print(resultado)


