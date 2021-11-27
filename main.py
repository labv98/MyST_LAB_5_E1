
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
login_count = 5400339 #'Bruno': 5400338 #'Chelsi': 5400342 #'Daniel': 5400339
password_count = '2qeDQrhu' #'Bruno': 'LHFFV4Nh' #'Chelsi': 'XN1xho9d' #'Daniel': '2qeDQrhu'
server_name = 'FxPro-MT5'
start_date_train = datetime(year=2018, month=1, day=1)
end_date_train = datetime(year=2018, month=12, day=31)
start_date_test = datetime(2019, 2, 1)
end_date_test = datetime(2020, 2, 1)

datos_train = historicos(meta_path, login_count, password_count, server_name,
                         start_date_train, end_date_train, usuario, symbol)
#
# ma = moving_average(meta_path, login_count, password_count, server_name,
#                      start_date_train, end_date_train, usuario, symbol)
#
# bb = boolinger_bands(meta_path, login_count, password_count, server_name,
#                      start_date_train, end_date_train, usuario, symbol)

#resultado = trading(meta_path, login_count, password_count, server_name,
#                    start_date_train, end_date_train, usuario, symbol)


print(datos_train)
# print(ma)
# print(bb)
#print(resultado)


