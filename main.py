
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
from functions import historicos, function, five_visualizations, resultados_heuristicos, PSO_Optimization

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

#Download data
datos_train, datos_prueba = historicos(meta_path, login_count, password_count, server_name,
                                       start_date_backtest, end_date_backtest, start_date_prueba,
                                       end_date_prueba, symbol)
print(datos_train, datos_prueba)

# Using trading sistem
num_acciones = 100
pp_ma = 0.02
pg_ma = 0.02
w_bb = 100
sd_bb = 2

funcion_ma_bb = function(num_acciones, pp_ma, w_bb, sd_bb, 'backtest')
print(funcion_ma_bb[1])
print(funcion_ma_bb[2])
print(funcion_ma_bb[3])

# Trading Visualizations
five_visualizations(funcion_ma_bb[2])

# Euristic optimization
heuristic = resultados_heuristicos()

# PSO Optimization
cost = PSO_Optimization()
# Checking pso results
funcion_ma_bb_opt = function(cost[1][0],cost[1][1],cost[1][2],cost[1][3], 'backtest')
print(funcion_ma_bb_opt[1])
print(funcion_ma_bb_opt[2])
print(funcion_ma_bb_opt[3])
# PSO Visualizations
five_visualizations(funcion_ma_bb_opt[2])

# Test Period
funcion_ma_bb_prueba = function(cost[1][0], cost[1][1], cost[1][2], cost[1][3], 'prueba')
print(funcion_ma_bb_prueba[1])
print(funcion_ma_bb_prueba[2])
print(funcion_ma_bb_prueba[3])
# PSO Visualizations test
five_visualizations(funcion_ma_bb_prueba[2])