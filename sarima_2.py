import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

production = pd.read_csv('Electric_Production.csv',
                       index_col ='DATE',
                       parse_dates = True)

result = seasonal_decompose(production['IPG2211A2N'], 
                            model ='multiplicative')

result.plot()
plt.show()

from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(production['IPG2211A2N'], start_p = 1, start_q = 1,
                          max_p = 3, max_q = 3, m = 12,
                          start_P = 0, seasonal = True,
                          d = None, D = 1, trace = True,
                          error_action ='ignore',   
                          suppress_warnings = True,  
                          stepwise = True)           

print(stepwise_fit.summary())

train = production.iloc[:len(production) - 12 * 2]
test = production.iloc[len(production) - 12 * 2:] 

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

model = model = SARIMAX(train['IPG2211A2N'], 
                        order = (1, 1, 1), 
                        seasonal_order =(2, 1, 2, 12))
result = model.fit()

predictOnTrainData = result.predict(start = 0, 
                          end = (len(train) - 1), 
                          typ = 'levels').rename('Forecast on Training Data')
predictOnTestData = result.predict(start = len(train), 
                          end = (len(test) + len(train) - 1), 
                          typ = 'levels').rename('Forecast on Test Data')

production['IPG2211A2N'].plot(color = 'green', figsize = (12, 6), legend = True)
predictOnTrainData.plot(legend = True)
predictOnTestData.plot(legend = True)

plt.show()

print(mean_squared_error(test["IPG2211A2N"], predictOnTestData))
print(mean_squared_error(train["IPG2211A2N"], predictOnTrainData))


# Forecast
model = model = SARIMAX(production['IPG2211A2N'], 
                        order = (1, 1, 1), 
                        seasonal_order =(2, 1, 2, 12))
result = model.fit()

forecast = result.predict(start = len(production), 
                          end = (len(production) - 1) + 4 * 12, 
                          typ = 'levels').rename('Forecast for next the 4 years')

production['IPG2211A2N'].plot(figsize = (12, 6), legend = True)
forecast.plot(color = 'red', legend = True)

plt.show()