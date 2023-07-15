import pandas as pd
import itertools
import warnings
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

data = pd.read_csv('Electric_Production.csv')

data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)

start_date = pd.to_datetime('1985-01-01')
end_date = pd.to_datetime('2018-12-31')
forecast_date = pd.to_datetime('2018-01-01')

data = data.loc[start_date:end_date]

y = data['IPG2211A2N'].resample('MS').mean()

y = y.fillna(y.bfill())

y.plot(figsize=(15, 6))
plt.show()

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore")

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['1985':].plot(label='Observed')
pred.predicted_mean.plot(ax=ax, label='One-step Ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Electric Production')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2012-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of the forecast is {}'.format(round(mse, 2)))

pred_dynamic = results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = y['1985':].plot(label='Observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(ax=ax, label='Dynamic Forecast')
ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Electric Production')
plt.legend()
plt.show()

forecast_period = 500
pred_uc = results.get_forecast(steps=forecast_period)

forecasted_values = pred_uc.predicted_mean
forecasted_ci = pred_uc.conf_int()

forecast_index = pd.date_range(start=forecast_date + pd.DateOffset(months=1), periods=forecast_period, freq='MS')

forecast_df = pd.DataFrame(index=forecast_index)
forecast_df['Forecast'] = forecasted_values

ax = y.plot(label='Observed', figsize=(20, 15))
forecast_df['Forecast'].plot(ax=ax, label='Forecast')
ax.fill_between(forecast_df.index,
                forecasted_ci.iloc[:, 0],
                forecasted_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Electric Production')
plt.legend()
plt.show()