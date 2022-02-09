#import pystan
#model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
#model = pystan.StanModel(model_code=model_code)
#y = model.sampling().extract()['y']
#y.mean()  # with luck the result will be near 0

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


m = Prophet()

df = pd.read_csv('example_wp_log_peyton_manning.csv')
print(df.head())

m.fit(df)

future = m.make_future_dataframe(periods=365)
print(future.head())

forecast = m.predict(future)
print(forecast.head())
print(forecast.tail())

fig1 = m.plot(forecast)
plt.show()

fig2 = m.plot_components(forecast)
plt.show()


