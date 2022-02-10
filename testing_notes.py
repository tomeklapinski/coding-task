import pandas as pd
import re

import pmdarima as pm
import numpy as np
import matplotlib.pyplot as plt
import logging

#data = pm.datasets.load_wineind()
#train, test = pm.model_selection.train_test_split(data, train_size=150)
#arima = pm.auto_arima(train, trace=True, suppress_warnings=True, maxiter=5, seasonal=True, m=12)

#x = np.arange(test.shape[0])
#plt.scatter(x, test, marker='x')
#plt.plot(x, arima.predict(n_periods=test.shape[0]))
#plt.title('Actual test samples vs. forecasts')
#plt.show()

'''
# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
'''
'''
series_code = 'NY.GDP.MKTP.CN'
country_code = 'AFG;'
a = 143434

if a:
    print('None is true')
else:
    print('None is false')

try:
    if not re.fullmatch('[A-Za-z]+', country_code):
        raise ValueError
    if not re.fullmatch('[A-Za-z.]+', series_code):
        raise ValueError
except ValueError:
    print('Incorrect script parameters. Need series code as first parameter and country code as second.\nProgram terminates.')



array = np.array([])
array = np.append(array,[1])
array = np.append(array,[1])
print(array)


df = pd.DataFrame(columns=['date','value'])
print(df)
df = df.append({'date': '2020', 'value': 1}, ignore_index=True)
df = df.append({'date': '2021M07', 'value': 10}, ignore_index=True)

a = np.array([1,2])
b = None
print(type(b))
if a is not None:
    print('ssss')

print(df.set_index('date').T)
dict = df.set_index('date').T.to_dict('records')[0]
print(type(dict),dict)
#print(reversed(df.index))
#for item in reversed(df.date):
#    print(item)

date = df['date'][1]
print(type(date))
def get_formatted_date(date):
    if re.fullmatch(r'\d{4}',date):
        date = date +'-01-01'
        print(date)
    elif re.fullmatch(r'\d{4}M\d{2}', date):
        date = re.sub(r'M','-',date)
        date = date + '-01'
        print(date)
    elif re.fullmatch(r'\d{4}Q\d{1}', date):
        if date[5] == '1':
            date = re.sub(r'Q1', '-01-01', date)
        elif date[5] == '2':
            date = re.sub(r'Q2', '-04-01', date)
        elif date[5] == '3':
            date = re.sub(r'Q3', '-07-01', date)
        elif date[5] == '4':
            date = re.sub(r'Q4', '-10-01', date)
    return date

formatted_date = df['date'].map(get_formatted_date)
df['date'] = formatted_date
print(df)'''