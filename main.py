from json import JSONDecodeError
from urllib.error import URLError
from abc import ABC, abstractmethod

import pmdarima as pm
from prophet import Prophet
import re
import sys
import urllib, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#####   PART 1 - LOADING AND CLEANING THE DATA  #####


### Reading parameters from command line interface
series_code = sys.argv[1]
country_code = sys.argv[2]
series_code = 'NY.GDP.MKTP.CN'
country_code = 'AFG'

if not re.fullmatch('[A-Za-z.]+', series_code) or not re.fullmatch('[A-Za-z]+', country_code):
    print('Incorrect script parameters. Need series code as first parameter and country code as second.\nProgram terminates.')
    raise ValueError

### Initial query for reading data parameters
query_string = 'https://api.worldbank.org/v2/country/' + country_code +'/indicator/' + series_code + '?format=json'
try:
    data_handle = urllib.request.urlopen(query_string)
    data = json.loads(data_handle.read())
except URLError:
    print('Establishing connection to URL was unsuccessful.\nProgram terminates')
    sys.exit(1)
except JSONDecodeError:
    print('Data to deserialize should be in json format.\nProgram terminates')
    sys.exit(1)

if not re.fullmatch('[0-9]+', str(data[0]['pages'])) or not re.fullmatch('[0-9]+', str(data[0]['per_page'])) or not re.fullmatch('[0-9]+', str(data[0]['total'])):
    print('Incorrect data parameters, such as number of pages, records, records per page.\nProgram terminates.')
    raise ValueError
else:
    number_of_pages = data[0]['pages']
    number_of_records_per_page = data[0]['per_page']
    total_number_of_records = data[0]['total']



### Query for each data page and load date and value field into data frame
data_df = pd.DataFrame(columns=['date','value'])
for n in range(number_of_pages-1,-1,-1):
    # Performing query
    query_string = 'https://api.worldbank.org/v2/country/' + country_code + '/indicator/' + series_code + '?format=json' + '&page=' + str(n+1)
    try:
        data_handle = urllib.request.urlopen(query_string)
        data = json.loads(data_handle.read())
    except URLError:
        print('Establishing connection to URL was unsuccessfull for page ', n+1, '\nPage', n+1,'will be excluded from the data')
    except JSONDecodeError:
        print('Data to deserialize should be json format.\nPage', n+1,'will be excluded from the data')

    # Account for that the number of records might be different on the last page
    if n == number_of_pages-1:
        number_of_records = total_number_of_records % number_of_records_per_page
    else:
        number_of_records = number_of_records_per_page

    # Loading values of data and value keys into data frame
    for i in range(number_of_records-1,-1,-1):
        if re.fullmatch('[0-9MQ]+',str(data[1][i]['date'])) and re.fullmatch('[0-9.,]+',str(data[1][i]['value'])):
            data_df = data_df.append({'date': data[1][i]['date'], 'value': data[1][i]['value']}, ignore_index=True)

if data_df.shape[0] == 0:
    print('The URL query lack of proper data points.\nProgram terminates')
    raise ValueError


### Reformat date to YYYY-MM-DD format
def get_formatted_date(date):
    if re.fullmatch(r'\d{4}',date):
        date = date +'-01-01'
        # print(date)
    elif re.fullmatch(r'\d{4}M\d{2}', date):
        date = re.sub(r'M','-',date)
        date = date + '-01'
        # print(date)
    elif re.fullmatch(r'\d{4}Q\d{1}', date):
        if date[5] == '1':
            date = re.sub(r'Q1', '-01-01', date)
        elif date[5] == '2':
            date = re.sub(r'Q2', '-04-01', date)
        elif date[5] == '3':
            date = re.sub(r'Q3', '-07-01', date)
        elif date[5] == '4':
            date = re.sub(r'Q4', '-10-01', date)
    else:
        print('Data point with date:', date, 'has incorrect format.\nThis point will be excluded form data.')
        date = None
    return date


data_df['date'] = data_df['date'].map(get_formatted_date)
data_df = data_df.dropna()

### Determining how many predictions need to be done and from which year
lastYearDate = None
for item in reversed(data_df.date):
    if item != None:
        lastYearDate = int(item[:4])
        break
number_of_predictions = 2030 - lastYearDate
first_prediction_year = lastYearDate + 1



#####   PART 2 - CREATING AND TRAINING THE MODELS  #####

class Model(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, number_of_predictions, first_prediction_year):
        pass


class ArimaModel(Model):
    def __init__(self):
        self.data_df = None
        self.arima_model = None

    def pmdarima_transformation(self, df):
        data_array = np.array([])
        for i in range(data_df.shape[0]):
            if re.fullmatch(r'\d{4}-01-01', data_df['date'][i]):
                data_array = np.append(data_array, df['value'][i])
        return data_array

    def fit(self, data):
        self.data_df = data
        try:
            if self.data_df.shape[0] == 0:
                raise ValueError
            train_data = self.pmdarima_transformation(self.data_df)
            self.arima_model = pm.auto_arima(train_data, error_action='ignore', trace=True,
                                  suppress_warnings=True, maxiter=5,
                                  seasonal=False)
        except ValueError:
            print('Lack of appropriate data points for ARIMA model.\nPredictions will be made without this model.')
        except:
            print('Creating and training ARIMA model was unsuccessful.\nPredictions will be made without this model.')

    def predict(self, number_of_predictions, first_prediction_year):
        try:
            return self.arima_model.predict(n_periods=number_of_predictions)
        except:
            print('Predicting with ARIMA model was unsuccessful.\nPredictions will be made without this model.')
            return None


class ProphetModel(Model):
    def __init__(self):
        self.data_df = None
        self.prophet_model = Prophet()

    def fit(self, data):
        self.data_df = data
        try:
            self.prophet_model.fit(data_df.rename(columns={'date': 'ds', 'value': 'y'}))
        except:
            print('Creating and training Prophet model was unsuccessful.\nPredictions will be made without this model.')
            
    def predict(self, number_of_predictions, first_prediction_year):
        try:
            future_df = pd.DataFrame(columns=['ds'])
            for n in range(number_of_predictions):
                future_df = future_df.append({'ds': str(first_prediction_year + n) + '-01-01'}, ignore_index=True)
            return self.prophet_model.predict(future_df)['yhat'].squeeze().to_numpy()
        except:
            print('Predicting with Prophet model was unsuccessful.\nPredictions will be made without this model.')
            return None


### Creating desired models
Models = pd.DataFrame(columns=['Name','Object','Predictions'])
Models.loc[0] = ['ARIMA', ArimaModel(), None]
Models.loc[1] = ['Prophet', ProphetModel(), None]


### Train the models
for i in range(Models.shape[0]):
    Models['Object'][i].fit(data_df)



#####   PART 3 - PERFORMING PREDICTIONS AND SAVING TO FILE

### Predicting with models
for i in range(Models.shape[0]):
    Models['Predictions'][i] = Models['Object'][i].predict(number_of_predictions, first_prediction_year)


### Creating predictions based on ensemble of available models predictions
size_of_ensemble = 0
predictions_dict = {str(first_prediction_year + n) + '-01-01': 0 for n in range(number_of_predictions)}

for model_predictions in Models['Predictions']:
    if model_predictions is not None:
        size_of_ensemble += 1
        for n in range(number_of_predictions):
            predictions_dict[str(first_prediction_year + n) + '-01-01'] += model_predictions[n]

if size_of_ensemble > 0:
    for n in range(number_of_predictions):
        predictions_dict[str(first_prediction_year + n) + '-01-01'] /= size_of_ensemble
else:
    print('Predicting with ensemble was unsuccessful due to lack of model predictions.\nProgram terminates.')
    sys.exit(1)


### Creating output dictionary, saving to json file and plotting results
data_dict = data_df.set_index('date').T.to_dict('records')[0]
output_dict = {'data': data_dict, 'predictions': predictions_dict}

try:
    with open('output.json', 'w') as output:
        json.dump(output_dict, output)
except:
    print('Writing to file was unsuccessful.')

plt.plot(*zip(*data_dict.items()))
plt.plot(*zip(*predictions_dict.items()),"r")
plt.show()



