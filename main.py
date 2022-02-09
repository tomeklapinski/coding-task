from json import JSONDecodeError
from urllib.error import URLError


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

### Initial query for reading data parameters

query_string = 'https://api.worldbank.org/v2/country/' + country_code +'/indicator/' + series_code + '?format=json'
try:
    if not re.fullmatch('[A-Za-z.]+', series_code):
        raise ValueError
    if not re.fullmatch('[A-Za-z]+', country_code):
        raise ValueError
    data_handle = urllib.request.urlopen(query_string)
    data = json.loads(data_handle.read())
except URLError:
    print('Establishing connection to URL was unsuccessful.\nProgram terminates')
    sys.exit(1)
except JSONDecodeError:
    print('Data to deserialize should be in json format.\nProgram terminates')
    sys.exit(1)
except ValueError:
    print('Incorrect script parameters. Need series code as first parameter and country code as second.\nProgram terminates.')
    sys.exit(1)

try:
    number_of_pages = data[0]['pages']
    number_of_records_per_page = data[0]['per_page']
    total_number_of_records = data[0]['total']
    if not re.fullmatch('[0-9]+', str(number_of_pages)):
        raise ValueError
    if not re.fullmatch('[0-9]+', str(number_of_records_per_page)):
        raise ValueError
    if not re.fullmatch('[0-9]+', str(total_number_of_records)):
        raise ValueError
except ValueError:
    print('Incorrect data parameters, such as number of pages, records, records per page.\nProgram terminates.')
    sys.exit(1)

### Query for each page and load date and value field into data frame
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
    print('The URL query lack of data points.\nProgram terminates')
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


formatted_date = data_df['date'].map(get_formatted_date)
data_df['date'] = formatted_date
data_df = data_df.dropna()

### Plotting the data frame
#plot = df.plot(x='date', y='value')
#plt.show()


#####   PART 2 - CREATING AND TRAINING THE MODELS  #####


### Pmdarima model

### Reducting data to annual as prediction will be done annually
def pmdarima_transformation(df):
    data_array = np.array([])
    for i in range(data_df.shape[0]):
        if re.fullmatch(r'\d{4}-01-01', data_df['date'][i]):
            data_array = np.append(data_array,df['value'][i])
    return data_array


### Train the model
try:fff
    train_data = pmdarima_transformation(data_df)
    if data_df.shape[0] == 0:
        raise ValueError
    arima = pm.auto_arima(train_data, error_action='ignore', trace=True,
                      suppress_warnings=True, maxiter=5,
                      seasonal=False)
except ValueError:
    print('Lack of appropriate data points for ARIMA model.\nPredictions will be made without this model.')
except:
    print('Creating and training ARIMA model was unsuccessful.\nPredictions will be made without this model.')




### Prophet model
try:
    model = Prophet()
    model.fit(data_df.rename(columns={'date': 'ds', 'value': 'y'}))
except:
    print('Creating and training Prophet model was unsuccessful.\nPredictions will be made without this model.')


#####   PART 3 - PERFORMING PREDICTIONS AND SAVING TO FILE

### Determining how many predictions need to be done
lastYearDate = None
for item in reversed(data_df.date):
    if item != None:
        lastYearDate = int(item[:4])
        break


### Predicting with pmdarima
try:
    number_of_predictions = 2030 - lastYearDate
    predictions = arima.predict(n_periods=number_of_predictions)
except:
    print('Predicting with ARIMA model was unsuccessful.\nPredictions will be made without this model.')


### Predicting with prophet

#future_df = model.make_future_dataframe(periods=number_of_predictions, freq='Y', include_history=False)
#print(future_df)
try:
    future_df = pd.DataFrame(columns=['ds'])
    for n in range(number_of_predictions):
        future_df = future_df.append({'ds': str(lastYearDate+1+n) + '-01-01'}, ignore_index=True)
    forecast_df = model.predict(future_df)
except:
    print('Predicting with Prophet model was unsuccessful.\nPredictions will be made without this model.')

### Creating output dictinary, ploting results and saving to json file

# Creating predictions from ensemble of predictions or for single model
arima1 = True
prophet1 = True
if arima1 or prophet1:
    predictions_dict = dict()
    for n in range(number_of_predictions):
        if arima1 and prophet1:
            prediction = (predictions[n] + forecast_df['yhat'][n])/2
        elif arima1 and not prophet1:
            prediction = predictions[n]
        elif not arima1 and prophet1:
            prediction = forecast_df['yhat'][n]
        predictions_dict[str(lastYearDate+1+n) + '-01-01'] = prediction
else:
    print('Predicting with both models was unsuccessful, performing overall prediction is unavailable.\nProgram terminates.')
    sys.exit(1)

# Creating output dictionary
data_dict = data_df.set_index('date').T.to_dict('records')[0]
output_dict = {'data': data_dict, 'predictions': predictions_dict}

# Saving data and predictions dictionary in the file
try:
    with open('output.json', 'w') as output:
        json.dump(output_dict, output)
except:
    print('Writing to file was unsuccessful.')


# Plotting predictions
plt.plot(*zip(*data_dict.items()))
plt.plot(*zip(*predictions_dict.items()),"r")
plt.show()



