# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

%matplotlib inline

# %%
df = pd.read_csv('Bitcoin prices.csv')
print(df.info())
print(df.isnull().sum())


# %%
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.info()


# %%
# Create a dataframe with only the Close Stock Price Column
data_target = df.filter(['Close'])

# Convert the dataframe to a numpy array to train the LSTM model
target = data_target.values
print(target)
# Splitting the dataset into training and test
# Target Variable: Close stock price value

training_data_len = round(len(target)* 0.8) # training set has 80% of the data
print(training_data_len)

# Normalizing data before model fitting using MinMaxScaler
# Feature Scaling

sc = MinMaxScaler(feature_range=(0,1))
training_scaled_data = sc.fit_transform(target)
print(training_scaled_data)
print("Min:", np.min(training_scaled_data))
print("Max:", np.max(training_scaled_data))

# %%
train_data = training_scaled_data[0:training_data_len  , : ]

X_train = []
y_train = []
for i in range(180, len(train_data)):
    X_train.append(train_data[i-180:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train) # converting into numpy sequences to train the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print('Number of rows and columns: ', X_train.shape)  #(854 values, 180 time-steps, 1 output)

# %%

model = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 30, return_sequences = True))
model.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 30, return_sequences = True))
model.add(Dropout(0.1))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 30))
model.add(Dropout(0.1))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 30, batch_size = 48)

# %%
# Getting the predicted stock price
test_data = training_scaled_data[training_data_len - 180: , : ]

#Create the x_test and y_test data sets
X_test = []
y_test =  target[training_data_len : , : ]
for i in range(180,len(test_data)):
    X_test.append(test_data[i-180:i,0])

# Convert x_test to a numpy array
X_test = np.array(X_test)

#Reshape the data into the shape accepted by the LSTM
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
print('Number of rows and columns: ', X_test.shape)

# %%
# Making predictions using the test dataset
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

