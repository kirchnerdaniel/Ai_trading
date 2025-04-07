from sklearn.preprocessing import MinMaxScaler
from indicators import  calculate_rsi, calculate_vwap
import numpy as np
from keras import Sequential, layers, callbacks, optimizers, Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot  as plt

class RNNModel:
    
    # Beállítom a modellt. Forecast_horizon: ennyivel jósol későbbre, timestep: ennyi időegységet vesz figyelembe, split: szét osztja validációs és tanító adatokra 
    def __init__(self, forecast_horizon=3, timestep=10, df=None, candles=None, split=1):
        
        self.forecast_horizon = forecast_horizon
        self.timestep = timestep
        self.X_scaler = MinMaxScaler(feature_range=(0,1)) #2 skálázót használtam, mert az Y-ban, csak 1 féle érték van
        self.Y_scaler = MinMaxScaler(feature_range=(0,1))
        self.model = self.create_model(df, candles, split)

    def fetch_candlestick_data_with_indicators(self, df=None, candles=None):

        if df is not None and candles is not None:
            raise ValueError("Provide either 'df' or 'candles', not both.")

        if candles is not None:
            closing_prices = [float(candle[4]) for candle in candles]
            opening_prices = [float(candle[1]) for candle in candles]
            high_prices = [float(candle[2]) for candle in candles]
            low_prices = [float(candle[3]) for candle in candles]
            volumen = [float(candle[5]) for candle in candles]

        elif df is not None:
            closing_prices = df['Close'].values
            opening_prices = df['Open'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            volumen = df['Volume'].values

        else:
            raise ValueError("Either 'df' or 'candles' must be provided.")
        
        #Indikátorokat számol
        vwap_values = calculate_vwap(close_prices=closing_prices, high_prices=high_prices, low_prices=low_prices, volume=volumen, period=8)
        rsi_values = calculate_rsi(close_prices=closing_prices, period=8)
        
        x_data = []
        y_data = []

        min_length = min(len(vwap_values), len(closing_prices), len(rsi_values), len(closing_prices))

        closing_prices = closing_prices[-min_length:]
        opening_prices = opening_prices[-min_length:]
        high_prices = high_prices[-min_length:]
        low_prices = low_prices[-min_length:]
        volumen = volumen[-min_length:]
        rsi_values = rsi_values[-min_length:]
        vwap_values = vwap_values[-min_length:]
        
        #Ezeket az adatokat kapja, azért, hogy ne legyen trend. Az RSI 0-100 között mozog. A volumen pár helyen 0, ezért van az epsilon
        epsilon = 1e-8
    
        for i in range(1, min_length - self.forecast_horizon):  
            x_data.append([
                (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1],
                rsi_values[i],  
                (volumen[i] - volumen[i-1]) / (volumen[i-1] + epsilon),  
                (vwap_values[i] - vwap_values[i-1]) / vwap_values[i-1]  
            ])

            y_data.append([
                (closing_prices[i + self.forecast_horizon] - closing_prices[i]) / closing_prices[i]
            ])

        print(len(x_data), len(y_data))

        return np.array(x_data), np.array(y_data)


    def create_time_series_data(self, X, Y):
        X_time_series, Y_time_series = [], []
        
        for i in range(len(X) - self.timestep):
            X_time_series.append(X[i : i + self.timestep])  # Az elmúlt `timestep` értékek
            Y_time_series.append(Y[i + self.timestep])      # A timestep utáni következő érték
        
        return np.array(X_time_series), np.array(Y_time_series)


    def prepare_learning_data(self, X_data, Y_data, split):

        # 1. Idősoros átalakítás 
        X_data_ts, Y_data_ts = self.create_time_series_data(X_data, Y_data)

        # 2. Szétválasztás tanító és teszt adatokra
        split_index = int(len(X_data_ts) * split)
        X_train_data, X_test_data = X_data_ts[:split_index], X_data_ts[split_index:]
        Y_train_data, Y_test_data = Y_data_ts[:split_index], Y_data_ts[split_index:]

        # 3. Skálázás
        scaled_train_data = self.X_scaler.fit_transform(X_train_data.reshape(-1, X_train_data.shape[-1])).reshape(X_train_data.shape)
        scaled_train_close = self.Y_scaler.fit_transform(Y_train_data.reshape(-1, 1)).reshape(Y_train_data.shape)

        scaled_test_data = self.X_scaler.transform(X_test_data.reshape(-1, X_test_data.shape[-1])).reshape(X_test_data.shape)
        scaled_test_close = self.Y_scaler.transform(Y_test_data.reshape(-1, 1)).reshape(Y_test_data.shape)


        return scaled_train_data, scaled_test_data, scaled_train_close, scaled_test_close

    #Ezeket úgy másoltam ki, csak átírtam 1D-re, küldöm róla a linket
    def IdentityBlock(self, prev_Layer , filters):
        f1 , f2 , f3 = filters

        x = layers.Conv1D(filters=f1, kernel_size=1 , strides=1, padding='valid')(prev_Layer)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        
        x = layers.Conv1D(filters=f2, kernel_size = 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        
        x = layers.Conv1D(filters=f3, kernel_size = 1, strides=1, padding='valid')(x)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        
        x = layers.concatenate([ x, prev_Layer ], axis=-1)
        x = layers.Activation(activation='relu')(x)

        return x   

    def ConvBlock(self, prev_Layer , filters , strides):
        f1 , f2 , f3 = filters
        
        #Path 1
        x = layers.Conv1D(filters=f1, kernel_size = 1 ,padding='valid', strides=strides)(prev_Layer)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        
        x = layers.Conv1D(filters=f2, kernel_size = 3 , padding='same' , strides=1)(x)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        
        x = layers.Conv1D(filters=f3, kernel_size = 1, padding='valid' , strides=1)(x)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        
        #Path 2
        
        x2 = layers.Conv1D(filters=f3, kernel_size=1, padding='valid' , strides=strides)(prev_Layer)
        x2 = layers.BatchNormalization(axis=2)(x2)
        
        x = layers.concatenate([x , x2], axis=-1)
        x = layers.Activation(activation='relu')(x)
        return x
    

    def create_model(self, df, candles, split):
        
        X, Y = self.fetch_candlestick_data_with_indicators(df, candles)

        X_train, X_test, Y_train, Y_test = self.prepare_learning_data(X, Y, split)

        print('Start creating the model!')

        model = Sequential()

        input_layer = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

        #Stage 1
        x = layers.ZeroPadding1D(3)(input_layer)
        x = layers.Conv1D(filters = 64, kernel_size = 7, strides=2) (x)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.MaxPool1D(pool_size=3 , strides=1)(x)

        #Stage 2
        x = self.ConvBlock(prev_Layer=x, filters = [64 , 64 , 256], strides = 1)
        x = self.IdentityBlock(prev_Layer=x, filters = [64,64,256])
        x = self.IdentityBlock(prev_Layer=x, filters = [64,64,256])

        #Stage 3
        x = self.ConvBlock(prev_Layer=x, filters = [128 , 128 , 512], strides = 2)
        x = self.IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])
        x = self.IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])
        x = self.IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])
        
        '''
        #Stage 1
        x = layers.ZeroPadding1D(3)(input_layer)
        x = layers.Conv1D(filters = 64, kernel_size = 7, strides=2) (x)
        x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.MaxPool1D(pool_size=3 , strides=1)(x)

        #Stage 2
        x = self.ConvBlock(prev_Layer=input_layer, filters = [64 , 64 , 256], strides = 1)
        x = self.IdentityBlock(prev_Layer=x, filters = [64,64,256])
        x = self.IdentityBlock(prev_Layer=x, filters = [64,64,256])
        
        #Stage 3
        x = self.ConvBlock(prev_Layer=x, filters = [128 , 128 , 512], strides = 2)
        x = self.IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])
        x = self.IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])
        x = self.IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])

        #Stage 4    
        x = self.ConvBlock(prev_Layer=x, filters = [256 , 256 , 1024], strides = 2)    
        x = self.IdentityBlock(prev_Layer=x, filters = [256 , 265 , 1024])
        x = self.IdentityBlock(prev_Layer=x, filters = [256 , 265 , 1024])
        x = self.IdentityBlock(prev_Layer=x, filters = [256 , 265 , 1024])
        x = self.IdentityBlock(prev_Layer=x, filters = [256 , 265 , 1024])
        x = self.IdentityBlock(prev_Layer=x, filters = [256 , 265 , 1024])
        
        #Stage 5
        x = self.ConvBlock(prev_Layer=x, filters = [512 , 512 , 2048], strides = 2)
        x = self.IdentityBlock(prev_Layer=x, filters = [512 , 512 , 2048])
        x = self.IdentityBlock(prev_Layer=x, filters = [512 , 512 , 2048])
        
        #Stage 6
        x = layers.AveragePooling1D(pool_size=2)(x)
        '''
        #x = layers.GlobalAveragePooling1D()(x)
        #x = layers.RepeatVector(10)(x)
        
        #x = layers.LSTM(128, return_sequences=False, dropout=0.2)(x)
        #x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        out = layers.Dense(units=1, activation='linear')(x)

        model = Model(inputs=input_layer, outputs=out)

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
            loss='mean_squared_error', 
            metrics=['mae'])

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=10, min_lr=1e-6)

        history = model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),  
            epochs=100,  
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],  
            verbose=1
        )


        if split != 1:
            predictions = model.predict(X_test)
            predictions = predictions.reshape(-1)

            mse = mean_squared_error(Y_test, predictions)
            mae = mean_absolute_error(Y_test, predictions)

            print(f"Mean Squared Error: {mse}")
            print(f"Mean Absolute Error: {mae}")

        loss = history.history['loss'] 
        val_loss = history.history['val_loss']  

        plt.plot(loss, label='Tanulás veszteség')
        plt.plot(val_loss, label='Validációs veszteség')
        plt.xlabel('Epoch')
        plt.ylabel('Veszteség')
        plt.legend()
        plt.show()

        #model.summary()

        return model
    
