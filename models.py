from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Dense, Dropout, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from getter import prepair_test
import numpy as np

#Main

def lstmgruconv(X_train, X_test, y_train, y_test):
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 3)))
    #model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=True))
    for _ in range(13):
        model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    #model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=10, callbacks=[early_stopping])
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    

    
def lstm(X_train, X_test, y_train, y_test):

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 3)))
    for _ in range(13):
        model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))  
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    
    early_stopping = EarlyStopping(monitor='loss', patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=50, batch_size=10, callbacks=[early_stopping])
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')



def check1(X_train, X_test, y_train, y_test):
    
    model = Sequential([
       Conv1D(64, kernel_size=3, activation='relu', input_shape=(60, 3)),
        Dropout(0.1),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        #Reshape((30, 128)),  
        #LSTM(64, return_sequences=True),
        #LSTM(32),
        #GRU(64, return_sequences=True),
        #Flatten(),
        # GRU(32),
        #LSTM(32),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
        ])
    early_stopping = EarlyStopping(monitor='loss', patience=10)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=50, batch_size=10, callbacks=[early_stopping])
    
    class_names = ["0", "1", "2", "3","4", "5", "6", "7","8", "9"]
    predictions = model.predict(prepair_test())
    predicted_classes = np.argmax(predictions, axis=1)

    predicted_class_names = [class_names[idx] for idx in predicted_classes]
    
    print("Я считаю, твой файл - это:",predicted_class_names)
    #print("Я считаю, твой файл - это:",model.predict(prepair_test()))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    