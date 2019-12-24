import keras
import os
import data_process
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD, Adam, RMSprop
import metrics_class

batch_size = 512
num_classes = 2
epochs = 300
conv_kernel_size = 3
#optimizer = SGD(lr=0.001,decay=0.0,momentum=0.9,nesterov=True)
optimizer = Adam(lr=0.001, decay=1e-6)
#optimizer = RMSprop(lr=0.0001, decay=1e-6)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_process.compose_data()
x_train = (x_train.reshape(x_train.shape[0], x_train.shape[1],1)).astype('float32')
x_val = (x_val.reshape(x_val.shape[0], x_val.shape[1],1)).astype('float32')
x_test = (x_test.reshape(x_test.shape[0], x_test.shape[1],1)).astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

#build model
model = Sequential()
model.add(LSTM(units=64, input_shape=(x_train.shape[1], 1),return_sequences=True,activation='sigmoid',dropout=0.2))
model.add(LSTM(units=64,return_sequences=False,activation='sigmoid',dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=batch_size, verbose=1)
model.summary()

if x_train.shape[0]==12000:
    temp_str='SC1'
elif x_train.shape[0]==66000:
    temp_str = 'SC10'
elif x_train.shape[0] == 606000:
    temp_str = 'SC100'
else:
    temp_str = 'SC200'
save_dir = os.path.join(os.getcwd(), 'saved_lstm_models')
model_name = 'keras_lstm_trained_model_'+temp_str+'.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

TP,FP,TN,FN = metrics_class.predict(model, x_test,y_test)
print('TP:',TP,',FN:',FN,',TN:',TN,',FP:',FP)
print('all:',TP+FP+TN+FN,',p:',TP+FN,',N:',TN+FP)
print('acc=',(TP+TN)/(TP+FP+TN+FN))
print('TPR=',TP/(TP+FN))
print('FPR=',FP/(FP+TN))
print('precision=',TP/(TP+FP))