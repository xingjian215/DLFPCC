import keras
import os
from keras.layers import Dense, Input, Dropout
from keras.models import Model
import data_process
import numpy as np
from keras.models import Sequential
import metrics_class
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop

batch_size = 512
num_classes = 2
epochs = 500
layers=5
#optimizer = SGD(lr=0.001,decay=0.0,momentum=0.9,nesterov=True)
#optimizer = Adam(lr=0.001, decay=1e-6)
#optimizer = RMSprop(lr=0.0001, decay=1e-6)

def get_intermediate_output(model, data_in, n_layer, train, dtype=np.float32):
    # 返回指定层的输出，作为下一个DAE的输入
    data_out = K.function([model.layers[0].input, K.learning_phase()], [model.layers[n_layer].output])
    data_out = data_out([data_in, train])[0]
    return data_out.astype(dtype, copy=False)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_process.compose_data()
x_train = (x_train.reshape(x_train.shape[0], x_train.shape[1])).astype('float32')
x_val = (x_val.reshape(x_val.shape[0], x_train.shape[1])).astype('float32')
x_test = (x_test.reshape(x_test.shape[0], x_train.shape[1])).astype('float32')

x_train_copy = x_train
x_val_copy = x_val

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

model_layers = [None]*layers
encoders = []
for i in range(layers):
    input_data = Input(shape=(x_train_copy.shape[1],))
    dropout_layer = Dropout(0.1)
    in_dropout = dropout_layer(input_data)
    encoder_layer = Dense(output_dim =20, activation='sigmoid')
    encoded = encoder_layer(in_dropout)
    n_out = x_train_copy.shape[1]
    decoder_layer = Dense(output_dim =n_out, activation='linear')
    decoded = decoder_layer(encoded)

    model = Model(input=input_data, output=decoded)
    model.compile(optimizer='rmsprop', loss='mse')

    model.fit(x_train_copy, x_train_copy, epochs=epochs, batch_size=batch_size,  shuffle=True, validation_data=(x_val_copy, x_val_copy))

    model_layers[i] = model
    encoders.append(model.layers[2])
    x_train_copy = get_intermediate_output(model, x_train_copy, n_layer = 2, train=0)
    x_val_copy = get_intermediate_output(model,x_val_copy, n_layer = 2, train=0)




final_model = Sequential()
input0 = Dropout(0.1, input_shape = (encoders[0].input_shape[1],))
final_model.add(input0)
for i in range(len(encoders)):
    encoders[i].inbound_nodes = []
    final_model.add(encoders[i])


output = Dense(num_classes, activation='softmax')
final_model.add(output)
final_model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop')
final_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), shuffle=True)


if x_train.shape[0]==12000:
    temp_str='SC1'
elif x_train.shape[0]==66000:
    temp_str = 'SC10'
elif x_train.shape[0] == 606000:
    temp_str = 'SC100'
else:
    temp_str = 'SC200'
save_dir = os.path.join(os.getcwd(), 'saved_sdae_models')
model_name = 'keras_sdae_trained_model_'+temp_str+'.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
final_model.save(model_path)
print('Saved trained model at %s ' % model_path)


TP,FP,TN,FN = metrics_class.predict(final_model, x_test,y_test)
print('TP:',TP,',FN:',FN,',TN:',TN,',FP:',FP)
print('all:',TP+FP+TN+FN,',p:',TP+FN,',N:',TN+FP)
print('acc=',(TP+TN)/(TP+FP+TN+FN))
print('TPR=',TP/(TP+FN))
print('FPR=',FP/(FP+TN))
print('precision=',TP/(TP+FP))