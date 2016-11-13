from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy
numpy.random.seed(7)

X_data = numpy.loadtxt('TRAIN_ON_THIS.csv', delimiter=',',skiprows=0)
X_data = X_data.astype('float32')
#X_test = numpy.loadtxt('testseta.csv', delimiter=',', skiprows=0)

nb_examples = X_data.shape[0]

X_train = X_data[0:nb_examples-2000, 0:7]
X_cv = X_data[nb_examples-2000:nb_examples, 0:7]
#X_test = X_test[:,0:7]

y_train = X_data[0:nb_examples-2000, 7]
y_cv = X_data[nb_examples-2000:nb_examples, 7]

#nb_tests = X_test.shape[0]

model = Sequential()

model.add(Dense(500, input_dim = 7, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(250, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, init='normal', activation='relu'))
#model=load_model('model.save')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(X_train, y_train,nb_epoch=10, batch_size=128, validation_data = (X_cv, y_cv))

model.save('model.save')
