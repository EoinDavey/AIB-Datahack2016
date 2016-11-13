from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy
numpy.random.seed(7)

#X_data = numpy.loadtxt('fiftySet.csv', delimiter=',',skiprows=1)
#X_data = X_data.astype('float32')
X_test = numpy.loadtxt('FINAL_VALUES.csv', delimiter=',', skiprows=0)

#nb_examples = X_data.shape[0]

#X_train = X_data[0:nb_examples-2000, 0:7]
#X_cv = X_data[nb_examples-2000:nb_examples, 0:7]
X_test = X_test[:,0:7]

#y_train = X_data[0:nb_examples-2000, 7]
#y_cv = X_data[nb_examples-2000:nb_examples, 7]

nb_tests = X_test.shape[0]

#model = Sequential()

#model.add(Dense(500, input_dim = 7, init='normal', activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(250, init='normal', activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(1, init='normal', activation='relu'))
model=load_model('model.save')

#model.compile(loss='mse', optimizer='adam', metrics=['mse'])

#model.fit(X_train, y_train,nb_epoch=10, batch_size=128, validation_data = (X_cv, y_cv))

score = model.predict(X_test)

grades = []
for i in xrange(0,nb_tests):
    p = score[i]
    ans = ""
    if p > 450:
        ans = "G"
    elif p>380:
        ans="F"
    elif p > 340:
        ans = "E2"
    elif p> 300:
        ans = "E1"
    elif p>260:
        ans = "D1"
    elif p > 225:
        ans = "D1"
    elif p > 200:
        ans = "C3"
    elif p > 175:
        ans = "C2"
    elif p > 150:
        ans = "C1"
    elif p > 125:
        ans = "B3"
    elif p > 100:
        ans = "B2"
    elif p > 75:
        ans = "B1"
    elif p > 50:
        ans = "A3"
    elif p > 25:
        ans = "A2"
    else:
        ans = "A1"
    grades.append(ans)

numpy.savetxt('final_prediction.csv', grades, fmt="%s")
