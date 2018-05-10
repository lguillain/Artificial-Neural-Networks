import pickle
import keras
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Masking, TimeDistributed, Dense, Concatenate, Dropout, LSTM, GRU, SimpleRNN
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

maxlen = 393
P_matrix = np.load('/data/P_matrix.npy')
T_matrix = np.load('/data/T_matrix.npy')
with open('/data/dictionaries.pkl', 'rb') as f:
    dictionaries = pickle.load(f)

def buildModel(dictionaries, batch_length, dropout=0.2, activation='GRU', Hsize=128):
    X = dict()
    H = dict()
    M = dict()
    Y = dict()
    
    X['T'] = Input(shape=(batch_length, len(dictionaries['T'])), name="XT")
    X['P'] = Input(shape=(batch_length, len(dictionaries['P'])), name="XP")
    
    M['T'] = Masking(mask_value=0., name="MT")(X['T'])
    M['P'] = Masking(mask_value=0., name="MP")(X['P'])
    
    H['1'] = Concatenate(name="MergeX")([M['T'], M['P']])
    if activation == 'GRU':
        #Your hidden layer(s) architecture with GRU
        H['2'] = GRU(Hsize, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(H['1'])
    elif activation == 'LSTM':
        #Your hidden layer(s) architecture with LSTM (For your own curiosity, not required for the project)
        H['2'] = LSTM(Hsize, recurrent_dropout=0.25, return_sequences=True)(H['1'])
    elif activation == 'RNN':
        #Your hidden layer(s) architecture with SimpleRNN
        H['2'] = SimpleRNN(Hsize, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(H['1'])
    
    Y['T'] = TimeDistributed(Dense(len(dictionaries['T']), activation='softmax'), name='YT')(H['2'])
    Y['P'] = TimeDistributed(Dense(len(dictionaries['P']), activation='softmax'), name='YP')(H['2'])
    
    model = Model(inputs = [X['T'], X['P']], outputs = [Y['T'], Y['P']])
    opt = Adam() 
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=opt,
        metrics=['acc'])

    return model

RNNmodel = buildModel(dictionaries, 
                      batch_length=maxlen-1, #Put here the number of notes (timesteps) you have in your Zero-padded matrices
                      activation='RNN')
#with open('/data/weights_0.pkl', 'rb') as f:
#    RNNmodel.set_weights(pickle.load(f))


# input = first element to n-1
XP = np.array(list(map(lambda x: x[:-1], P_matrix)))
XT = np.array(list(map(lambda x: x[:-1], T_matrix)))

# output = second element to last element
YP = np.array(list(map(lambda x: x[1:], P_matrix)))
YT = np.array(list(map(lambda x: x[1:], T_matrix)))

epochs = 50
history = []

for i in range(5):
    history = RNNmodel.fit([XT, XP], [YT, YP], batch_size=128, \
                           epochs=epochs, validation_split=0.2, verbose=2) #validation split
    with open('/output/history_{}.pkl'.format(i), 'wb') as f:
        pickle.dump(history.history, f)
    with open('/output/weights_{}.pkl'.format(i), 'wb') as f:
        pickle.dump(RNNmodel.get_weights(), f)

