import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Flatten

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=12, fc2_dims=12):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.flatten = Flatten()
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=12, fc2_dims=12):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.flatten = Flatten()
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        q = self.q(x)

        return q
