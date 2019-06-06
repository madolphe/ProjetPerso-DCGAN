from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Input
from keras.optimizers import RMSprop


class Discriminator:
    def __init__(self, shape=(28, 28, 1), depth=64, dropout=0.4, verbose=False, lr=0.0008, trainable=True):
        """Build discriminator with default parameters"""
        model = Sequential()

        model.add(Conv2D(depth, strides=2, kernel_size=5, padding='same',
                         name='conv_1', input_shape=shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth * 2, strides=2, kernel_size=5, padding='same',
                         name='conv_2'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth * 4, strides=2, kernel_size=5, padding='same',
                         name='conv_3'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth * 8, strides=2, kernel_size=5, padding='same',
                         name='conv_4'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid', name='dense'))
        model.trainable = trainable

        if verbose:
            model.summary()

        img = Input(shape=shape)
        out = model(img)
        self.model = Model(img, out)

        self.lr = lr

        self.optimizer = RMSprop(lr=self.lr, clipvalue=1, decay=6 * 10 ^ (-8))
