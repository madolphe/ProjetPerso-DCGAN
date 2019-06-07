from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Conv2DTranspose, Dense, Input, Dropout, Reshape, UpSampling2D, Activation, ReLU, Conv2D
from keras.optimizers import RMSprop


class Generator:
    def __init__(self, start_shape, depth, dropout=0.4, verbose=False, lr=0.0004):
        self.latent_start = 100
        model = Sequential()
        model.add(Dense(start_shape*start_shape*depth, input_shape=(self.latent_start,)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        model.add(Reshape((start_shape, start_shape, depth)))
        model.add(Dropout(dropout))

        model.add(UpSampling2D())
        model.add(Conv2D(512, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())

        model.add(UpSampling2D())
        model.add(Conv2DTranspose(256, 3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())

        model.add(Conv2D(128,3, padding='same'))
        model.add(Conv2D(1, 3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())
        if verbose:
            model.summary()

        noise = Input(shape=(self.latent_start,))
        img = model(noise)
        self.model = Model(noise, img)

        self.lr = lr
        self.optimizer = RMSprop(lr=self.lr, clipvalue=1, decay=3 * 10 ^ (-8))

