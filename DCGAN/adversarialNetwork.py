import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input


class AdversarialModel:
    def __init__(self, discriminator, generator, img_rows=28, img_cols=28, channels=1,
                 latent_dim=100):
        """
        Constructeur intialisant les optimizers et la perte

        """
        # Chargement du jeu de données et paramétrage des images à traiter:
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        # Initialisation du discriminateur:
        self.discriminator = discriminator.model
        self.discriminator.compile(loss='binary_crossentropy', optimizer=discriminator.optimizer, metrics=['accuracy'])
        self.discriminator.summary()

        # Initialisation du générateur:
        self.generator = generator.model
        z = Input(shape=(self.latent_dim,))
        img_gen = self.generator(z)
        # Pour le modèle combiné on entraîne seulement le generateur:
        self.discriminator.trainable = False
        out = self.discriminator(img_gen)
        self.AM = Model(z, out)
        self.AM.compile(loss='binary_crossentropy', optimizer=generator.optimizer, metrics=['accuracy'])
        self.AM.summary()

    def train(self, dataset, batch_size, save_interval, epochs):
        """
        dataset:
        batch_size:
        save_intervals:
        epochs:
        """
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # ---------------------
            #  Entrainement discriminateur
            # ---------------------
            # Choisisons au hasard dans le jeu de données "batch_size" images
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            imgs = np.expand_dims(dataset[idx], axis=4)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Entrainement du generateur
            # ---------------------
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.AM.train_on_batch(noise, valid)

            # Plot the progress
            print("{} [D loss: {}, acc.: {}] [G loss: {}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss))

            # On garde une visualisation pour chaque epoch
            #if epoch % save_interval == 0:
            #    self.visulazisation(epoch, 3, 3)

    def visulazisation(self, epoch, row, column):
        """Sauver et voir l'apprentissage"""
        noise = np.random.normal(0, 1, (row * column, self.latent_dim))
        gen_imgs = self.std_to_img(self.generator.model.predict(noise))
        fig, axs = plt.subplots(row, column)
        cnt = 0
        for i in range(row):
            for j in range(column):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./output/imgs/mnist_%d.png" % epoch)
        plt.close()

    @classmethod
    def std_to_img(cls, img):
        """Pour le moment pas de grandes fonctions de pre-processing, juste une manière de standardiser"""
        img = img/255 - 0.5
        return img.astype(np.float32)

    @classmethod
    def img_to_std(cls, img):
        """Fonction inverse"""
        img = img*255
        return img.astype(np.float32)





