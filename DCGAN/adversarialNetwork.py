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
        # Load the dataset
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        self.discriminator = discriminator
        self.discriminator.model.compile(loss='binary_crossentropy',
                                         optimizer=self.discriminator.optimizer,
                                         metrics=['accuracy'])
        self.generator = generator

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img_gen = self.generator.model(z)
        # Pour le modèle combiné on entraîne seulement le gen

        self.discriminator.model.trainable = False
        out = self.discriminator.model(img_gen)
        self.AM = Model(z, out)
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=self.generator.optimizer,
                        metrics=['accuracy'])

    def train(self, dataset, batch_size, save_interval, epochs):
        """

        """
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Choisisons au hasard dans le jeu de données "batch_size" images
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            imgs = dataset[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.generator.latent_start))
            gen_imgs = self.generator.model.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.model.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.model.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.AM.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.visulazisation(epoch, 3, 3)

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





