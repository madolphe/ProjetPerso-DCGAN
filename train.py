from DCGAN.discriminator import Discriminator
from DCGAN.generator import Generator
from DCGAN.adversarialNetwork import AdversarialModel
import data.get_data_MNIST as dataset

if __name__ == '__main__':
    print(dataset.X_train.shape)
    disc = Discriminator(verbose=False)
    gen = Generator(7, 1, verbose=False)
    AM = AdversarialModel(disc, gen)
    AM.train(dataset.X_train, 12, 100, 1000)

#TODO ajout logger pour suivi apprentissage
#TODO comparer les loss utilisées
#TODO implémenter l'architecture du réseau initial
#TODO save model
