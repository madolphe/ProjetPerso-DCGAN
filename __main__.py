from DCGAN.discriminator import Discriminator
from DCGAN.generator import Generator
from DCGAN.adversarialNetwork import AdversarialModel
import data.get_data as dataset

if __name__ == '__main__':
    disc = Discriminator(verbose=False)
    gen = Generator(7, 1, verbose=False)
    AM = AdversarialModel(disc, gen)
    AM.train(dataset.X, 12, 100, 100)

#TODO ajout logger pour suivi apprentissage
#TODO comparer les loss utilisées
#TODO implémenter l'architecture du réseau initial
#TODO save model
