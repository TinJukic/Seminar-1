# Tin Jukić, seminar 1
# Solving Vehicle Routing Problem (VRP) using Genetic Algorithm (GA)
# GA -> 3-tournament selection

from Utils import *
import GeneticAlgorithm


def _VRP_GA_one():
    """
    Method which starts genetic algorithm
    """
    pass


def main():
    print("Solving Vehicle Routing Problem (VRP) using Genetic Algorithm (GA)")
    print()

    _VRP_GA_one()

    ga: GeneticAlgorithm.GeneticAlgorithm = GeneticAlgorithm.GeneticAlgorithm()
    ga.start()


if __name__ == '__main__':
    main()
