# Tin Jukić, seminar 1
# 3-Tournament Genetic Algorithm (GA)

import numpy as np
from Utils import *
from functools import cmp_to_key


class GeneticAlgorithm:
    """
    3-Tournament Genetic Algorithm for solving Vehicle Routing Problem\n
    Uses VRP UCS to determine the quality of the generated solution\n
    Author: Tin Jukić
    """

    def __init__(self, population_size=10, num_of_iterations=2000):
        """
        Constructor for the Genetic Algorithm
        :param population_size: size of the population
        :param num_of_iterations: number of iterations
        """
        self._weights = []  # weights population; every weight has 3 genes; sorted: best to worse
        self._population_size = population_size  # number of solutions created for every instance if genetic algorithm
        self._num_of_iterations = num_of_iterations  # number of iterations before stopping the algorithm
        self._previous_generation_score = 0.0  # the best score of the previous generation
        self._iteration = 0  # number of the current iteration

    def start(self) -> tuple:
        """
        Starts the genetic algorithm
        :return: the best solution weights and score
        """
        self._fit()
        return self._the_end()

    def _fit(self):
        """
        Optimizes weights to get the best solution
        Optimization is stopped when there is not a huge improvement made to the score
        or when the maximum number of iterations is reached
        """
        self._generate_random_weights()

        for iteration in range(1, self._num_of_iterations + 1):
            self._iteration = iteration
            print(self._iteration)

            _previous_score = self._previous_generation_score
            for i in range(self._population_size):
                self._weights[i].update_score(score=self._evaluate(weights=self._weights[i].get_weights()))
            # self._weights.sort()  # weights are now sorted
            self._weights = sorted(self._weights, key=cmp_to_key(self._WeightsAndScore.comparator))  # weights are now sorted

            self._previous_generation_score = self._weights[0].get_score()

            # if abs(_previous_score - self._weights[0].get_score()) < pow(10, -10):
            #     break  # improvement too low

            # improvement not too low -> continue with the algorithm
            # 3 phases of Genetic Algorithm
            _best_weights = self._selection()
            _new_weights = self._crossover(best_weights=_best_weights)
            self._weights = self._weights[:self._population_size - len(_new_weights)] + _new_weights[:]
            self._mutation()

    def _selection(self) -> []:
        """
        Selects 3 best weights to go to the crossover section
        :return: 3 best weights (3 Tournament Genetic Algorithm)
        """
        _best_weights = []
        for i in range(3):
            _best_weights.append(self._weights[i])
        return _best_weights

    def _crossover(self, best_weights: []):
        """
        :param best_weights: weights that were selected in the previous phase
        :return:
        """
        return [
            self._WeightsAndScore(weights=best_weights[0].get_weights() + best_weights[1].get_weights() / 2),
            self._WeightsAndScore(weights=best_weights[0].get_weights() + best_weights[2].get_weights() / 2),
            self._WeightsAndScore(weights=best_weights[1].get_weights() + best_weights[2].get_weights() / 2),
            self._WeightsAndScore(weights=best_weights[0].get_weights() + best_weights[1].get_weights() +
                                  best_weights[2].get_weights() / 3)
        ]

    def _mutation(self):
        """
        Mutates every gene by adding small value to it
        """
        _mutation_value = 0.00001

        for i in range(self._population_size):
            _new_weights = self._weights[i].get_weights()
            for j in range(2):
                if np.random.random() > 0.5:
                    _new_weights[j] += _mutation_value
            self._weights[i].update_weights(_new_weights)

    def _the_end(self) -> tuple:
        """
        Prints the best solution of the VRP
        :return: the best solution weights and score
        """
        result = self.best_solution_result()
        print(f"\n\n\nBest solution for VRP:\n\tweights = {result[0]}\n\tevaluation = {result[1]}\n")
        return result

    def _generate_random_weights(self):
        """
        Randomly generates initial weights for the genetic algorithm
        """
        for i in range(self._population_size):
            self._weights.append(self._WeightsAndScore(weights=np.random.random(size=2), score=0.0))

    def _evaluate(self, weights: []) -> float:
        """
        Calculates the score of the given solution (grades the solution)
        :param weights: weights of the graded solution
        :return: calculated score of the given solution
        """
        evaluation = 0.0

        data = createDataModel()
        vehicles = []
        for i in range(data["numOfVehicles"]):
            vehicles.append(Vehicle())

        while True:
            finished = False

            for i in range(len(vehicles)):
                if self._iteration % 1000 == 0:
                    print(f"Trenutno sam na autu broj: {i + 1}")

                vehicle = vehicles[i]
                currentNodeIndex = data["depotIndex"]  # svaki auto krece od depota
                currentNode = data["coordinates"][currentNodeIndex]

                if len(data["visitedNodes"]) == (data["numOfLocations"] - 1):
                    # obisao si sve cvorove ili si potrosio sve aute...
                    finished = True
                    break

                result: dict = findNodeToVisitGA(vehicle=vehicle, currentNodeIndex=currentNodeIndex,
                                                 data=data, weights=weights)
                nodeToVisitIndex = result["minDistanceNodeIndex"]
                evaluation += result["evaluation"]
                while nodeToVisitIndex != -1:
                    currentNodeIndex = nodeToVisitIndex
                    result: dict = findNodeToVisitGA(vehicle=vehicle, currentNodeIndex=currentNodeIndex,
                                                     data=data, weights=weights)
                    nodeToVisitIndex = result["minDistanceNodeIndex"]
                    evaluation += result["evaluation"]

                # gotov si s tim autom
                if self._iteration % 1000 == 0:
                    print("Niti jedan cvor ne mozes obici, odi u depot")
                vehicle.capacity -= calculateDistance(currentNode, data["coordinates"][data["depotIndex"]])
                vehicle.traveledDistance += calculateDistance(currentNode, data["coordinates"][data["depotIndex"]])

                # iskoristio si sve aute koje imas na raspolaganju
                if i == data["numOfVehicles"] - 1 and not finished:
                    finished = True

                if self._iteration % 1000 == 0:
                    print()

            if finished:
                if self._iteration % 1000 == 0:
                    # ispisi put za svaki auto
                    vehicleNum = 0
                    for vehicle in vehicles:
                        vehicleNum += 1
                        print(f"Vehicle number: {vehicleNum}:")
                        print(f"Traveled distance: {vehicle.traveledDistance}")
                        print(f"Remaining capacity: {vehicle.capacity}")
                        print(f"Used capacity: {vehicle.usedCapacity}")
                        print(f"Visited customers: {len(vehicle.path)}")
                        print("Route", end=": ")
                        for p in vehicle.path:
                            if vehicle.path[len(vehicle.path) - 1] == p:
                                print(p)
                            else:
                                print(p, end=" -> ")
                        print()
                        if len(vehicle.path) == 0:
                            print()
                break
            else:
                print("Something went wrong and GA algorithm for VRP did not finish!")
                break

        return evaluation

    def best_solution_result(self) -> tuple:
        """
        Gives final vehicle routes calculated by genetic algorithm
        :return: weights used for the solution calculation and evaluation of the used solution (tuple)
        """
        weights = self._weights[0].get_weights()  # using the best weights
        return weights, self._evaluate(weights=weights)

    def get_population_size(self):
        """
        :return: population size of the genetic algorithm
        """
        return self._population_size

    def __str__(self):
        """
        To string method
        :return: current weights of the genetic algorithm
        """

        _text = "Weights: "
        for i in range(len(self._weights)):
            _text += str(self._weights[i])
            if i != len(self._weights) - 1:
                _text += ", "
        return _text

    class _WeightsAndScore:
        """
        Private inner class used to sort solutions
        """

        def __init__(self, weights: [], score=0.0):
            self._weights = weights
            self._score = score

        def __eq__(self, other):
            if self._score > other.get_score():
                return 1
            else:
                return -1

        @staticmethod
        def comparator(value1, value2):
            return value1.get_score() - value2.get_score()

        def get_weights(self) -> []:
            """
            :return: weights of the solution
            """
            return self._weights

        def update_weights(self, weights: []):
            """
            Updates the weights of the solution
            :param weights: new weights of the solution
            """
            self._weights = weights

        def get_score(self):
            """
            :return: the score of the solution
            """
            return self._score

        def update_score(self, score: float):
            """
            :param score: calculated score
            """
            self._score = score
