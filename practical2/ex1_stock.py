import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_float", random.random)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def prediction(coeffs, row):
    dotproduct = sum(c * r for c, r in zip(coeffs, row))
    return dotproduct / sum(coeffs)


trainingData = np.genfromtxt('training.csv', delimiter=",")


def eval_mean_squ_diff(individual):
    total_squ_diff = sum((prediction(individual, row) - row[4]) ** 2 for row in trainingData)
    return total_squ_diff / len(trainingData),


toolbox.register("evaluate", eval_mean_squ_diff)
toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.25)
toolbox.register("select", tools.selTournament, tournsize=3)

NGEN = 200
CXPB = 0.6
MUTPB = 0.2

random.seed(64)
pop = toolbox.population(n=300)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

meanFitness = []

# Begin the evolution
for g in range(NGEN):
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2, 0.5)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    # fits = [ind.fitness.values[0] for ind in pop]

    # length = len(pop)
    # mean = sum(fits) / length
    # sum2 = sum(x * x for x in fits)
    # std = abs(sum2 / length - mean ** 2) ** 0.5
    #
    # print("  Min %s" % min(fits))
    # print("  Max %s" % max(fits))
    # print("  Avg %s" % mean)
    # print("  Std %s" % std)

    # meanFitness.append(mean)

best = tools.selBest(pop, 1)[0]
total_error = 0

testData = np.genfromtxt('testing.csv', delimiter=",")
for test in testData:
    pred = prediction(best, test)

    error = (test[4] - pred) ** 2
    total_error += error

print(total_error / len(testData))

# x = range(1, NGEN + 1)
# plt.plot(x, meanFitness)
# plt.ylabel('Fitness')
# plt.xlabel('Generation')
# plt.show()
