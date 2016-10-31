import random

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def evaluteInd(individual):
    a = sum(individual)
    b = len(individual)
    return a, 1. / b

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluteInd)

IND_SIZE = 5

toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)

pop = [toolbox.individual(), toolbox.individual(), toolbox.individual()]

print(pop)

NGEN = 10
CXPB = 0.5
MUTPB = 0.2
for g in range(NGEN):
    offspring = toolbox.select(pop, len(pop))
    offspring = [toolbox.clone(i) for i in offspring]

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # replace completely
    pop[:] = offspring

    best = tools.selBest(pop, 1)
    print(evaluteInd(best))
