import random
import numpy
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 18)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    n = 3
    blocks = [individual[i:i + n] for i in range(0, len(individual), n)]

    def calclulateFitness(block):
        if sum(block) == 0:
            return 0.9
        elif sum(block) == 1:
            return 0.8
        elif sum(block) == 2:
            return 0.0
        elif sum(block) == 3:
            return 1.0

    return sum(map(calclulateFitness, blocks)) / len(blocks),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)

NGEN = 200
CXPB = 0.6
MUTPB = 0.2

# def main():
random.seed(64)
pop = toolbox.population(n=300)

logbook = tools.Logbook()

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

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

    record = stats.compile(pop)
    logbook.record(gen=g, **record)

gen = logbook.select("gen")
fit_mins = logbook.select("min")

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

lns = line1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()


