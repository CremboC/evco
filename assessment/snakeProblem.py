# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
from functools import partial

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
TOTAL_SIZE = (YSIZE - 2) * (XSIZE - 2)
NFOOD = 1  # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)

locMap = {
    S_DOWN: lambda y, x: (y + 1, x),
    S_UP: lambda y, x: (y - 1, x),
    S_RIGHT: lambda y, x: (y, x + 1),
    S_LEFT: lambda y, x: (y, x - 1)
}

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2):
    return partial(progn, out1, out2)

def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

def is_wall(coord):
    [y, x] = coord
    return y in [0, YSIZE - 1] or x in [0, XSIZE - 1]

# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1), self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    def getAhead2Location(self):
        self.getAheadLocation()
        y, x = self.ahead
        return locMap[self.direction](y, x)

    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE - 1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return self.hit

    ## You are free to define more sensing options to the snake

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return is_wall(self.ahead)

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    def sense_danger_ahead(self):
        return self.sense_tail_ahead() or self.sense_wall_ahead()

    def sense_danger_2_ahead(self):
        ahead2 = self.getAhead2Location()
        return ahead2 in self.body or is_wall(ahead2)

    def sense_food(self, cond):
        if len(self.food) == 0: False
        first_food = self.food[0]
        head = self.body[0]
        return cond(first_food, head)

    def sense_food_up(self):
        return self.sense_food(lambda food, head: head[0] > food[0])

    def sense_food_down(self):
        return self.sense_food(lambda food, head: head[0] < food[0])

    def sense_food_left(self):
        return self.sense_food(lambda food, head: head[1] < food[1])

    def sense_food_right(self):
        return self.sense_food(lambda food, head: head[1] > food[1])

    def moves_in_direction(self, direction):
        return self.direction == direction

    def danger_in_direction(self, direction):
        [y, x] = self.body[0]
        loc = locMap[direction](y, x)
        return loc in snake.body or is_wall(loc)

    def danger_2_in_direction(self, direction):
        [y, x] = self.body[0]
        [y1, x1] = locMap[direction](y, x)
        loc = locMap[direction](y1, x1)
        return loc in snake.body or is_wall(loc) 

def generatePossibleFoodLocations():
    possibleFoodLocations = []
    for y in range(1, YSIZE - 1):
        for x in range(1, XSIZE - 1):
            possibleFoodLocations.append([y, x])

    return possibleFoodLocations

# This function places a food item in the environment
def placeFood(snake):
    food = []
    possibleFoodLocations = generatePossibleFoodLocations()
    while len(food) < NFOOD:
        if len(possibleFoodLocations) == 0:
            return None
        potentialfood = [random.randint(1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        possibleFoodLocations = filter(lambda x: x != potentialfood, possibleFoodLocations)
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)

    snake.food = food  # let the snake know where the food is
    return food


snake = SnakePlayer()


# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    snake._reset()
    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], '@')

    timer = 0
    collided = False
    while not collided and not timer == ((2 * XSIZE) * YSIZE):

        # Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
        win.getch()

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##

        routine()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
            food = placeFood(snake)
            for f in food: win.addch(f[0], f[1], '@')
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1  # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], 'o')

        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2 * XSIZE) * YSIZE))

    curses.endwin()

    print collided
    print hitBounds
    print snake.score

    return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(individual, compiled=False):
    global snake

    if not compiled:
        func = toolbox.compile(expr=individual)
    else:
        func = individual

    totalScore = 0

    snake._reset()
    food = placeFood(snake)
    timer = 0

    locations = set()
    timesFinished = 0
    visited = 0
    timeSurvived = 0
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:
        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        func()

        snake.updatePosition()

        loc = "{},{}".format(snake.body[0][0], snake.body[0][1])
        if loc not in locations:
            locations.add(loc)
            visited += 1

        if len(locations) == TOTAL_SIZE:
            timesFinished += 1
            locations = set()

        if snake.body[0] in food:
            snake.score += 1
            food = placeFood(snake)
            if food is None:
                # found perfect snake. inability to place food means the grid is filled by the snake
                return totalScore, snake.score, 99999, timeSurvived
            timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten

        timeSurvived += 1
        totalScore += snake.score

    return totalScore, snake.score, (timesFinished * TOTAL_SIZE + visited), timeSurvived

pset = gp.PrimitiveSet("MAIN", 0)
pset.addTerminal(snake.changeDirectionRight, name="right")
pset.addTerminal(snake.changeDirectionDown, name="down")
pset.addTerminal(snake.changeDirectionLeft, name="left")
pset.addTerminal(snake.changeDirectionUp, name="up")
pset.addTerminal(lambda: True, name="forward")

pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)

pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_danger_ahead, out1, out2), 2, name="if_danger_ahead")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_danger_2_ahead, out1, out2), 2, name="if_danger_2_ahead")

pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_DOWN), out1, out2), 2, name="if_danger_down")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_2_in_direction, S_DOWN), out1, out2), 2, name="if_danger_2_down")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_UP), out1, out2), 2, name="if_danger_up")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_2_in_direction, S_UP), out1, out2), 2, name="if_danger_2_up")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_LEFT), out1, out2), 2, name="if_danger_left")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_2_in_direction, S_LEFT), out1, out2), 2, name="if_danger_2_left")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_RIGHT), out1, out2), 2, name="if_danger_right")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_2_in_direction, S_RIGHT), out1, out2), 2, name="if_danger_2_right")

pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_DOWN), out1, out2), 2, name="if_moving_down")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_RIGHT), out1, out2), 2, name="if_moving_right")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_LEFT), out1, out2), 2, name="if_moving_left")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_UP), out1, out2), 2, name="if_moving_up")

# food sensing, used for Food&Time fitness function
# pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_up, out1, out2), 2, name="if_food_up")
# pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_down, out1, out2), 2, name="if_food_down")
# pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_left, out1, out2), 2, name="if_food_left")
# pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_right, out1, out2), 2, name="if_food_right")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def stepsFitness(individual):
    ## execute 4 times to get an average
    func = toolbox.compile(expr=individual)

    absScore, absFood, absSteps, absTime = 0, 0, 0, 0
    rounds = 4
    for i in range(rounds):    
        totalScore, foodsEaten, steps, timeSurvived = runGame(func, compiled=True)

        absScore += totalScore
        absFood += foodsEaten
        absSteps += steps
        absTime += timeSurvived

    avgScore = (absScore / rounds)
    avgFoodEaten = (absFood / rounds)
    avgSteps = (absSteps / rounds)
    avgTimeSurvived = (absTime / rounds)

    return avgSteps,

def foodAndTimeFitness(individual):
    ## execute 4 times to get an average
    func = toolbox.compile(expr=individual)

    absScore, absFood, absSteps, absTime = 0, 0, 0, 0
    rounds = 4
    for i in range(rounds):    
        totalScore, foodsEaten, steps, timeSurvived = runGame(func, compiled=True)

        absScore += totalScore
        absFood += foodsEaten
        absSteps += steps
        absTime += timeSurvived

    avgScore = (absScore / rounds)
    avgFoodEaten = (absFood / rounds)
    avgSteps = (absSteps / rounds)
    avgTimeSurvived = (absTime / rounds)

    return avgFoodEaten * 1000 + avgTimeSurvived * 100 + avgScore * 10,

def evaluateByFood(individual):
    ## execute 4 times to get an average
    absScore, absFood, absSteps, absTime = 0, 0, 0, 0
    rounds = 4
    for i in range(rounds):    
        totalScore, foodsEaten, steps, timeSurvived = runGame(individual)

        absScore += totalScore
        absFood += foodsEaten
        absSteps += steps
        absTime += timeSurvived

    avgScore = (absScore / rounds)
    avgFoodEaten = (absFood / rounds)
    avgSteps = (absSteps / rounds)
    avgTimeSurvived = (absTime / rounds)

    return avgFoodEaten

toolbox.register("evaluate", stepsFitness)
toolbox.register("select", tools.selRoulette)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

def main(seed = 118):
    global snake
    global pset
    global toolbox
    
    random.seed(seed)

    pop = toolbox.population(n=2000)
    hof = tools.HallOfFame(5)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.6, 200, stats=mstats, halloffame=hof, verbose=True)

    best = tools.selBest(pop, 1)[0]
    print evaluateByFood(best) # fitness by food


## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #

if __name__ == '__main__':
    main()
