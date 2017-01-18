# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
import multiprocessing
from functools import partial

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
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
        # self.ahead[0] == 0 or self.ahead[0] == (YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE - 1)

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

    # def sense_food(self, cond):
    #     if len(self.food) == 0: False
    #     first_food = self.food[0]
    #     head = self.body[0]
    #     return cond(first_food, head)

    def sense_food_up(self):
        loc = locMap[S_UP](self.body[0][0], self.body[0][1])
        return loc in self.food

    def sense_food_down(self):
        loc = locMap[S_UP](self.body[0][0], self.body[0][1])
        return loc in self.food

    def sense_food_left(self):
        loc = locMap[S_UP](self.body[0][0], self.body[0][1])
        return loc in self.food

    def sense_food_right(self):
        loc = locMap[S_UP](self.body[0][0], self.body[0][1])
        return loc in self.food

    def if_danger_right(self, out1, out2):
        [y, x] = self.body[0]
        dir = self.direction
        loc = [y, x]
        if dir == S_RIGHT:
            loc = [y + 1, x]
        elif dir == S_UP:
            loc = [y, x + 1]
        elif dir == S_DOWN:
            loc = [y, x - 1]
        elif dir == S_LEFT:
            loc = [y - 1, x]

        out1() if loc in self.body or is_wall(loc) else out2()

    def if_danger_left(self, out1, out2):
        [y, x] = self.body[0]
        dir = self.direction
        loc = [y, x]
        if dir == S_RIGHT:
            loc = [y - 1, x]
        elif dir == S_UP:
            loc = [y, x - 1]
        elif dir == S_DOWN:
            loc = [y, x + 1]
        elif dir == S_LEFT:
            loc = [y + 1, x]

        out1() if loc in self.body or is_wall(loc) else out2()

    def moves_in_direction(self, direction):
        return self.direction == direction

    def danger_in_direction(self, direction):
        [y, x] = self.body[0]
        loc = locMap[direction](y, x)
        return loc in snake.body or is_wall(loc)

    def food_same_vertical(self):
        food = self.food[0]
        head = self.body[0]
        return food[0] == head[0]

    def food_same_horizontal(self):
        food = self.food[0]
        head = self.body[0]
        return food[1] == head[1]

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
def runGame(individual):
    global snake

    func = toolbox.compile(expr=individual)

    ## execute 4 times to get an average
    absScore, absFood, absTimer = 0, 0, 0
    rounds = 4
    for i in range(rounds):
        totalScore = 0

        snake._reset()
        food = placeFood(snake)
        timer = 0
        foodsEaten = 0
        while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:
            ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
            func()

            snake.updatePosition()

            if snake.body[0] in food:
                foodsEaten += 1
                snake.score += 1
                food = placeFood(snake)
                if food is None:
                    return totalScore, foodsEaten, timer
                timer = 0
            else:
                snake.body.pop()
                timer += 1  # timesteps since last eaten

            totalScore += snake.score

        absScore += totalScore
        absFood += foodsEaten
        absTimer += timer

    return (absScore / rounds), (absFood / rounds), (absTimer / rounds)

pset = gp.PrimitiveSet("MAIN", 0)
pset.addTerminal(snake.changeDirectionRight, name="right")
pset.addTerminal(snake.changeDirectionDown, name="down")
pset.addTerminal(snake.changeDirectionLeft, name="left")
pset.addTerminal(snake.changeDirectionUp, name="up")
pset.addTerminal(lambda: True, name="forward")
# pset.addTerminal(snake.turnLeft, name="turn_left")
# pset.addTerminal(snake.turnRight, name="turn_right")

pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)
# pset.addPrimitive(lambda out1, out2: partial(snake.if_danger_ahead_2, out1, out2), 2, name="danger_ahead_2")
# pset.addPrimitive(lambda out1, out2: partial(snake.if_danger_right, out1, out2), 2, name="danger_right")
# pset.addPrimitive(lambda out1, out2: partial(snake.if_danger_left, out1, out2), 2, name="danger_left")
# pset.addPrimitive(snake.if_wall_ahead, 2)
# pset.addPrimitive(snake.if_tail_ahead, 2)
# pset.addPrimitive(snake.if_food_ahead, 2)
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_danger_ahead, out1, out2), 2, name="if_danger_ahead")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_danger_2_ahead, out1, out2), 2, name="if_danger_2_ahead")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_DOWN), out1, out2), 2, name="if_danger_down")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_UP), out1, out2), 2, name="if_danger_up")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_LEFT), out1, out2), 2, name="if_danger_left")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.danger_in_direction, S_RIGHT), out1, out2), 2, name="if_danger_right")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_up, out1, out2), 2, name="if_food_up")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_down, out1, out2), 2, name="if_food_down")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_left, out1, out2), 2, name="if_food_left")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.sense_food_right, out1, out2), 2, name="if_food_right")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_DOWN), out1, out2), 2, name="if_moving_down")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_RIGHT), out1, out2), 2, name="if_moving_right")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_LEFT), out1, out2), 2, name="if_moving_left")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, partial(snake.moves_in_direction, S_UP), out1, out2), 2, name="if_moving_up")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.food_same_horizontal, out1, out2), 2, name="food_same_horizontal")
pset.addPrimitive(lambda out1, out2: partial(if_then_else, snake.food_same_vertical, out1, out2), 2, name="food_same_vertical")
# pset.addPrimitive(snake.sense_danger_two_ahead, 2)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate(individual):
    totalScore, foodsEaten, timer = runGame(individual)
    return foodsEaten * 1000 + timer * 100 + totalScore * 10,
    # return foodsEaten,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.05)
# toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

def main():
    global snake
    global pset
    random.seed(107)
    # 103

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(5)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    # mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.45, 100, stats=mstats, halloffame=hof, verbose=True)

    best = tools.selBest(pop, 1)[0]

    print best

    import pygraphviz as pgv
    nodes, edges, labels = gp.graph(best)
    g = pgv.AGraph(nodeSep=1.0)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw("tree.pdf")

    raw_input("Press to continue display best run...")
    random.seed()
    displayStrategyRun(best)

    # print log
    return pop, log, hof


## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #

if __name__ == '__main__':
    main()
