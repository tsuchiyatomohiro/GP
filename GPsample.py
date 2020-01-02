# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:40:52 2019

@author: tomo
"""
import numpy as np
#import random
from random import random, randint, seed 
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt
#

POP_SIZE        = 60   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
GENERATIONS     = 250  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate 
PROB_MUTATION   = 0.2  # per-node mutation probability 

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
FUNCTIONS = [add, sub, mul]
TERMINALS = ['x', -2, -1, 0, 1, 2] 

def target_func(x): # evolution's target
    return x*x*x*x + x*x*x + x*x + x + 1

def generate_dataset(): # generate 101 data points from target_func
    dataset = []
    for x in range(-100,101,2): 
        x /= 100
        dataset.append([x, target_func(x)])
    return dataset

class GPTree:
    def __init__(self, data = None, left = None, right = None):
        self.data  = data
        self.left  = left
        self.right = right
        
    def node_label(self): # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x): 
        if (self.data in FUNCTIONS): 
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif self.data == 'x': return x
        else: return self.data
            
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)            
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION: # mutate at this node
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation() 

    def size(self): # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self): # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)  
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree
# end class GPTree
                   
def init_population(): # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            #print(t)
            pop.append(t) 
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t) 
            
    #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",pop[0])
    return pop

def fitness(individual, dataset): # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]))
                
def selection(population, fitnesses): # select one individual using tournament selection
    #print(population)
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    #print(tournament)
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    #print(tournament_fitnesses)
    #print(tournament_fitnesses.index(max(tournament_fitnesses)))
    #print("WWWWW")
    #print(tournament[tournament_fitnesses.index(max(tournament_fitnesses))])
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]]) 

def gomi():
    pool_next = []
    next_list=[i for i in range(self.N)]
    while len(pool_next) < self.N:
        #offspring1=copy.deepcopy(self.pool[random.choices(next_list,weights=self.Glist)])
        #pool_next.append(self.pool[random.choices(next_list,weights=self.Glist)])
        a=random.choices(next_list,weights=self.Glist)
        
        #print(a)
        #print(self.pool[a[0]])
        offspring1=copy.deepcopy(self.pool[a[0]])
        #pass
        pool_next.append(offspring1)
    self.pool = pool_next[:]  
    
def rouletteSelection(population, fitnesses):
    import random
    #print("A")
    # ルーレット選択の関数
    # ここを実装する
    #a=[1,2,3,4,5,6]
    #b=[1,3,56,787,989,1]
    #c=random.choices(a,weights=b)
    next_list=[i for i in range(POP_SIZE)]
    a=random.choices(next_list,weights=fitnesses)
    #print(a)
    b=a[0]
    #print(b)
    return deepcopy(population[b])
    #print(population,"\n" , fitnesses)

         
def main():      
    # init stuff
    seed() # init internal state of random number generator
    dataset = generate_dataset()
    population= init_population() 
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    AVE_list=[]
    MAX_list=[]
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

    # go evolution!
    for gen in range(GENERATIONS):        
        nextgen_population=[]
        for i in range(POP_SIZE):
            #rouletteSelection(population, fitnesses)
            #トーナメント
            parent1 = selection(population, fitnesses)
            #print(parent1)
            parent2 = selection(population, fitnesses)
            
# =============================================================================
#             #ルーレット
#             parent1 = rouletteSelection(population, fitnesses)
#             parent2 = rouletteSelection(population, fitnesses)
#             
# =============================================================================
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population=nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        print("________________")
        print("GEN :" ,gen)
        print("AVE :",np.mean(fitnesses))
        print("MAX :",max(fitnesses))
        AVE_list.append(np.mean(fitnesses))
        MAX_list.append(max(fitnesses))
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses),3), ", best_of_run:") 
            best_of_run.print_tree()
        if best_of_run_f == 1: break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f,3)))
    best_of_run.print_tree()
    y=range(0,len(MAX_list))
    plt.plot(y,AVE_list)
    plt.plot(y,MAX_list) 
    plt.show()
main()