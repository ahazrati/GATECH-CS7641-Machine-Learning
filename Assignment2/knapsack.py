import sys
import os
import time
import csv
from abalone_test import TRAINING_ITERATIONS
sys.path.append("/Users/luwang/Desktop/ML/ABAGAIL.jar")

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array




"""
Commandline parameter(s):
    none
"""

# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 40
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack 
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

TRAINING_ITERATIONS = [2000*i for i in range(100)]
REPEAT = 3

for n in range(REPEAT):
    iterdata = []
    round_start = time.time()
    print "REPEAT " + str(n)
    for iteration in TRAINING_ITERATIONS:
        print "iteration " + str(iteration)
        # create copies
        fill = [COPIES_EACH] * NUM_ITEMS
        copies = array('i', fill)
        
        # create weights and volumes
        fill = [0] * NUM_ITEMS
        weights = array('d', fill)
        volumes = array('d', fill)
        for i in range(0, NUM_ITEMS):
            weights[i] = random.nextDouble() * MAX_WEIGHT
            volumes[i] = random.nextDouble() * MAX_VOLUME
        
        
        # create range
        fill = [COPIES_EACH + 1] * NUM_ITEMS
        ranges = array('i', fill)
        
        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = UniformCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        
        start = time.time()
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, iteration)
        fit.train()
        end = time.time()
        rhc_time = end - start
        rhc_fit = ef.value(rhc.getOptimal())
#         print "RHC: " + str(rhc_fit)
        
        start = time.time()
        sa = SimulatedAnnealing(1e11, .95, hcp)
        fit = FixedIterationTrainer(sa, iteration/200)
        fit.train()
        end = time.time()
        sa_time = end - start
        sa_fit = ef.value(sa.getOptimal())
#         print "SA: " + str(sa_fit)
        
        start = time.time()
        ga = StandardGeneticAlgorithm(200, 150, 25, gap)
        fit = FixedIterationTrainer(ga, iteration)
        fit.train()
        end = time.time()
        ga_time = end - start
        ga_fit = ef.value(ga.getOptimal())
#         print "GA: " + str(ga_fit)
        
        start = time.time()
        mimic = MIMIC(200, 100, pop)
        fit = FixedIterationTrainer(mimic, iteration/200)
        fit.train()
        end = time.time()
        mimic_time = end - start
        mimic_fit = ef.value(mimic.getOptimal())
#         print "MIMIC: " + str(mimic_fit)
        
        row = [iteration,rhc_fit,rhc_time,sa_fit,sa_time,ga_fit,ga_time,mimic_fit,mimic_time] 
        iterdata.append(row) 
    
    round_end = time.time()   
    with open("knapsack_{}.csv".format(str(n)),'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(iterdata)  
          
    round_end = time.time()
    print "Takes total "+ str(round_end - round_start) + " secs"

