# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
import sys
import os
import csv
import time
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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array




"""
Commandline parameter(s):
    none
"""

# set N value.  This is the number of points
N = 50
random = Random()
# TRAINING_ITERATIONS = [500*i for i in range(11)]
TRAINING_ITERATIONS = [500*i for i in range(1,400)]
REPEAT = 1


iterdata = []
for n in range(REPEAT):
    round_start = time.time()
    print "Repeat" + str(n)
    
    for iteration in TRAINING_ITERATIONS:  
        print "iteration " + str(iteration)     
        points = [[0 for x in xrange(2)] for x in xrange(N)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()
        
        ef = TravelingSalesmanRouteEvaluationFunction(points)
        odd = DiscretePermutationDistribution(N)
        nf = SwapNeighbor()
        mf = SwapMutation()
        cf = TravelingSalesmanCrossOver(ef)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        
        start = time.time()
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, iteration)
        fit.train()
        end = time.time()
        rhc_fit = ef.value(rhc.getOptimal())
        rhc_time = end - start
        print "RHC: " + str(rhc_fit)
         
        start = time.time() 
        sa = SimulatedAnnealing(1E12, .999, hcp)
        fit = FixedIterationTrainer(sa, iteration)
        fit.train()
        end = time.time()
        sa_fit = ef.value(sa.getOptimal())
        sa_time = end - start
        print "SA: " + str(sa_fit)    
        
#         if iteration <= 5000:
#               
#             start = time.time()
#             ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
#             fit = FixedIterationTrainer(ga, iteration)
#             fit.train()
#             end = time.time()
#             ga_fit = ef.value(ga.getOptimal())
#             ga_time = end - start
#             print "GA: " + str(ga_fit)
#              
#              
#             # for mimic we use a sort encoding
#             ef = TravelingSalesmanSortEvaluationFunction(points);
#             fill = [N] * N
#             ranges = array('i', fill)
#             odd = DiscreteUniformDistribution(ranges);
#             df = DiscreteDependencyTree(.1, ranges); 
#             pop = GenericProbabilisticOptimizationProblem(ef, odd, df);
#              
#             start = time.time()
#             mimic = MIMIC(500, 100, pop)
#             fit = FixedIterationTrainer(mimic, iteration)
#             fit.train()
#             end = time.time()
#             mimic_fit = ef.value(mimic.getOptimal())
#             mimic_time = end - start
#             print "MIMIC: " + str(mimic_fit)
#             row = [n,iteration,rhc_fit,rhc_time,sa_fit,sa_time,ga_fit,ga_time,mimic_fit,mimic_time]
#         else:
#             row = [n,iteration,rhc_fit,rhc_time,sa_fit,sa_time]
#         print
        
        row =[n,iteration,rhc_fit, rhc_time, sa_fit,sa_time] 
        iterdata.append(row)
    
with open("travelingsalesman_200000.csv".format(n),'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(iterdata) 
        
        
# for n in range(REPEAT):
#     round_start = time.time()
#     iterdata = []
#     
#     for iteration in TRAINING_ITERATIONS:  
#         print "iteration" + iteration     
#         points = [[0 for x in xrange(2)] for x in xrange(N)]
#         for i in range(0, len(points)):
#             points[i][0] = random.nextDouble()
#             points[i][1] = random.nextDouble()
#         
#         ef = TravelingSalesmanRouteEvaluationFunction(points)
#         odd = DiscretePermutationDistribution(N)
#         nf = SwapNeighbor()
#         mf = SwapMutation()
#         cf = TravelingSalesmanCrossOver(ef)
#         hcp = GenericHillClimbingProblem(ef, odd, nf)
#         gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
#         
#         start = time.time()
#         rhc = RandomizedHillClimbing(hcp)
#         fit = FixedIterationTrainer(rhc, iteration)
#         fit.train()
#         end = time.time()
#         rhc_fit = ef.value(rhc.getOptimal())
#         rhc_time = end - start
#         print "RHC: " + str(rhc_fit)
#          
#         start = time.time() 
#         sa = SimulatedAnnealing(1E12, .999, hcp)
#         fit = FixedIterationTrainer(sa, iteration)
#         fit.train()
#         end = time.time()
#         sa_fit = ef.value(sa.getOptimal())
#         sa_time = end - start
#         print "SA: " + str(sa_fit)    
#         
#         if iteration < 5000:
#              
#             start = time.time()
#             ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
#             fit = FixedIterationTrainer(ga, iteration)
#             fit.train()
#             end = time.time()
#             ga_fit = ef.value(ga.getOptimal())
#             ga_time = end - start
#             print "GA: " + str(ga_fit)
#             
#             
#             # for mimic we use a sort encoding
#             ef = TravelingSalesmanSortEvaluationFunction(points);
#             fill = [N] * N
#             ranges = array('i', fill)
#             odd = DiscreteUniformDistribution(ranges);
#             df = DiscreteDependencyTree(.1, ranges); 
#             pop = GenericProbabilisticOptimizationProblem(ef, odd, df);
#             
#             start = time.time()
#             mimic = MIMIC(500, 100, pop)
#             fit = FixedIterationTrainer(mimic, iteration)
#             fit.train()
#             end = time.time()
#             mimic_fit = ef.value(mimic.getOptimal())
#             mimic_time = end - start
#             print "MIMIC: " + str(mimic_fit)
#             row = [iteration,rhc_fit,rhc_time,sa_fit,sa_time,ga_fit,ga_time,mimic_fit,mimic_time]
#         else:
#             row = [iteration,rhc_fit,rhc_time,sa_fit,sa_time]
#         print
#         
#         
#         iterdata.append(row)
#     
#     with open("travelingsalesman.csv".format(n),'wb') as resultFile:
#         wr = csv.writer(resultFile, dialect='excel')
#         wr.writerows(iterdata) 
#         
#     round_end = time.time()    
#     print "REPEAT " + str(n) + " takes {} seconds".format(round_end - round_start)
#     print 
           
    
