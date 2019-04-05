import sys
import os
import time
from abalone_test import TRAINING_ITERATIONS
import csv
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
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array



"""
Commandline parameter(s):
   none
"""
TRAINING_ITERATIONS = [2000*i for i in range(100)]
REPEAT = 3
# GA/MIMIC use actually 1/200

N=60
T=N/10
fill = [2] * N
ranges = array('i', fill)

for n in range(REPEAT):
    iterdata = []
    round_start = time.time()
    for iteration in TRAINING_ITERATIONS:
        print "iteration " + str(iteration)
        
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
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
        sa = SimulatedAnnealing(1E11, .95, hcp)
        fit = FixedIterationTrainer(sa, iteration)
        fit.train()
        end = time.time()
        sa_time = end - start
        sa_fit = ef.value(sa.getOptimal())
        print "SA: " + str(sa_fit)
        
        start = time.time()
        ga = StandardGeneticAlgorithm(200, 100, 10, gap)
        fit = FixedIterationTrainer(ga, iteration/200)
        fit.train()
        end = time.time()
        ga_time = end - start
        ga_fit = ef.value(ga.getOptimal())
#         print "GA: " + str(ga_fit)
        
        start = time.time()
        mimic = MIMIC(200, 20, pop)
        fit = FixedIterationTrainer(mimic, iteration/200)
        fit.train()
        end = time.time()
        mimic_time = end - start
        mimic_fit = ef.value(mimic.getOptimal())
#         print "MIMIC: " + str(mimic_fit)
        
        row = [iteration,rhc_fit,rhc_time,sa_fit,sa_time,ga_fit,ga_time,mimic_fit,mimic_time] 
        iterdata.append(row)    
        
    with open("continouspeaks_{}.csv".format(str(n)),'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(iterdata)  
          
    round_end = time.time()
    print "Takes total "+ str(round_end - round_start) + " secs"
