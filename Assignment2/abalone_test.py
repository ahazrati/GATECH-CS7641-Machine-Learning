from __future__ import with_statement
import csv
import time

import sys
import csv
sys.path.append("/Users/luwang/Desktop/ML/ABAGAIL.jar")

from func.nn.backprop import BackPropagationNetworkFactory
# from func.nn.backprop import StochasticBackPropagationTrainer
# from func.nn.backprop import StandardUpdateRule

from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

## for toy project
# INPUT_FILE = "/Users/luwang/Desktop/ML/lwang628/assignment1_0909/census-encoded-5.csv"
# INPUT_LAYER = 5
# HIDDEN_LAYER = 3
# OUTPUT_LAYER = 1
# 
# TRAINING_ITERATIONS = range(3000)
# interval = 10
# REPEAT = 5

## for real challenge
INPUT_FILE = "/Users/luwang/Desktop/ML/lwang628/assignment1_0909/census-encoded-62.csv"
INPUT_LAYER = 62
HIDDEN_LAYER = 30
OUTPUT_LAYER = 1

TRAINING_ITERATIONS = range(10000)
interval = 100
REPEAT = 1


def initialize_instances(FILE):
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []
    # Read in the abalone.txt CSV file
    with open(FILE, "r") as abalone:
        reader = csv.reader(abalone)
        
        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) < 0.5 else 1))
            instances.append(instance)
    return instances

    
def train(oa, network, oaName, instances,measure):
    """Train a given network on a set of instances.
        
        :param OptimizationAlgorithm oa:
        :param BackPropagationNetwork network:
        :param str oaName:
        :param list[Instance] instances:
        :param AbstractErrorMeasure measure:
        """
    #print "\nError results for %s\n---------------------------" % (oaName,)
    
    # training error each iteration
    iterdata = []
    training_time = 0.
    
    for iteration in TRAINING_ITERATIONS:
        #if oaName == "GA" and iteration >= int(len(TRAINING_ITERATIONS)/5):
        #    continue
        if iteration >= int(len(TRAINING_ITERATIONS)/5):
            break
        
        start = time.time()
        oa.train()
        end = time.time()
        training_time += end - start       
        
        if iteration % interval != 0:
            continue  
         
        correct,incorrect = 0,0  
        error = 0.         
        for instance in instances[:1000]:            
            network.setInputValues(instance.getData())
            network.run()
             
            actual = instance.getLabel().getContinuous()
            predicted = network.getOutputValues().get(0)
            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1
                
            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)
            #print "output, example,error",output, example,measure.value(output, example)
            
        accuracy =  1.*correct/(correct + incorrect) 
                 
        iterdata.append([iteration,accuracy,error,training_time])
#         print 'iteration,accuracy,error,training_time',iteration,accuracy,error,training_time  
    return iterdata   

        
    
def main():
    """Run algorithms on the abalone dataset."""
##  for optimizers with default setting   
#     for n in range(REPEAT):
#         instances = initialize_instances(INPUT_FILE)[:5000]
#          
#         factory = BackPropagationNetworkFactory()
#         measure = SumOfSquaresError()
#         data_set = DataSet(instances)
#          
#         networks = []  # BackPropagationNetwork
#         nnop = []  # NeuralNetworkOptimizationProblem
#         oa = []  # OptimizationAlgorithm
#         oa_names = ["RHC", "SA", "GA"]
#         results = ""
#          
#         for name in oa_names:
#             classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
#             networks.append(classification_network)
#             nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))
#          
#         oa = [RandomizedHillClimbing(nnop[0]),
#               SimulatedAnnealing(1E11, .95, nnop[1]),
#               StandardGeneticAlgorithm(200, 100, 10, nnop[2])]
#          
#          
#         for i, name in enumerate(oa_names):
#             round_start = time.time()
#             if name == "GA" and n >= int(REPEAT/2):
#                 continue 
#                            
#             iterdata = train(oa[i], networks[i], oa_names[i], instances,measure)
#             output_name = name + "_ANN_{}.csv".format(n)
#             round_end = time.time()
#             with open(output_name,'wb') as resultFile:
#                 wr = csv.writer(resultFile, dialect='excel')
#                 wr.writerows(iterdata) 
#                 print output_name, " : ",round_end - round_start,"seconds"

    
    for n in range(REPEAT):
        instances = initialize_instances(INPUT_FILE)[:5000]
         
        factory = BackPropagationNetworkFactory()
        measure = SumOfSquaresError()
        data_set = DataSet(instances)
         
        networks = []  # BackPropagationNetwork
        nnop = []  # NeuralNetworkOptimizationProblem
        oa = []  # OptimizationAlgorithm
        
#         oa_names = ["1e10","1e12", "1e13", "1e15"]
#         oa_names = ["cf0.1","cf0.25","cf0.5", "cf0.75"]
         
        oa_names = ["toMutate20","toMutate50","toMutate100","toMutate180"]
        
        results = ""
         
        for name in oa_names:
            classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
            networks.append(classification_network)
            nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))
         
#         oa = [SimulatedAnnealing(1E10, .95, nnop[0]),
#               SimulatedAnnealing(1E12, .95, nnop[1]),
#               SimulatedAnnealing(1E13, .95, nnop[2]),
#               SimulatedAnnealing(1E15, .95, nnop[3])]
        
#         oa = [SimulatedAnnealing(1E11, .1, nnop[0]),
#               SimulatedAnnealing(1E11, .25, nnop[1]),
#               SimulatedAnnealing(1E11, .5, nnop[2]),
#               SimulatedAnnealing(1E11, .75, nnop[3])]
        
        oa = [StandardGeneticAlgorithm(200, 100, 20, nnop[0]),
              StandardGeneticAlgorithm(200, 100, 50, nnop[1]),
              StandardGeneticAlgorithm(200, 100, 100, nnop[2]),
              StandardGeneticAlgorithm(200, 100, 180, nnop[3])]
        
                
         
        for i, name in enumerate(oa_names):
            round_start = time.time()
#             if name == "GA" and n >= int(REPEAT/2):
#                 continue 
#                            
            iterdata = train(oa[i], networks[i], oa_names[i], instances,measure)
            output_name = name + "_ANN_{}.csv".format(n)
            round_end = time.time()
            with open(output_name,'wb') as resultFile:
                wr = csv.writer(resultFile, dialect='excel')
                wr.writerows(iterdata) 
                print output_name, " : ",round_end - round_start,"seconds"        
            

if __name__ == "__main__":
    main()


