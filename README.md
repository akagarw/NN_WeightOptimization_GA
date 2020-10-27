# Neural Network Weight Optimization using Genetic Algorithms
  Given Python Code in "NN_WtOpt.py" aims to solve the problem of Weight Optimization in Neural Networks using Genetic Algorithms.  
  Here the Model is evaluated on the Iris Dataset. 
### Architecture of NN:
- Number of input neurons = 4
- Number of hidden layers = 1
- Number of hidden neurons = 4
- Number of output neurons = 3
- Activation Function at the Hidden Layer = ReLu
- Activation Function at the Output Layer = Softmax
- Total Params = 35
 
### Genetic Algorithm Parameters:
- Population Size = 30
- Max Generations = 30
- Total Genes = Total Params = 35
- Using Value Encoding to represent chromosome solution
- Each Chromosome defined as = ( WtParam1, WtParam2,... , WtParam35 )
- Weight Param Range = \[ -2.0, 2.0 \]
- Probability Of Crossover = 0.8
- Probability Of Mutation = 0.8
- Objective Function = NN Model Evaluation Loss and Accuracy
- Selection Operator = Roulette Wheel Selection
- Crossover Operator = One Point Crossover
- Mutation Operator = One Point Mutation

## Algorithm Workflow
### Initial Population Generation:
Popuplation of Chromosome Solution. Each Chromosome is generated with a set of genes, whose values are randomly generated within the Weight Param Range. Exhibiting Value Encoding in Genotype Representation.
### Creating Next Generation:
#### Fitness Calculation : 
First, Each chromosome is conidered in its phenotype representation to be evaluated by the objective function to calculate its the NN Model Loss 
on the Iris Dataset to get the following metrics. Then the fitness of each chromesome is calculated using the Loss Metric in reference to the whole population. 
The Statistical Representations(Min, Max, Mean) of the calculated metrics are further stored for each generation.
#### Elitism:
Now, The Best 10% of the Population are chosen and taken as it is in the next Generation to keep best solution and control diversity and maintain exploration.
#### Selection:
To get the remaining population, from the Current Generation offsprings are generated. So based on the Fitness Metric, Two parents are selected at random 
by the Roulette Wheel Selection Method, from which Two offspring will be created based on their genetic material in genotype representation. 
#### Crossover and Mutation:
Based on a randomly generated number value, it is decided whether crossover and mutation operators will be performed on the Two chosen parents to generate corresponding offsprings.
The Probability Of Crossover and Probability Of Mutation are kept as such to explore more in the initial iterations of creating new generations to increase diversity 
whereas do more of exploitation in the generations to decrease diversity and converge to the optimal solution found.

  

