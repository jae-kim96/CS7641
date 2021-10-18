from os import stat
from typing import ValuesView
import mlrose_hiive as mlrose
from mlrose_hiive import fitness
from mlrose_hiive.fitness import four_peaks, queens
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


## Randomized Optimization Assignment - Fall 2021
## by Jaeyong Kim
##
## Utilizing mlrose to look at the different algorithms

## Randomized Hill Climbing
def random_hill_climbing(problem, name, prob_size):
    '''
    Randomized Hill Climbing Algorithm on the discrete optimization problems
    '''

    rand_hill = mlrose.RHCRunner(
        problem = problem,
        experiment_name = 'Random Hill Climbing - ' + name,
        max_attempts = 100,
        iteration_list = [100, 1000],
        restart_list = [0],
        seed = 99
    )

    print(f'====================START of RHC {name}====================')
    ## Wall Time
    t0 = time.time()
    stats, curves = rand_hill.run()
    wall_time = time.time() - t0
    # print(f'Wall Time = {wall_time} s')

    ## Min FEvals
    best_fit = curves['Fitness'].max()
    evals = curves[['FEvals', 'Fitness']].loc[curves['Fitness'] == best_fit]
    min_feval = evals['FEvals'].min()

    # print(curves.describe())
    ## Best Fitness 
    best_restart_avg = curves[['current_restart', 'Fitness']].groupby(by = ['current_restart']).mean()
    best_fit = best_restart_avg.idxmax()
    best_restart = best_fit[0]

    best_curves = curves.loc[curves['current_restart'] == best_restart]
    # print(best_curves)
    ## Plotting Function Evalutations over Iteration to show Computational Cost
    feval_plot = best_curves[['Iteration', 'FEvals']].set_index('Iteration')
    feval_plot.plot()
    plt.title('FEvals over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('FEvals')
    plt.savefig(f'rhc/{name}-{prob_size} - RHC FEval vs Iteration.png')

    ## Plotting Fitness vs Iteration
    best_plot = best_curves[['Iteration', 'Fitness']].set_index('Iteration')
    best_plot.plot()
    plt.title('Fitness over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.savefig(f'rhc/{name} - Random Hill Climbing (prob_size = {prob_size}).png')
    
    print(f'====================END of RHC {name}====================')

    return wall_time, min_feval

## Simulated Annealing Algorithm
def simulated_annealing(problem, name, prob_size):
    '''
    Simulated Annealing Algorithm on the discrete optimization problems
    '''
    print(f'====================START of SA {name}====================')
    decays = {
        'geom_decay': mlrose.GeomDecay,
        'exp_decay': mlrose.ExpDecay,
        'arith_decay': mlrose.ArithDecay
    }
    times = 0
    min_evals = 0
    # all_curves = dict()
    # for d in decays.keys():
    sa = mlrose.SARunner(
        problem = problem,
        experiment_name = 'Simulated Annealing - ' + name,
        max_attempts = 100,
        iteration_list = [100, 1000],
        temperature_list = [1, 100, 1000, 10000],
        # decay_list = [decays[d]],
        decay_list = [mlrose.GeomDecay, mlrose.ExpDecay, mlrose.ArithDecay], 
        seed = 99
    )
    
    ## Wall Time
    t0 = time.time()
    stats, curves = sa.run()
    wall_time = time.time() - t0
    times += wall_time
    # stats.to_csv('stats.csv')
    ## Getting Minimum Function Evals for best fitness
    best_fit = curves['Fitness'].max()
    evals = curves[['FEvals', 'Fitness']].loc[curves['Fitness'] == best_fit]
    min_feval = evals['FEvals'].min()
    min_evals += min_feval

    # print(f'Decay {d}: Wall Time = {wall_time} s')
    curves['Temperature'] = curves['Temperature'].astype(str).astype(int)
    
    # curves.to_csv(f'sa-{name}-{d}curves.csv')
    # stats.to_csv(f'sa-{name}-{d}stats.csv')
    ## Fitness Curves and Best Fitness
    avg_fit = curves[['Fitness', 'Temperature']].groupby(by = ['Temperature']).mean()
    best_params = avg_fit.idxmax()
    best_temp = best_params[0]
    
    best_curves = curves.loc[curves['Temperature'] == best_temp]
    # print(best_curves)
    ## Plotting Function Evalutations over Iteration to show Computational Cost
    feval_plot = best_curves[['Iteration', 'FEvals']].set_index('Iteration')
    feval_plot.plot()
    plt.title('FEvals over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('FEvals')
    plt.savefig(f'sa/{name}-{prob_size} - SA FEval vs Iteration.png')
    
    ## Plotting Fitness Over Iteration
    plot_curve = best_curves[['Iteration', 'Fitness']].set_index('Iteration')
    plot_curve.to_csv('Sample.csv')
    plot_curve.plot()
    plt.title('Fitness over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    # plt.savefig(f'sa/{name} - Simulated Annealing (Prob Size = {prob_size}, Decay = {d}, Temp = {best_temp}).png')
    plt.savefig(f'sa/{name} - Simulated Annealing (Prob Size = {prob_size}, Temp = {best_temp}).png')
        
    print(f'====================END of SA {name}====================')
    # avg_evals = min_evals / 3
    # return times, avg_evals
    return wall_time, min_feval

## Genetic Annealing Algorithm
def genetic_algorithm(problem, name, prob_size):
    '''
    Genetic Annealing Algorithm on the discrete optimization problems
    '''
    print(f'====================START of GA {name}====================')
    
    sa = mlrose.GARunner(
        problem = problem,
        experiment_name = 'Genetic Algorithm - ' + name,
        max_attempts = 100,
        iteration_list = [100, 1000],
        mutation_rates = [0.1, 0.25, 0.5],
        population_sizes = [50, 100, 200, 400],
        seed = 99
    )
    
    ## Wall Time
    t0 = time.time()
    stats, curves = sa.run()
    wall_time = time.time() - t0
    # print(f'Wall Time = {wall_time} s')

    ## Getting Minimum Function Evals for best fitness
    best_fit = curves['Fitness'].max()
    evals = curves[['FEvals', 'Fitness']].loc[curves['Fitness'] == best_fit]
    min_feval = evals['FEvals'].min()
    
    ## FInding the Best Fitting Parameters
    avg_fit = curves[['Fitness', 'Mutation Rate', 'Population Size']].groupby(by = ['Mutation Rate', 'Population Size']).mean()
    best_params = avg_fit.idxmax()
    # print(type(best_params))
    best_mutation = best_params[0][0]
    best_pop = best_params[0][1]

    best_curves = curves.loc[curves['Mutation Rate'] == best_mutation]
    best_curves = best_curves.loc[best_curves['Population Size'] == best_pop]
    
    # curves.to_csv(f'ga-{name}-curves.csv')
    # stats.to_csv(f'ga-{name}-stats.csv')

    ## Function Evals over Iterations
    feval_plot = best_curves[['Iteration', 'FEvals']].set_index('Iteration')
    feval_plot.plot()
    plt.title('FEvals over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('FEvals')
    plt.savefig(f'ga/{name}-{prob_size} - GA FEval vs Iteration.png')

    plot_curve = best_curves[['Iteration', 'Fitness']].set_index('Iteration')
    plot_curve.plot()
    plt.title('Fitness over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.savefig(f'ga/{name} - Genetic Algorithm (Prob Size = {prob_size}).png')

    print(f'====================END of GA {name}====================')
    return wall_time, min_feval


## Mimic algorithm
def mimic(problem, name, prob_size):
    '''
    Mimic Algorithm on the discrete optimization problems
    '''
    print(f'====================START of MIMIC {name}====================')
    
    sa = mlrose.MIMICRunner(
        problem = problem,
        experiment_name = 'MIMIC - ' + name,
        max_attempts = 100,
        iteration_list = [100, 1000],
        keep_percent_list = [0.25, 0.5, 0.75],
        population_sizes = [200, 500],
        seed = 99,
        use_fast_mimic = True
    )
    
    ## Wall Time
    t0 = time.time()
    stats, curves = sa.run()
    wall_time = time.time() - t0
    # print(f'Wall Time = {wall_time} s')
    
    ## Getting Minimum Function Evals for best fitness
    best_fit = curves['Fitness'].max()
    evals = curves[['FEvals', 'Fitness']].loc[curves['Fitness'] == best_fit]
    min_feval = evals['FEvals'].min()

    ## FInding the Best Fitting Parameters
    # print(curves)
    avg_fit = curves[['Fitness', 'Keep Percent', 'Population Size']].groupby(by = ['Keep Percent', 'Population Size']).mean()
    best_params = avg_fit.idxmax()
    # # print(type(best_params))
    best_keep = best_params[0][0]
    best_pop = best_params[0][1]

    best_curves = curves.loc[curves['Keep Percent'] == best_keep]
    best_curves = best_curves.loc[best_curves['Population Size'] == best_pop]
    
    feval_plot = best_curves[['Iteration', 'FEvals']].set_index('Iteration')
    feval_plot.plot()
    plt.title('FEvals over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('FEvals')
    plt.savefig(f'mimic/{name}-{prob_size} - MIMIC FEval vs Iteration.png')

    plot_curve = best_curves[['Iteration', 'Fitness']].set_index('Iteration')
    plot_curve.plot()
    plt.title('Fitness over Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.savefig(f'mimic/{name} - MIMIC (Prob Size = {prob_size}).png')
    
    print(f'====================END of MIMIC {name}====================')
    return wall_time, min_feval
    


if __name__ == "__main__":
    '''
    Running the scripts above exploring the optimizations
    1) Four Peaks
    2) OneMax
    3) Knapsack
    '''

    prob_sizes = [10, 20, 40, 80, 100]
    for prob_size in prob_sizes:
        print(f'||||||||||||||||||||||EVALUATING FOR PROB SIZE = {prob_size}||||||||||||||||||||||')
        ## Initializing 3 Optimization Problems
        ## Four Peaks Fitness Function
        four_peaks = mlrose.FourPeaks(t_pct = 0.1)
        four_peaks_disc_prob = mlrose.DiscreteOpt(
            length = prob_size,
            fitness_fn = four_peaks
        )
        rhc_time, rhc_min_feval  = random_hill_climbing(four_peaks_disc_prob, 'four_peaks', prob_size)
        sa_time, sa_min_feval  = simulated_annealing(four_peaks_disc_prob, 'four_peak', prob_size)
        ga_time, ga_min_feval = genetic_algorithm(four_peaks_disc_prob, 'four_peak', prob_size)
        mm_time, mm_min_feval = mimic(four_peaks_disc_prob, 'four_peak', prob_size)

        # Time Bar Plot
        four_peaks_times = pd.DataFrame.from_dict({
            'rhc': [rhc_time],
            'sa': [sa_time],
            'ga': [ga_time],
            'mm': [mm_time],
        }, orient = 'index')

        four_peaks_times.plot.bar()
        plt.xlabel('Algorithms')
        plt.ylabel('Times')
        plt.title('Wall Times for Algorithms')
        plt.savefig(f'times/FOUR PEAKS Wall Times for Prob Size = {prob_size}.png')

        # FEval Plots
        four_peaks_fevals = pd.DataFrame.from_dict({
            'rhc': [rhc_min_feval],
            'sa': [sa_min_feval],
            'ga': [ga_min_feval],
            'mm': [mm_min_feval],
        }, orient = 'index')

        four_peaks_fevals.plot.bar()
        plt.xlabel('Algorithms')
        plt.ylabel('FEvals for Best Fitness')
        plt.title('MIN FEvals for Algorithms')
        plt.savefig(f'fevals/FOUR PEAKS FEvals for Prob Size = {prob_size}.png')
        
        ## OneMax Fitness
        one_max = mlrose.OneMax()
        one_max_disc_prob = mlrose.DiscreteOpt(
            length = prob_size,
            fitness_fn = one_max
        )
        rhc_time, rhc_min_feval = random_hill_climbing(one_max_disc_prob, 'one_max', prob_size)
        sa_time, sa_min_feval = simulated_annealing(one_max_disc_prob, 'one_max', prob_size)
        ga_time, ga_min_feval = genetic_algorithm(one_max_disc_prob, 'one_max', prob_size)
        mm_time, mm_min_feval = mimic(one_max_disc_prob, 'one_max', prob_size)

        one_max_times = pd.DataFrame.from_dict({
            'rhc': [rhc_time],
            'sa': [sa_time],
            'ga': [ga_time],
            'mm': [mm_time],
        }, orient = 'index')

        one_max_times.plot.bar()
        plt.xlabel('Algorithms')
        plt.ylabel('Times')
        plt.title('Wall Times for Algorithms')
        plt.savefig(f'times/ONE MAX Wall Times for Prob Size = {prob_size}.png')

        ## FEval Plots
        one_max_fevals = pd.DataFrame.from_dict({
            'rhc': [rhc_min_feval],
            'sa': [sa_min_feval],
            'ga': [ga_min_feval],
            'mm': [mm_min_feval],
        }, orient = 'index')

        one_max_fevals.plot.bar()
        plt.xlabel('Algorithms')
        plt.ylabel('FEvals for Best Fitness')
        plt.title('MIN FEvals for Algorithms')
        plt.savefig(f'fevals/ONE MAX FEvals for Prob Size = {prob_size}.png')

        # ## Knapsack Fitness Function
        weights = np.random.randint(low = 1, high = 20, size = prob_size).tolist()
        values = np.random.randint(low = 1, high = 20, size = prob_size).tolist()
        knapsack = mlrose.Knapsack(
            weights = weights,
            values = values
        )
        knapsack_disc_prob = mlrose.DiscreteOpt(
            length = prob_size,
            fitness_fn = knapsack
        )

        rhc_time, rhc_min_feval = random_hill_climbing(knapsack_disc_prob, 'knapsack', prob_size)
        sa_time, sa_min_feval = simulated_annealing(knapsack_disc_prob, 'knapsack', prob_size)
        ga_time, ga_min_feval = genetic_algorithm(knapsack_disc_prob, 'knapsack', prob_size)
        mm_time, mm_min_feval = mimic(knapsack_disc_prob, 'knapsack', prob_size)

        ## Time Plot
        knapsack_times = pd.DataFrame.from_dict({
            'rhc': [rhc_time],
            'sa': [sa_time],
            'ga': [ga_time],
            'mm': [mm_time],
        }, orient = 'index')

        knapsack_times.plot.bar()
        plt.xlabel('Algorithms')
        plt.ylabel('Times')
        plt.title('Wall Times for Algorithms')
        plt.savefig(f'times/KNAPSACK Wall Times for Prob Size = {prob_size}.png')

        ## FEval Plots
        knapsack_fevals = pd.DataFrame.from_dict({
            'rhc': [rhc_min_feval],
            'sa': [sa_min_feval],
            'ga': [ga_min_feval],
            'mm': [mm_min_feval],
        }, orient = 'index')

        knapsack_fevals.plot.bar()
        plt.xlabel('Algorithms')
        plt.ylabel('FEvals for Best Fitness')
        plt.title('MIN FEvals for Algorithms')
        plt.savefig(f'fevals/Knapsack FEvals for Prob Size = {prob_size}.png')