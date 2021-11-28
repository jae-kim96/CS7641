from math import gamma
from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def plot_results(result_dict, test_name, xlabel, learner):
    '''
    A helper function to help plot
    '''
    for k in result_dict.keys():
        curr = result_dict[k]
        plt.plot(curr[0], curr[1])
        plt.xlabel(xlabel)
        plt.ylabel(k)
        plt.title(f'{test_name} vs {k}')
        plt.savefig(f'forest/{learner}/{test_name} vs {k}')
        plt.close()

def value_iteration(P, R, prob_size):
    '''
    Value Iteration experiment for the Forest Management Problem
    '''
    ## Testing Different Values of Gamma
    gamma = np.arange(0.01, 1.0, 0.01).tolist()
    gamma_results = {
        'iterations': [gamma, []],
        'rewards': [gamma, []],
        'time': [gamma, []],
    }
    for g in gamma:
        learner = ValueIteration(P, R, gamma = g)
        learner.run()

        ## Getting Stats
        gamma_results['iterations'][1].append(learner.iter)
        gamma_results['rewards'][1].append(np.mean(learner.V))
        gamma_results['time'][1].append(learner.time)
        
    # print(gamma_results)
    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(gamma_results, f'VI ({prob_size}) - Gamma', 'gamma', 'vi')
    ## Printing highest Reward
    best_reward = max(gamma_results['rewards'][1])
    print(f'for Num States = {prob_size}, Best Reward = {best_reward}')

    ## Testing Different Values of Gamma
    eps = np.arange(0.01, 1.0, 0.01).tolist()
    eps_results = {
        'iterations': [eps, []],
        'rewards': [eps, []],
        'time': [eps, []],
    }
    for e in eps:
        learner = ValueIteration(P, R, gamma = 0.99, epsilon = e)
        learner.run()

        ## Getting Stats
        eps_results['iterations'][1].append(learner.iter)
        eps_results['rewards'][1].append(np.mean(learner.V))
        eps_results['time'][1].append(learner.time)
        

    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(eps_results, f'VI ({prob_size}) - Epsilon', 'epsilon', 'vi')
       

def policy_iteration(P, R, prob_size):
    '''
    Policy Iteration experiment for the Forest Management Problem
    '''
    ## Testing Different Values of Gamma
    gamma = np.arange(0.01, 1.0, 0.01).tolist()
    gamma_results = {
        'iterations': [gamma, []],
        'rewards': [gamma, []],
        'time': [gamma, []],
    }
    for g in gamma:
        learner = PolicyIteration(P, R, gamma = g)
        learner.run()

        ## Getting Stats
        gamma_results['iterations'][1].append(learner.iter)
        gamma_results['rewards'][1].append(np.mean(learner.V))
        gamma_results['time'][1].append(learner.time)
        
    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(gamma_results, f'PI ({prob_size}) - Gamma', 'gamma', 'pi')
    ## Printing highest Reward
    best_reward = max(gamma_results['rewards'][1])
    print(f'for Num States = {prob_size}, Best Reward = {best_reward}')


def qlearning(P, R, prob_size):
    '''
    QLearner experiment for the Forest Management Problem
    '''
    ## Testing Different Values of Gamma
    alpha = np.arange(0.01, 1.0, 0.01).tolist()
    # alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    alpha_results = {
        'rewards': [alpha, []],
        'time': [alpha, []],
    }
    for a in alpha:
        learner = QLearning(P, R, gamma = 0.99, alpha = a, epsilon = 0.5, n_iter = 20000)
        learner.run()

        ## Getting Stats
        # gamma_results['iterations'][1].append(learner.iter)
        alpha_results['rewards'][1].append(np.mean(learner.V))
        alpha_results['time'][1].append(learner.time)
        
    # print(gamma_results)
    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(alpha_results, f'Q ({prob_size}) - Alpha', 'alpha', 'q')

    ## Testing Different Values of Gamma
    gamma = np.arange(0.01, 1.0, 0.01).tolist()
    # gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    gamma_results = {
        'rewards': [gamma, []],
        'time': [gamma, []],
    }
    for g in gamma:
        learner = QLearning(P, R, gamma = g, alpha = 0.99, epsilon = 0.5, n_iter = 20000)
        learner.run()

        ## Getting Stats
        # gamma_results['iterations'][1].append(learner.iter)
        gamma_results['rewards'][1].append(np.mean(learner.V))
        gamma_results['time'][1].append(learner.time)
        
    # print(gamma_results)
    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(gamma_results, f'Q ({prob_size}) - Gamma', 'gamma', 'q')
    ## Printing highest Reward
    # best_reward = max(gamma_results['rewards'][1])
    # print(f'for Num States = {prob_size}, Best Reward = {best_reward}')

    ## Testing Different Values of Gamma
    eps = np.arange(0.01, 1.0, 0.01).tolist()
    # eps = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    eps_results = {
        'rewards': [eps, []],
        'time': [eps, []],
    }
    for e in eps:
        learner = QLearning(P, R, gamma = 0.99, epsilon = e, n_iter = 20000)
        t0 = time.time()
        learner.run()
        total_time = time.time() - t0

        ## Getting Stats
        # eps_results['iterations'][1].append(learner.iter)
        eps_results['rewards'][1].append(np.mean(learner.V))
        eps_results['time'][1].append(total_time)
        
    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(eps_results, f'Q ({prob_size}) - Epsilon', 'epsilon', 'q')

    ## Testing the number of iterations
    n_iters = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
    rewards = list()
    times = list()
    for n in n_iters:
        learner = QLearning(P, R, gamma = 0.99, alpha = 0.99, epsilon = 0.0001, n_iter = n)
        t0 = time.time()
        learner.run()
        total_time = time.time() - t0

        rewards.append(np.mean(learner.V))
        times.append(total_time)
    iter_results = {
        'rewards': [n_iters, rewards],
        'time': [n_iters, times]
    }
    plot_results(iter_results, f'Q ({prob_size}) - N-Iter', 'n_iter', 'q')

def vi_experiment(sizes):
    '''
    Testing the problem size and average reward and cut ratio from those problem sizes
    Gamma = 0.99
    '''
    print('Running VI Experiment')
    rewards = list()
    cut_ratios = list()
    for s in sizes:
        # print(f'Running VI for size = {s}')
        P, R = forest(S = s, r1 = 200, r2 = 2)
        learner = ValueIteration(P, R, gamma = 0.99)
        learner.run()
        ## Appending Cut_Ratio
        # print(f'{sum(learner.policy)}/{len(learner.policy)}')
        cut_ratios.append((sum(learner.policy)) / (len(learner.policy)))

        ## Appending Mean Rewards
        rewards.append(np.mean(learner.V))
    ## Rewards Plotting
    results = {
        'Sizes': sizes,
        'Avg_Reward': rewards
    }
    results_df = pd.DataFrame(results).set_index('Sizes')
    results_df.plot()
    plt.title('Problem Size vs Avg Rewards')
    plt.xlabel('Problem Size')
    plt.xscale('log')
    plt.ylabel('Rewards')
    plt.savefig('forest/vi/Problem Size vs Avg Rewards.png')
    plt.close()
    ## Cut Ratio Plotting
    results = {
        'Sizes': sizes,
        'Cut Ratio': cut_ratios
    }
    results_df = pd.DataFrame(results).set_index('Sizes')
    results_df.plot()
    plt.title('Problem Size vs Cut Ratio')
    plt.xlabel('Problem Size')
    plt.xscale('log')
    plt.ylabel('Cut Ratio')
    plt.savefig('forest/vi/Problem Size vs Cut Ratio.png')
    plt.close()

def pi_experiment(sizes):
    '''
    Testing the problem size and average reward and cut ratio from those problem sizes
    Gamma = 0.99
    '''
    print('Running PI Experiment')
    rewards = list()
    cut_ratios = list()
    for s in sizes:
        print(f'Running PI for size = {s}')
        P, R = forest(S = s, r1 = 200, r2 = 2)
        learner = PolicyIteration(P, R, gamma = 0.99)
        learner.run()
        ## Appending Cut_Ratio
        # print(f'{sum(learner.policy)}/{len(learner.policy)}')
        cut_ratios.append((sum(learner.policy)) / (len(learner.policy)))

        ## Appending Mean Rewards
        rewards.append(np.mean(learner.V))
    ## Rewards Plotting
    results = {
        'Sizes': sizes,
        'Avg_Reward': rewards
    }
    results_df = pd.DataFrame(results).set_index('Sizes')
    results_df.plot()
    plt.title('Problem Size vs Avg Rewards')
    plt.xlabel('Problem Size')
    plt.xscale('log')
    plt.ylabel('Rewards')
    plt.savefig('forest/pi/Problem Size vs Avg Rewards.png')
    plt.close()
    ## Cut Ratio Plotting
    results = {
        'Sizes': sizes,
        'Cut Ratio': cut_ratios
    }
    results_df = pd.DataFrame(results).set_index('Sizes')
    results_df.plot()
    plt.title('Problem Size vs Cut Ratio')
    plt.xlabel('Problem Size')
    plt.xscale('log')
    plt.ylabel('Cut Ratio')
    plt.savefig('forest/pi/Problem Size vs Cut Ratio.png')
    plt.close()

def q_experiment(sizes):
    '''
    Testing the problem size and average reward and cut ratio from those problem sizes
    Gamma = 0.99
    '''
    print('Running Q Experiment')
    rewards = list()
    cut_ratios = list()
    for s in sizes:
        # print(f'Running VI for size = {s}')
        P, R = forest(S = s, r1 = 200, r2 = 2)
        learner = QLearning(P, R, gamma = 0.99, alpha = 0.99, n_iter = 40000)
        learner.run()
        ## Appending Cut_Ratio
        # print(f'{sum(learner.policy)}/{len(learner.policy)}')
        cut_ratios.append((sum(learner.policy)) / (len(learner.policy)))

        ## Appending Mean Rewards
        rewards.append(np.mean(learner.V))
    ## Rewards Plotting
    results = {
        'Sizes': sizes,
        'Avg_Reward': rewards
    }
    results_df = pd.DataFrame(results).set_index('Sizes')
    results_df.plot()
    plt.title('Problem Size vs Avg Rewards')
    plt.xlabel('Problem Size')
    plt.xscale('log')
    plt.ylabel('Rewards')
    plt.savefig('forest/q/Problem Size vs Avg Rewards.png')
    plt.close()
    ## Cut Ratio Plotting
    results = {
        'Sizes': sizes,
        'Cut Ratio': cut_ratios
    }
    results_df = pd.DataFrame(results).set_index('Sizes')
    results_df.plot()
    plt.title('Problem Size vs Cut Ratio')
    plt.xlabel('Problem Size')
    plt.xscale('log')
    plt.ylabel('Cut Ratio')
    plt.savefig('forest/q/Problem Size vs Cut Ratio.png')
    plt.close()

if __name__ == '__main__':
    ## Initializing Forest Problem
    size = [200, 2000]
    # size = [2000]
    for s in size:
        P, R = forest(S = s, r1 = 200, r2 = 2)
        prob_size = f'states = {s}'
        # Experiments for Problems
        print(f'------------------------RUNNING VALUE ITERATION => States = {s}------------------------')
        value_iteration(P, R, prob_size)
        print(f'------------------------RUNNING POLICY ITERATION => States = {s}------------------------')
        policy_iteration(P, R, prob_size)
        print(f'------------------------RUNNING QLEARNER => States = {s}------------------------')
        qlearning(P, R, prob_size)

    print(f'------------------------Forest Experiment with Sizes------------------------')
    sizes = [2, 20, 200, 2000, 20000]
    vi_experiment(sizes)
    sizes = [2, 20, 200, 2000]
    pi_experiment(sizes)
    sizes = [2, 20, 200, 2000, 20000]
    q_experiment(sizes)



    