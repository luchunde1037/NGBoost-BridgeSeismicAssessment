import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


class SSA:
    def __init__(self, N, dim, x_min, x_max, iterate_max, log_file='ssa_iteration_log.txt'):
        """
        Initialize SSA algorithm parameters
        :param N: population size (number of sparrows)
        :param dim: dimension of the hyperparameters
        :param x_min: lower bound of the search space
        :param x_max: upper bound of the search space
        :param iterate_max: maximum number of iterations
        """
        self.N = N
        self.dim = dim
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.iterate_max = iterate_max
        self.log_file = log_file

        # Initialize logger, set append mode
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s'
        )
        # If the file does not exist, write the header
        try:
            with open(log_file, 'x') as f:
                f.write('Iteration,Best Score\n')
        except FileExistsError:
            pass

    def log_iteration(self, iteration, best_score):
        """
        Log the results of each iteration to the log file
        :param iteration: current iteration
        :param best_score: best score in the current iteration
        """
        logging.info(f'{iteration},{best_score}')
        # Ensure the log content is flushed to the file immediately
        logging.getLogger().handlers[0].flush()

    def optimize(self, fitness_function):
        """
        Perform hyperparameter optimization using SSA
        :param fitness_function: fitness function to evaluate the hyperparameters
        :return: best fitness value, best hyperparameter combination
        """
        varepsilon = np.exp(-16)
        ST = 0.6  # safety threshold
        PD = int(0.6 * self.N)  # number of producers
        SD = int(0.2 * self.N)  # number of scouts

        # Initialize sparrow positions
        params = self.x_min + (self.x_max - self.x_min) * np.random.random([self.N, self.dim])

        # Calculate initial fitness
        fitness_scores = np.array([fitness_function(params[i]) for i in range(self.N)])
        iteration_data = []
        iterate = 0
        while iterate < self.iterate_max:
            # Sort fitness values
            sorted_indices = np.argsort(fitness_scores)
            best_params = params[sorted_indices[0]]
            best_score = fitness_scores[sorted_indices[0]]
            x_best = best_params.copy()
            x_worst = params[sorted_indices[-1]].copy()

            new_params = params.copy()

            # Producer update
            if np.random.rand() < ST:
                for i in range(PD):
                    new_params[sorted_indices[i]] = params[sorted_indices[i]] * \
                                                    np.exp(-sorted_indices[i] / (
                                                                np.random.random([1, self.dim]) * self.iterate_max))
            else:
                new_params[sorted_indices[:PD]] = params[sorted_indices[:PD]] * np.random.random([PD, self.dim])

            # Scrounger update
            for i in range(PD, self.N):
                if sorted_indices[i] > self.N / 2:
                    new_params[sorted_indices[i]] = np.random.random([1, self.dim]) * \
                                                    np.exp((x_worst - params[sorted_indices[i]]) / (
                                                                sorted_indices[i] ** 2))
                else:
                    A = np.array([np.random.choice([1, -1]) for _ in range(self.dim)])
                    A_inv = np.dot(A.T, 1 / np.dot(A, A.T))
                    new_params[sorted_indices[i]] = x_best + \
                                                    np.abs(params[sorted_indices[i]] - x_best) * A_inv

            # Scout update
            for i in range(SD):
                f_i = fitness_function(new_params[i])
                if f_i > best_score:
                    new_params[i] = x_best + \
                                    np.random.random([1, self.dim]) * np.abs(new_params[i] - x_best)
                elif f_i == best_score:
                    k = -1 + np.random.random() * 2
                    new_params[i] = new_params[i] + k * \
                                    (np.abs(new_params[i] - x_worst) / (
                                                f_i - fitness_scores[sorted_indices[-1]] + varepsilon))

            # Position boundary check
            new_params = np.clip(new_params, self.x_min, self.x_max)

            # Update fitness
            for i in range(self.N):
                new_score = fitness_function(new_params[i])
                if new_score < fitness_scores[i]:
                    params[i] = new_params[i].copy()
                    fitness_scores[i] = new_score

            iterate += 1
            print(f"Iteration {iterate} completed, Best Score: {best_score}")
            iteration_data.append({'Iteration': iterate + 1, 'Best Score': best_score})
            self.log_iteration(iterate, best_score)

        # Save iteration number and scores to CSV
        iteration_df = pd.DataFrame(iteration_data)
        iteration_df.to_csv('ssa_iteration.csv', index=False)

        return best_score, best_params
