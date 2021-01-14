from BasicBayes.Exceptions import *
from BasicBayes.Parameter import Parameter
import math as m
import BasicBayes.LikelihoodFunctions as likelihoods
import BasicBayes.Utils as utils
import numpy as np
import BasicBayes.Options as opt
import random
import scipy.stats as stats

class Estimator():

    def __init__(self, function, parameters, name="Distribution", description="A Distribution"):
        self.function = function
        self.parameters = parameters
        self.name = name
        self.description = description
        # Set the probabilities
        size = tuple( (parameter.bins.size for parameter in self.parameters) )
        self.probabilities = np.ones( size )
        parameter_state = tuple( (0 for _ in self.parameters) )
        initial = True
        while parameter_state != tuple((0 for i in range(len(self.parameters)))) or initial:
            initial = False
            for parameter_idx in range(len(self.parameters)):
                self.probabilities[parameter_state] *= self.parameters[parameter_idx].probabilities[parameter_state[parameter_idx]]
            parameter_state = self._increment_parameter_state(parameter_state)

    def set_probabilities(self, function):
        size = tuple( (parameter.bins.size for parameter in self.parameters) )
        parameter_state = tuple( (0 for _ in self.parameters) )
        initial = True
        while parameter_state != tuple((0 for i in range(len(self.parameters)))) or initial:
            initial = False
            parameter_values = tuple((self.parameters[i].bins[parameter_state[i]] for i in range(len(self.parameters))))
            self.probabilities[parameter_state] = function(*parameter_values)
            parameter_state = self._increment_parameter_state(parameter_state)

    def _increment_parameter_state(self, parameter_state):
        parameter_state = list(parameter_state)
        for i in range(len(parameter_state)):
            if parameter_state[i] < self.parameters[i].bins.size - 1:
                parameter_state[i] += 1
                break
            else:
                parameter_state[i] = 0
        return tuple(parameter_state)

    def _increment_parameter_state_mcmc(self, parameter_state):
        parameter_state = list( [idx for idx in parameter_state] )
        for parameter_idx in range(len(self.parameters)):
            mu = parameter_state[parameter_idx]
            sigma = self.parameters[parameter_idx].bins.size/10
            parameter_state[parameter_idx] = round(stats.truncnorm((0 - mu) / sigma, (self.parameters[parameter_idx].bins.size - 1 - mu) / sigma, loc=mu, scale=sigma).rvs())
        return tuple(parameter_state)

    # Fix this method up later
    def _get_log_likelihood(self, X, parameter_state, n_datapoints):
        parameter_values = tuple((self.parameters[i].bins[parameter_state[i]] for i in range(len(self.parameters))))
        log_likelihood = 0
        for x in random.sample(X, n_datapoints):
            func = self.function(x, *parameter_values)
            if func > 0:
                log_likelihood += m.log(self.function(x, *parameter_values))
            elif func < 0:
                assert False, "Function returned negative probability."
            else:
                log_likelihood = -float("inf")
                break
        return log_likelihood

    def fit(self, X, data_sampling=1, mcmc=False, mcmc_tolerance=0.001):
        """Fits the model object to the series of data points given by X.

        :param X: A set of observations
        :type X: Iterable
        :param mcmc: Whether to apply Markov-Chain Monte-Carlo method, defaults to False
        :type mcmc: bool, optional
        :param data_sampling: The proportion of the dataset to use, defaults to 1.
        :type data_sampling: float, optional
        :return: The model object
        :rtype: Model
        """
        # If X is not a list, make it one
        if type(X) != type([]):
            X = [X,]
        # Set the number of data points from X that we'll use at each point
        n_datapoints = int(data_sampling*len(X))

        # Normal parameter estimation
        if not mcmc:
            # Initialize the parameter state, and the "initial" flag
            parameter_state = tuple( (0 for _ in self.parameters) )
            initial = True
            # Initialize the log likelihoods, and the max log likelihood variables
            size = tuple( (parameter.bins.size for parameter in self.parameters) )
            log_likelihood_space = np.zeros( size )
            max_log_likelihood = -float("inf")
            # Go through each parameter state, and calculate the log of the likelihood, as well as the prior for that state
            while parameter_state != tuple((0 for i in range(len(self.parameters)))) or initial:
                # Set the parameter values
                initial = False
                parameter_values = tuple((self.parameters[i].bins[parameter_state[i]] for i in range(len(self.parameters))))
                # Get the log-likelihood, normalize, and exponentiate
                log_likelihood = 0
                for x in random.sample(X, n_datapoints):
                    func = self.function(x, *parameter_values)
                    if func > 0:
                        log_likelihood += m.log(self.function(x, *parameter_values))
                    elif func < 0:
                        # Shouldn't be here; if so throw an error
                        assert False
                    else:
                        log_likelihood = -float("inf")
                        break
                log_likelihood_space[parameter_state] = log_likelihood
                likelihood = np.exp(log_likelihood)
                if likelihood > max_log_likelihood:
                    max_log_likelihood = likelihood
                parameter_state = self._increment_parameter_state(parameter_state)
            likelihood_space = np.exp(log_likelihood_space - max_log_likelihood)
            # Get the posterior space
            self.probabilities = np.multiply(likelihood_space, self.probabilities)
            self.probabilities /= self.probabilities.sum()
        
        # MCMC parameter estimation
        elif mcmc:
            # Initialize the parameter state (middle of the joint distribution), and the "converged" flag
            parameter_state = tuple( (parameter.bins.size//2 for parameter in self.parameters) )
            converged = False
            # Initialize the MCMC variables
            numerator = m.log(self.probabilities[parameter_state]) + self._get_log_likelihood(X, parameter_state, n_datapoints)
            shape = tuple( (parameter.bins.size for parameter in self.parameters) )
            frequencies = np.zeros(shape)
            last_frequencies = np.zeros(shape)
            # While we're not converged, keep at it
            n_iters = 0
            while not converged:
                # Count the current state
                n_iters += 1
                frequencies[parameter_state] += 1
                # Sample a next state, find the likelihood at that state, and the probability of jump
                next_parameter_state = self._increment_parameter_state_mcmc(parameter_state)
                next_parameter_state_numerator = m.log(self.probabilities[next_parameter_state]) + self._get_log_likelihood(X, next_parameter_state, n_datapoints)
                prob_jump = m.exp(next_parameter_state_numerator - numerator)
                # If we jump, change the necessary variables
                if prob_jump >= 1 or np.random.uniform() < prob_jump:
                    parameter_state = next_parameter_state
                    numerator = next_parameter_state_numerator
                # If we're in a multiple of 1000 iterations, check if we've converged. If not, set last_frequencies to the current
                # For the next time we do this comparison
                # if n_iters%10000 == 0 and abs( (frequencies/frequencies.sum() - last_frequencies/last_frequencies.sum()).sum() ) < mcmc_tolerance:
                #     converged = True
                # else: 
                #     last_frequencies = frequencies
                if n_iters == 5000: converged = True
            self.probabilities = frequencies / frequencies.sum()
        
        # Marginalize across all the dimensions, and set the probabilities for the parameters
        for parameter_idx in range(len(self.parameters)):
            summing_axes = tuple( (i for i in range(len(self.parameters)) if i != parameter_idx) )
            self.parameters[parameter_idx].probabilities = self.probabilities.sum(axis=summing_axes)

        return self



    def __call__(self, x):
        parameter_state = tuple( (0 for _ in self.parameters) )
        initial = True

        # If the argument is a parameter, set that parameter's probabilities
        if type(x) == type(Parameter()):
            probability = np.zeros(x.probabilities.shape)
            for i in range(x.probabilities.size):
                probability[i] = self.__call__(x.bins[i])
            return_parameter = x._copy()
            # NOTE: Should this be a pdf? or just the probabilities?
            return_parameter.probabilities = probability/probability.sum()
            return return_parameter

        # If the argument is an iterable, then output the probabilities for each element
        elif type(x) in (type(()), type([]), type(set()), type(np.empty(0))):
            output = 1
            for e in x:
                output *= self.__call__(e)
            return output

        # If the argument is a single value, return just the probability for that value
        else:
            probability = 0
            while parameter_state != tuple((0 for i in range(len(self.parameters)))) or initial:
                # Set the parameter values
                initial = False
                parameter_values = tuple((self.parameters[i].bins[parameter_state[i]] for i in range(len(self.parameters))))
                likelihood = self.function(x, *parameter_values)
                posterior = 1
                for parameter_idx in range(len(self.parameters)):
                    state_idx = parameter_state[parameter_idx]
                    posterior *= self.parameters[parameter_idx].probabilities[state_idx]
                probability += posterior * likelihood
                parameter_state = self._increment_parameter_state(parameter_state)
            return probability

    def mode(self, rounded=opt.rounded):
        indices = np.where(self.probabilities == self.probabilities.max())
        out = []
        for idx in range(len(self.parameters)):
            out.append( round(self.parameters[idx].bins[indices[idx]][0], rounded) )
        return tuple(out)

    def report(self, probability=0.95, rounded=opt.rounded):
        out = f"Mode of joint PDF is at:"
        mode = self.mode(rounded)
        for parameter_idx in range(len(self.parameters)):
            out += f"\n     {self.parameters[parameter_idx].name} = {mode[parameter_idx]}"
        out += "\nMarginalized PDFs:"
        for parameter_idx in range(len(self.parameters)):
            out += f"\n     {self.parameters[parameter_idx].name} = {self.parameters[parameter_idx].mode()} with {probability*100}% CI {self.parameters[parameter_idx].CI(probability)}"
      
        print(out)

    def __getitem__(*args):
        index = tuple( (self.parameters[parameter_idx]._get_index(args[parameter_idx]) for parameter_idx in range(len(self.parameters))) )
        return self.probabilities[index]

    def parameter_names(self):
        return( (parameter.name for parameter in self.parameters) )

if __name__ == "__main__":
    # Create the parameters and "bin" them to make the hypotheses
    mu = Parameter(name="Mu", description="Gaussian mean")
    mu.bin(min(X), max(X), 100)
    sigma = Parameter(name="Sigma", description="Gaussian standard deviation")
    sigma.bin(0, max(X) - min(X), 100, skip_first=True)

    # Get the likelihood function
    from BasicBayes.LikelihoodFunctions import gaussian

    # Create the estimator, and fit the data
    gaussian_estimator = Estimator(function=gaussian, parameters=[mu, sigma])
    gaussian_estimator.fit(X)
    mu.plot_pdf()
    sigma.plot_pdf()

    # Plot the PDFs, and get the mode
    mode = gaussian_estimator.mode()
    print(f"The best fit gaussian has a mean of {mode[0]} and a standard deviation of {mode[1]}")
