from numpy import array, full
import numpy as np
import BasicBayes.LikelihoodFunctions as likelihoods
import matplotlib.pyplot as plt
import BasicBayes.Options as opt
import BasicBayes.Utils as utils

class Parameter():

    def __init__(self, name="Parameter", description="A Parameter"):
        self.name = name
        self.description = description
        self.bins = None
        self.steps = None
        self.num_steps = None
        self.probabilities = None
        self._max_index = 0
        self.type = None
    
    def bin(self, start, stop, num_steps = None, step = None, skip_first=False):
        self.bins = utils.discretize(start, stop, step, num_steps, skip_first)
        self.probabilities = np.full((self.bins.size),1/self.bins.size)
        self.num_steps = self.bins.size
        self.steps = (stop - start)/self.num_steps
        self.type = "Numeric"
        return self

    def categorical_bin(self, categories):
        self.bins = np.array(categories)
        self.probabilities = np.full((self.bins.size),1/self.bins.size)
        self.type = "Categorical"
        return self

    def plot_pdf(self):
        if self.type == "Numeric": plt.plot(self.bins, self.probabilities/self.steps)
        elif self.type == "Categorical": plt.bar(self.bins, self.probabilities)
        plt.xlabel(self.description)
        plt.ylabel("Probability Density")
        plt.title(f"PDF for {self.name}")
        plt.show()

    def mode(self, rounded=opt.rounded):
        self._max_index = utils.max_index(self.probabilities)
        return round(self.bins[self._max_index], rounded)

    def CI(self, probability=0.95, rounded=opt.rounded):

        pair = zip(self.probabilities, range(self.bins.size))
        mode_idx = max(pair, key = lambda t: t[0])[1]
        
        total_prob = self.probabilities[mode_idx]
        upper_idx = mode_idx
        lower_idx = mode_idx
        widened = True
        while widened:
            widened = False
            if (upper_idx + 1 < self.bins.size) and (total_prob + self.probabilities[upper_idx + 1] < probability):
                total_prob += self.probabilities[upper_idx + 1]
                upper_idx += 1
                widened = True
            if (lower_idx - 1 >= 0) and (total_prob + self.probabilities[lower_idx - 1] < probability):
                total_prob += self.probabilities[lower_idx - 1]
                lower_idx -= 1
                widened = True

        upper = self.bins[upper_idx]
        lower = self.bins[lower_idx]
        
        return(round(lower, rounded), round(upper, rounded))

    def set_probabilities(self, function):
        for bin_idx in range(self.bins.size):
            self.probabilities[bin_idx] = function(self.bins[bin_idx])
        return self

    def _copy(self):
        x = Parameter(name=self.name, description=self.description)
        x.bins = self.bins
        x.steps = self.steps
        x.num_steps = self.num_steps
        x.probabilities = self.probabilities
        x._max_index = self._max_index
        x.type = self.type
        return x

    def __sub__(self, other):
        difference = self._copy()
        if type(other) in (type(1), type(1.0)):
            for i in range(self.bins.size):
                difference.bins[i] = difference.bins[i] - other
        elif type(other) in (type(Parameter())):
            probabilities = np.zeros(self.probabilities.shape)
            bins = np.zeros(self.bins.shape)
            for bin_idx in range(bins.size):
                pass
    
    def __le__(self, other):
        probability = 0
        if type(other) in (type(1), type(1.0)):
            return sum( [self.probabilities[bin_idx] for bin_idx in range(self.bins.size) if self.bins[bin_idx] <= other] )
        elif type(other) in (type(Parameter()),) and other.type == "Continuous" and self.type == "Continuous":
            return sum( [self.probabilities[self_bin_idx]*other.probabilities[other_bin_idx] for self_bin_idx in range(self.bins.size - 1) for other_bin_idx in range(self.bins.size - 1) if self.bins[self_bin_idx] <= other.bins[other_bin_idx]] )
        return probability       

    def __ge__(self, other):
        probability = 0
        if type(other) in (type(1), type(1.0)):
            return sum( [self.probabilities[bin_idx] for bin_idx in range(self.bins.size) if self.bins[bin_idx] >= other] )
        elif type(other) in (type(Parameter()),) and other.type == "Continuous" and self.type == "Continuous":
            return sum( [self.probabilities[self_bin_idx]*other.probabilities[other_bin_idx] for self_bin_idx in range(self.bins.size - 1) for other_bin_idx in range(self.bins.size - 1) if self.bins[self_bin_idx] >= other.bins[other_bin_idx]] )
        return probability       

    def __lt__(self, other):
        probability = 0
        if type(other) in (type(1), type(1.0)):
            return sum( [self.probabilities[bin_idx] for bin_idx in range(self.bins.size) if self.bins[bin_idx] < other] )
        elif type(other) in (type(Parameter()),) and other.type == "Continuous" and self.type == "Continuous":
            return sum( [self.probabilities[self_bin_idx]*other.probabilities[other_bin_idx] for self_bin_idx in range(self.bins.size - 1) for other_bin_idx in range(self.bins.size - 1) if self.bins[self_bin_idx] < other.bins[other_bin_idx]] )
        return probability       

    def __gt__(self, other):
        probability = 0
        if type(other) in (type(1), type(1.0)):
            return sum( [self.probabilities[bin_idx] for bin_idx in range(self.bins.size) if self.bins[bin_idx] > other] )
        elif type(other) in (type(Parameter()),) and other.type == "Continuous" and self.type == "Continuous":
            return sum( [self.probabilities[self_bin_idx]*other.probabilities[other_bin_idx] for self_bin_idx in range(self.bins.size - 1) for other_bin_idx in range(self.bins.size - 1) if self.bins[self_bin_idx] > other.bins[other_bin_idx]] )
        return probability       

    def __eq__(self, other):
        return self.probabilities == other.probabilities

    def __ne__(self, other):
        return not self.__eq__(other)

    def _get_index(self, hypothesis_value):
        if utils.isnumeric(hypothesis_value):
            assert self.type == "Numeric"
            where = np.where((self.bins - hypothesis_value) < self.step/2)
        else:
            assert self.type == "Categorical"
            where = np.where(self.bins == hypothesis_value)
        if len(where[0]) == 0:
            raise KeyError
        return where[0][0]

    def __getitem__(self, key):
        return self.probabilities(self._get_index(key))

if __name__ == "__main__":
    mu = Parameter("Mu", "Mean of Gaussian")
    mu.bin(0, 10, num_steps=100)
    print(mu[0.56])


