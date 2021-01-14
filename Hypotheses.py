from numpy import array
from numpy import zeros
from BayesPyUtils import *

class GaussianHypothesis():
    
    def __init__(self, mu_hypotheses, sigma_hypotheses, mu_fit = lambda x: x, sigma_fit = lambda x: x):
        """Hypothesized parameters of a gaussian distribution
        
        Arguments:
            mu_hypotheses {array} -- Array of hypothesized means, or parameters for mean fit
            sigma_hypotheses {array} -- Array of hypothesized StDev, or parameters for StDev fit

        Keyword Arguments:
            mu_fit {function} -- Optional: Function to model mean
            sigma_fit {function} -- Optional: Function to model StDev
        """
        self.mu_hypotheses = array(mu_hypotheses)
        self.sigma_hypotheses = array(sigma_hypotheses)
        self.mu_fit = mu_fit
        self.sigma_fit = sigma_fit

    def likelihoods(self, data):
        self.likelihoods = zeros( (len(self.mu_hypotheses), len(self.sigma_hypotheses)) )
        for mu_idx in range(len(self.mu_hypotheses)):
            for sig_idx in range(len(self.sigma.hypotheses)):
                likelihood = 1
                for datum in data:
                    likelihood *= 




