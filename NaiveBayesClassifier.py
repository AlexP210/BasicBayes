class NaiveBayesClassifier():

    def __init__(self, config_dict={}, classification_categories=[]):
        self.estimators = config_dict
        self.classification_categories = classification_categories
        self.priors = Parameter("Prior Distribution", "Prior Probabilities for Classes").categorical_bin(classification_categories)

    def train(self, features, labels):
        for column in features:
            data = features[column]
            # First, see if the user specified the estimator to use. If so, fit it.
            if column in self.estimators:
                self.estimators[column].fit(data)
            # If none specified, use gaussian for numeric types.
            elif utils.isnumeric(data):
                mean = np.mean(data)
                stdev = np.std(data)
                standardized = (data - mean)/stdev
                mu = Parameter("Mean", f"Mean for Standardized {column}").bin(min(transformed), max(transformed), 50)
                sigma = Parameter("Standard Deviation", f"Standard Devaition for Standardized {column}").bin(0, max(transformed) - min(transformed), 50, skip_first = True)
                self.estimators[column] = Estimator(function=gaussian, parameters=[mu, sigma]).fit(data)
            # If its not numeric, then it's categorical, and we can use a multinomial distribution
            # else:
            #     parameters = []
            #     histogram = utils.get_histogram(data)
            #     for key in histogram.keys():
            #         probability = Parameter("Multinomial Probability", f"Probability of {val} in {column}").bin(0, 1, 20)
            #     self.estimators[column] = Estimator(function=utils.multinomial, parameters=parameters).fit(histogram)
        # Now, get the priors for the labels - not as a parameter
        if self.classification_categories == None: self.classification_categories = set(labels)
        self.priors.categorical_bin(classification_categories)
        self.priors.set_probabilities(function=lambda x: count(x, labels)/len(labels))

    def classify(self, feature_vector, plot=False):
        posteriors = Parameter("Posteriors", "Posterior PDF for Class of Data Point").categorical_bin(self.classification_categories)
        max_posterior = -float("inf")
        for category_idx in range(len(self.priors.bins)):
            p = self.priors[key] * self.estimators[key](feature_vector[key])
            posteriors.probabilities[category_idx] = p
            s += p
        posteriors.probabilities /= s
        return posteriors.mode()