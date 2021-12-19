from LikelihoodFunctions import *

# Lets get the data
import csv
import requests
total_cases = [0,]
N_days = 14
url = 'https://data.ontario.ca/datastore/dump/ed270bb8-340b-41f9-a7c6-e8ef587e6d11?bom=True'
with requests.get(url, stream=True) as r:
    lines = (line.decode('utf-8-sig') for line in r.iter_lines())
    for row in csv.DictReader(lines):
        if row['Total Cases'] != "":
            total_cases.append(int(row["Total Cases"]))
last_10_days = list( ( int(row["Total tests completed in the last day"]), total_cases[i] - total_cases[i-1]) for i in range(1, len(total_cases)))[-N_days:]
X = list(zip(list( [i+1 for i in range(N_days)] ), last_10_days*))

# Set-up the likelihood function
def likelihood(data_point, SIR_new_cases, specificity, sensitivity):
    if data_point[1] >= SIR_new_cases:
        return binomial(data_point[1] - SIR_new_cases, data_point[1], (1-specificity)/(1-specificity+sensitivity))
    elif data_point[1] < SIR_new_cases:
        


# Set-up the fit-function - should take parameters and output statistical arguments for likelihood function
def time_dependent_SIR_model(slope, y_intercept, gamma):
