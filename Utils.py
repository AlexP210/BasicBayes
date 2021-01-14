import numpy as np

def isiterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def isnumeric(obj):
    try:
        obj + 1
    except TypeError:
        return False
    return True

def discretize(start, stop, step=None, num_steps=None, skip_first=False):
    # Check the inputs
    if step == None and num_steps == None: raise NoStepInfo
    if start > stop: raise InvalidStartStop

    # Get the step info
    if num_steps == None and step != None:
        num_steps = int((stop - start)/step)
    if num_steps != None and step == None:
        step = (stop - start)/num_steps
    if not skip_first:
        return np.linspace(start, stop, num=num_steps)
    if skip_first:
        return np.linspace(start + step, stop, num=num_steps)

def get_histogram(data_series, categories=[]):
    hist = {}
    if categories == []: categories = set(data_series)
    for val in categories:
        hist[val] = sum([observation == val for observation in data_series])
    return hist

def max_index(L):
    m = 0
    m_idx = 0
    for i in range(len(L)):
        if L[i] > m:
            m = L[i]
            m_idx = i
    return m_idx