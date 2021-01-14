class NoStepInfo(Exception):
    def __repr__(self): return "No Step Information Supplied"

class InvalidStartStop(Exception):
    def __repr__(self): return "Start must be less than start"