'''
Just a utility module for generators
'''

def inf_range(start=0, step=1):
    '''
    Infinite range starting from `start` and going up with a step size of `step`
    '''
    value = start
    while True:
        yield value
        value += step
