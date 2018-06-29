'''
Annealing schedules
'''

class Annealing(object):
    '''
    Annealing schedule
    '''
    def get_coeff(self, relative_time):
        raise NotImplementedError


class LinearAnneal(Annealing):
    def __init__(self, *interpolants):
        self.interpolants = interpolants
        self.nb_segments = len(self.interpolants) - 1
    def get_coeff(self, relative_time):
        seg_num = int(relative_time * self.nb_segments)
        i_start = max(0, min(seg_num, self.nb_segments-1))
        alpha = relative_time * self.nb_segments - i_start
        return self.interpolants[i_start] * (1-alpha) + self.interpolants[i_start+1] * alpha

