'''
Object Utilities
'''

class ObjectWrapper(object):
    '''
    Makes proxy object
    '''
    def __init__(self, wrap_object):
        self.set_wrapped(wrap_object)
    def set_wrapped(self, wrapped):
        self.__wrapped__ = wrapped
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getattr__(self, attr_name):
        if hasattr(self.__wrapped__, attr_name):
            return getattr(self.__wrapped__, attr_name)
        raise AttributeError
