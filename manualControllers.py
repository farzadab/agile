from controllers import Controller


class KeyframeController(Controller):
    def __init__(self, env, keyframes, steps_per_frame=1):
        super().__init__(env)
        self.keyframes = keyframes
        self.steps_per_frame = steps_per_frame
        self.reset()

        # # check action size
        # klens = [len(k) for k in keyframes]
        # if max(klens) != min(klens) or min(klens) != env.action_space.shape[0]:
        #     raise ValueError('The size of each keyframe must be equal to the size of the action space, i.e. `env.action_space.shape[0]`')
    
    def reset(self):
        self.i_step = -1
    
    def get_action(self, state):
        self.i_step += 1
        kf = self.keyframes[int(self.i_step / self.steps_per_frame) % len(self.keyframes)]
        if self.nb_actions() == len(kf):
            return kf
        # convert NabiRos to PDCrab2d
        return kf[0:2] + [0] + kf[2:4] + [0]

class FixedLeftController(Controller):
    def __init__(self, env, keyframes, steps_per_frame=1):
        self.reset()

        # # check action size
        # klens = [len(k) for k in keyframes]
        # if max(klens) != min(klens) or min(klens) != env.action_space.shape[0]:
        #     raise ValueError('The size of each keyframe must be equal to the size of the action space, i.e. `env.action_space.shape[0]`')
    
    def reset(self):
        self.i_step = -1
    
    def get_action(self, state):
        self.i_step += 1
        kf = self.keyframes[int(self.i_step / self.steps_per_frame) % len(self.keyframes)]
        if self.nb_actions() == len(kf):
            return kf
        # convert NabiRos to PDCrab2d
        return kf[0:2] + [0] + kf[2:4] + [0]

def getKeyframeController(steps_per_frame, *keyframes):
    return lambda env: KeyframeController(env, keyframes, steps_per_frame)    


leanLeftRightController = getKeyframeController(
    12,
    # [0, 0, 0, 0 , 0, 0],
    # [1, 0, 0, 1, 0, 0],
    # [1, 1, 1, 1, 1, 1],
    # [-1, 0, 0, -1, 0, 0],
    # [0, 0, .5, 0, 0, -.5],  # lean left
    # [0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0],
    # [0, 0, -.5, 0, 0, .5],  # lean right
)

leanLeftRightController = getKeyframeController(
    12,
    [1, -1, 1, -1],  # tall stance
    [0, 0, 0, 0],  # stance
    [1, -1, 1, -1],  # tall stance
    [0, 0, 0, 0],  # stance
    [1, -1, 1, -1],  # tall stance
    # [0, 0, 0, 0],  # stance
    [1, -1, 1, -1],  # tall stance
    # [1, -1, 1, -1],  # tall stance
    # [1, -1, 1, -1],  # tall stance


    [1, -1, .9, -1],  # lean left
    [1, -1, .8, -1],  # lean left
    [1, -1, .7, -1],  # lean left
    [1, -1, .6, -1],  # lean left
    [1, -1, .5, -1],  # lean left
    [1, -1, .3, -1],  # lean left
    [1, -1, .3, -.5],  # lean left
    [1, -1, .3, 0],  # lean left
    [0.3, -1, .9, -.7],  # lean left
    [0.3, -1, .9, -1],  # lean left
    # [1, -1, .9, -1],  # lean left
    # [1, -1, .7, -.5],  # lean left
    # [1, -1, 1, -1],  # tall stance

    [-1, -1, 1, -1],  # tall stance

    [1, -1, 1, -1],  # tall stance
    [1, -1, 1, -1],  # tall stance
    [1, -1, 1, -1],  # tall stance


    [1, -1, 1, -1],  # tall stance
    [1, -1, .5, -.5],  # lean left
    [1, -1, .5, -.5],  # lean left
    [1, -1, 1, -1],  # tall stance
    [1, -1, 1, -1],  # tall stance
    [1, -1, .5, -.5],  # lean left
    [1, -1, .5, -.5],  # lean left
    [1, -1, 1, -1],  # tall stance
    [1, -1, 1, -1],  # tall stance


    # [1, -1, 1, -1],  # tall stance
    # [0, 0, 0, 0],  # stance
    # [.1, -.5, 0, 0],  # lean left
    # [.3, -.8, 0, 0],  # lean left
    # [0, 0, 0, 0],  # stance
    # [0, 0, 0, 0],  # stance
    # [0, -.15, 0, .15],  # lean left
)

leanLeftRightController = getKeyframeController(
    5,
    # [1, -1, 1, -1],  # tall stance
    # [1, -1, 1, -1],  # tall stance
    # [1, -1, 1, -1],  # tall stance
    [0, 0, 0, 0],  # stance
    [0, 0, 0, 0],  # stance
    [0, 0, 0, 0],  # stance
    [0, 0, 0, 0],  # stance
    [-1, 0, -1, 0],  # stance
    [1, 0, 1, 0],  # stance
    [-1, 0, -1, 0],  # stance
    [1, 0, 1, 0],  # stance
    [-1, 0, -1, 0],  # stance
    [1, 0, 1, 0],  # stance
    [-1, 0, -1, 0],  # stance
    [1, 0, 1, 0],  # stance
)

leanLeftRightController = getKeyframeController(
    12,
    # [1, -1, 1, -1],  # tall stance
    # [1, -1, 1, -1],  # tall stance
    # [1, -1, 1, -1],  # tall stance
    # [0, 0, 0, 0],  # stance
    # [0, 0, 0, 0],  # stance
    # [0, 0, 0, 0],  # stance
    # [0, 0, 0, 0],  # stance
    [-1, 0, -1, 0],  # stance
    [1, -1, 1, -1],  # stance
    [-1, 0, -1, 0],  # stance
    [1, -1, 1, -1],  # stance
    [-1, 0, -1, 0],  # stance
    [1, -1, 1, -1],  # stance
    [-1, 0, -1, 0],  # stance
    [1, -1, 1, -1],  # stance
)


# leanLeftRightController = getKeyframeController(
#     6,
#     [0, 0, 0, 0],
#     [1, -1, 1, -1],  # tall stance
#     [0, 0, 0, 0],  # stance
#     [1, -1, 1, -1],  # tall stance
#     [0, 0, 0, 0],  # stance
#     [0, 0, 0, 0],  # stance
    
#     [0, 0, 0, 0],  # stance
#     [.3, 0, -.3, 0],  # stance
#     # [.3, 0, -.3, 0],  # stance
    
#     [-.3, .2, .3, -1],  # stance
#     [-.3, .4, .3, -1],  # stance
#     [-.3, .6, .3, -1],  # stance
    
#     [0, 0, 0, 0],  # stance
#     [0, 0, 0, 0],  # stance
#     # [.3, 0, -.3, 0],  # stance
#     # [0, 0, 0, 0],  # stance
#     # [.3, 0, -.3, 0],  # stance

#     # [1, 0, 0, 1, 0, 0],
#     # [1, 1, 1, 1, 1, 1],
#     # [-1, 0, 0, -1, 0, 0],
#     # [0, 0, .5, 0, 0, -.5],  # lean left
#     # [0, 0, 0, 0, 0, 0],
#     # [0, 0, 0, 0, 0, 0],
#     # [0, 0, -.5, 0, 0, .5],  # lean right
# )