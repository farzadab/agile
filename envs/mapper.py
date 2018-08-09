from core.object_utils import ObjectWrapper
import copy
import gym

from algorithms.senv import PointMass, PointMassV2, NStepPointMass, CircularPointMass, \
                            CircularPointMassSAG, CircularPhaseSAG, CircularPhaseSAG2, \
                            LinePhaseSAG, SmoothLinePhaseSAG, LinePhaseSAG_NC, SquarePhaseSAG, \
                            SquarePhaseSAG_NC, StepsPhaseSAG, StepsPhaseSAG_NC, PhaseRN1, \
                            PhaseRN1_NC, PhaseRN2, PhaseRN2_NC

from envs.pmfollow import PMFollow, PMFollow1, PMFollow4, PMFollow8\
                          PMFollowIceMid, PMFollowIceMid1, PMFollowIceMid4, PMFollowIceMid8


class MultiStepEnv(ObjectWrapper):
    def __init__(self, env, nb_steps=5):
        super().__init__(env)
        self.__wrapped__ = env
        self.__rendering__ = True
        self.nb_steps = nb_steps
    
    def step(self, action):
        total_reward = 0
        for i in range(self.nb_steps):
            obs, reward, done, extra = self.__wrapped__.step(action)
            total_reward += reward

            if done:
                break

        return obs, total_reward, done, extra


_ENV_MAP = dict(
    PointMass=PointMass, PM=PointMass,
    PM2=PointMassV2,
    NSPM=NStepPointMass,
    CircularPointMass=CircularPointMass, CPM=CircularPointMass,
    CircularPointSAG=CircularPointMassSAG, CPSAG=CircularPointMassSAG,
    CircularPhaseSAG=CircularPhaseSAG, CPhase=CircularPhaseSAG,
    CPhase2=CircularPhaseSAG2,
    LPhase1=LinePhaseSAG,
    LPhase2=SmoothLinePhaseSAG,
    LPhase3=LinePhaseSAG_NC,
    SqPhase1=SquarePhaseSAG,
    SqPhase2=SquarePhaseSAG_NC,
    StepsPhase1=StepsPhaseSAG,
    StepsPhase2=StepsPhaseSAG_NC,
    PhaseRN1=PhaseRN1,
    PhaseRN1_NC=PhaseRN1_NC,
    PhaseRN2=PhaseRN2,
    PhaseRN2_NC=PhaseRN2_NC,
    # TPhase=TriPhaseSAG,
    PMFollow1=PMFollow1,
    PMFollow2=PMFollow,
    PMFollow4=PMFollow4,
    PMFollow8=PMFollow8,
    PMFollowIceMid1=PMFollowIceMid1,
    PMFollowIceMid2=PMFollowIceMid,
    PMFollowIceMid4=PMFollowIceMid4,
    PMFollowIceMid8=PMFollowIceMid8,
)


class SerializableEnv(ObjectWrapper):
    def __init__(self, **kwargs):
        super().__init__(None)
        self.__params = kwargs
        self.__setstate__(kwargs)

    def __getstate__(self):
        state = copy.copy(self.__params)
        # FIXME: not saving this for now, but may change that later
        state['writer'] = None
        return state
    
    def __setstate__(self, state):
        self.__params = state
        env = SerializableEnv._get_env(**self.__params)
        super().set_wrapped(env)

    @staticmethod
    def _get_env(name, multi_step=None, **kwargs):
        if name in _ENV_MAP:
            env = _ENV_MAP[name](**kwargs)
        else:
            env = gym.make(name)
        if multi_step:
            env = MultiStepEnv(env, nb_steps=multi_step)
        return env
