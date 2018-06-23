import os
import torch as th
import numpy as np
import random
import tensorboardX

from algorithms.models import ActorNet
from nets import make_net
from dynamics import Data
from generators import inf_range
from algorithms.normalization import Stats


class PPO(object):
    def __init__(
            self, env,
            hidden_layer_size=16,
            nb_layers=1, nb_critic_layers=None,
            eps=0.2,
            init_lr=0.01,
            annealing_rate=0.99,
            gamma=0.95,
            gae_lambda=0.2,
            running_norm=False,
            anneal_eps=True,
            writer=None, render=False):
        '''
        @param env: the environment (Gym env)
        @param hidden_layer_size: size of each hidden layer to be used in actor and critic networks
        @param nb_layers: number of hidden layers to use for actor
            (and for critic if `nb_critic_layers` is not specified)
        @param nb_critic_layers: if specified, only changes the number of layers for critic net
        @param critic_layers: hidden layers to use for the critic network (list of integers)
        @param eps: the ϵ (epsilon) parameter for clipping the "surrogate" objective in PPO
        @param init_lr: the initial learning rate
        @param annealing_rate: rate by which the lr is annealed: lr = init_lr * annealing_rate ** epoch
        @param gamma: discount factor parameter (scalar)
        @param gae_lambda: GAE λ parameter (scalar)
        @param running_norm: whether or not to use a running average for normalization
        @param anneal_eps: whether or not to anneal the clipping ϵ parameter over time
        @param writer: writer for logging training curves and images(tensorboardX.SummaryWriter)
        @param render: whehter to call the `env.render(mode='human')` or not. If yes, done at every step (bool)
        '''
        self.eps = eps
        self.env = env
        self.gamma = gamma
        self.init_lr = init_lr
        self.annealing_rate = annealing_rate
        self.anneal_eps = anneal_eps
        self.running_norm = running_norm
        self.gae_lambda = gae_lambda

        self.render = render
        self.writer = writer

        self.save_path = None
        if self.writer is not None:
            self.save_path = os.path.join(self.writer.file_writer.get_logdir(), 'models')

        if nb_critic_layers is None:
            nb_critic_layers = nb_layers

        self.actor = ActorNet(env, [hidden_layer_size] * nb_layers, log_std_noise=-1)

        self.critic = make_net(
            [env.observation_space.shape[0]] +
            [hidden_layer_size] * nb_critic_layers +
            [1],
            [th.nn.ReLU() for _ in range(nb_critic_layers)]
        )

        self.init_optims()

        self.norm_state = Stats(env.observation_space.shape[0])
        # can only scale the reward
        self.norm_rew = Stats(1, shift_mean=False, scale=True, clip=False)

    def init_optims(self):
        self.actor_optim = th.optim.SGD(self.actor.net.parameters(), lr=self.init_lr)
        self.critic_optim = th.optim.SGD(self.critic.parameters(), lr=self.init_lr)
        
        self.scheduler = MultiOptimScheduler(
            th.optim.lr_scheduler.LambdaLR,
            [self.actor_optim, self.critic_optim],
            lr_lambda=lambda epoch: self.annealing_rate ** epoch
        )
    
    def _value_function(self, s):
        '''
        Helper function that evaluates the value function at a given state.
        '''
        return float(self.critic(th.FloatTensor(s)))

    def train(self, nb_iters, nb_max_steps, nb_updates, batch_size):

        for i_iter in range(nb_iters):
            print('\rIteration: %d' % (i_iter+1), end='')

            mem = ReplayMemory(gamma=self.gamma, gae_lambda=self.gae_lambda)

            # TODO: for GAE and TD(lambda) may need to assume that mem stores a single episode
            # and requires all the observations to be added sequentially and in order
            # TODO: fix name
            self.sample_episode(nb_max_steps, mem, i_iter)

            old_actor = self.actor.make_copy()
            lc = 0
            # la = 0
            sum_v = 0
            sum_vhat = 0
            # sum_adv = 0
            mem.to_tensor()
            mem['adv'] = (mem['adv'] - mem['adv'].mean()) / mem['adv'].std()
            self.scheduler.step()
            for _ in range(nb_updates):
                # batch = mem.sample(batch_size)

                # TODO: better logger

                lcn, v, vhat = self.update_critic(mem, batch_size)
                lc += lcn
                sum_v += v
                sum_vhat += vhat

                self.update_actor(mem, old_actor, batch_size)
                # la += lan
                # sum_adv += adv

            if self.writer:
                self.writer.add_scalar('Train/LossCritic', lc / nb_updates, i_iter)
                # self.writer.add_scalar('Train/LossActor', la / nb_updates, i_iter)
                self.writer.add_scalar('Extra/Value/Next', sum_v / nb_updates, i_iter)
                self.writer.add_scalar('Extra/Value/Current', sum_vhat / nb_updates, i_iter)
                # self.writer.add_scalar('Extra/Action/Adv', mem['adv'].mean(), i_iter)

            if callable(getattr(self.env, 'visualize_solution')):
                self.env.visualize_solution(
                    policy=lambda s: self.actor.forward(th.FloatTensor(s)).detach(),
                    value_func=self._value_function,
                    i_iter=i_iter
                )

            # save the models. we do it 10 times in the whole training cycle
            if self._time_to_save(i_iter, nb_iters):
                self.save_models(self.save_path, index=i_iter)
    

    def _time_to_save(self, i_iter, nb_iters):
        '''
        (Private to PPO.train)
        Gives the signal for saving the model
        Does it exactly `nb_save_instances` times in the whole training
        '''
        nb_save_instances = 10
        return self.save_path and (nb_iters - i_iter - 1) % int(nb_iters / nb_save_instances) == 0


    def sample_normalization(self, nb_steps):
        '''
        Samples data for normalization.
        Usually used for initial normalization, but the values can still change if `running_norm=True`
        '''
        done = True  # force reset
        
        for _ in range(nb_steps):
            if done:
                state = self.env.reset()
                self.norm_state.observe(state)

            state_p, rew, done, _ = self.env.step(self.env.action_space.sample())

            self.norm_state.observe(state_p)
            self.norm_rew.observe([rew])

            state = state_p
            

    def sample_episode(self, nb_min_steps, mem, i_episode):
        '''
        Samples a certain number of steps from the environment. Always resets at the start.
        TODO fix name: doesn't just sample a single episode, it sample `nb_min_steps` steps now!
        TODO fix outputs: I'm currently outputing multiple values for logging, but it's not the correct way to do it
        '''
        # maybe use an env wrapper?
        total_rew = 0
        acts = []

        done = True  # force reset
        # first = True

        for i_step in inf_range():
            if done:
                mem.calc_episode_targets(self._value_function)
                _state = self.env.reset()
                nstate = self.norm_state.normalize(_state)
                # if not first:
                #     print('')
                # else:
                #     first = False
                if self.running_norm:
                    self.norm_state.observe(_state)

                if i_step > nb_min_steps:
                    break

            if self.render:
                self.env.render(mode='human')

            act = self.actor.get_action(th.FloatTensor(nstate), explore=True).detach().numpy()
            acts.append(act)

            _state_p, _rew, done, _ = self.env.step(act)
            # print('\r%5.2f' % _rew, end='')

            nstate_p = self.norm_state.normalize(_state_p)
            nrew = self.norm_rew.normalize([_rew])[0]
            mem.record(nstate, act, nrew, nstate_p)

            if self.running_norm:
                self.norm_state.observe(_state_p)
                self.norm_rew.observe([_rew])

            total_rew += _rew
            nstate = nstate_p
        
        # mem.calc_episode_rewards()
        # print('')
        
        if self.writer:
            self.writer.add_scalar('Train/AvgReward', float(total_rew) / (i_step+1), i_episode)
            self.writer.add_scalar('Extra/Action/Avg', float(np.array(acts).mean()), i_episode)
            self.writer.add_scalar('Extra/Action/Std', float(np.array(acts).std()), i_episode)
    

    def update_critic(self, mem, batch_size):
        loss = 0
        sum_v = 0
        sum_vhat = 0
        sum_loss = 0
        criterion = th.nn.MSELoss()
        for batch in mem.iterate_once(batch_size):
            # mem.calculate_td(sample)
            # loss += sample.td()
            # TODO: use TD(lambda) later
            # FIXME: hmm, I'm using an updated version of `critic` every time. is that alright?
            # v = sample.cr #sample.r + self.gamma * self.critic(sample.ns)
            self.critic.zero_grad()  # better way?
            v = batch['creward'].unsqueeze(1)
            vhat = self.critic(batch['state'])
            loss = criterion(vhat, v)
            # print(loss)
            loss.backward()
            self.critic_optim.step()
            # print('backed!')

            sum_v += float(v.sum())
            sum_vhat += float(vhat.sum())
            sum_loss += float(loss.sum())

        return sum_loss / mem.size(), sum_v / mem.size(), sum_vhat / mem.size()
    
    def update_actor(self, mem, old_policy, batch_size):
        total_loss = 0
        # sum_adv = 0

        eps = self.eps
        if self.anneal_eps:
            # print(eps, self.scheduler.get_lr())
            eps *= self.scheduler.get_lr()[0]

        for batch in mem.iterate_once(batch_size):
            self.actor.zero_grad()
            # FIXME: hmm, I'm using an updated version of `critic` every time. is that alright?
            # # never mind for now, not using TD style update
            # adv = sample.r + self.gamma * self.critic(sample.ns) - self.critic(sample.s)
            # batch['adv']
            # adv = sample.cr - self.critic(sample.s)
            # print(batch['action'].shape)
            ratio = self.actor.get_nlog_prob(batch['action'], batch['state']) - old_policy.get_nlog_prob(batch['action'], batch['state'])
            # print(ratio)
            ratio = ratio.exp()
            losses = th.min(
                ratio * batch['adv'],
                th.clamp(ratio, 1 - eps, 1 + eps) * batch['adv']
            )
            loss = -1 * losses.mean()  # we can only do gradient descent not **ascent**
            loss.backward()
            self.actor_optim.step()
            total_loss += losses.sum()
            # sum_adv += adv
        # loss /= -1 * len(batch)  
        return float(loss) / mem.size()

    def save_models(self, path, index=''):
        os.makedirs(path, exist_ok=True)
        save_obj = dict(
            actor=self.actor.net.state_dict(),
            critic=self.critic.state_dict(),
            norm_rew=self.norm_rew,
            norm_state=self.norm_state
        )
        th.save(save_obj, os.path.join(path, '%s-ppo.pt' % str(index)))
        # self.actor.save_model(os.path.join(path, '%s-actor.pt' % str(index)))
        # th.save(self.critic, os.path.join(path, '%s-critic.pt' % str(index)))

    def load_models(self, path, actor=True, critic=True, norm=True):
        loaded_obj = th.load(path)
        
        if actor:
            self.actor.net.load_state_dict(loaded_obj['actor'])
        if critic:
            self.critic.load_state_dict(loaded_obj['critic'])
        if norm:
            self.norm_rew = loaded_obj['norm_rew']
            self.norm_state = loaded_obj['norm_state']
        # self.actor.load_model(os.path.join(path, '%s.actor' % str(index)))
        # self.critic.load_state_dict(th.load(os.path.join(path, '%s.critic' % str(index))))
        self.init_optims()


# # TODO: merge with dynamics.Data
# class ReplayMemory(object):
#     def __init__(self, gamma, gae_lambda):
#         self.data = []
#         self.gamma = gamma
#         self.gae_lambda = gae_lambda
#         self.episode_start_index = 0

#     def size(self):
#         return len(self.data)

#     def record(self, s, a, r, ns):
#         self.data.append(ObservedTuple(s, a, r, ns))

#     def calc_episode_rewards(self, vfunc):
#         # rew = 0
#         # TODO: compute advantages and gae ....
#         # TODO: compute TD(λ)
#         vnext = vfunc(self.data[-1].ns)
#         target = 0
#         for t in reversed(self.data[self.episode_start_index:-1]):
#             vcurr = vfunc(t.s)
#             t.set_adv(t.r + self.gamma * vnext - vcurr)
#             target = t.adv + target * self.gamma * self.gae_lambda
#             t.set_target(target)
#             # rew = t.r + self.gamma * rew
#             # t.set_cum_rew(rew)

#         self.episode_start_index = len(self.data)
#         # if len(self.data):
#         #     print('\nminmax')
#         #     print(np.min([t.cr for t in self.data]))
#         #     print(np.max([t.cr for t in self.data]))
#             # print(np.min([t.s.numpy() for t in self.data], 0))
#             # print(np.max([t.s.numpy() for t in self.data], 0))

#     # def extend(self, other):
#     #     for k in self.key_names:
#     #         self[k].extend(other[k])
#     #     self._size += other.size()

#     def sample(self, sample_size=None):
#         if sample_size is None or sample_size > self.size():
#             sample_size = self.size()
#         return random.sample(self.data, sample_size)

#     # def get_all(self):
#     #     return self.data


class ReplayMemory(Data):
    def __init__(self, gamma, gae_lambda):
        super().__init__(['state', 'action', 'reward', 'nstate', 'td', 'adv', 'creward'])
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.episode_start_index = 0
        self.tensored = False

    def record(self, s, a, r, ns):
        super().add_point(s, a, r, ns, 0, 0, 0)

    def calc_episode_targets(self, vfunc):
        # TODO: compute TD(λ)
        if self.size() == 0:
            return
        crew = 0
        adv = 0
        vpred = vfunc(self['nstate'][-1])
        for i in range(self.size()-1, self.episode_start_index, -1):
            crew = self['reward'][i] + crew * self.gamma
            
            vnext = vpred
            vpred = vfunc(self['state'][i])
            td_delta = self['reward'][i] + self.gamma * vnext - vpred
            adv = td_delta + adv * self.gamma * self.gae_lambda

            if i == self.size()-1:  # correcting for the last episode
                adv *= 1-self.gae_lambda

            self['td'][i] = td_delta
            self['adv'][i] = adv
            self['creward'][i] = crew
            # self['vpred'][i] = vpred
            # self['vnext'][i] = vnext

        self.episode_start_index = self.size()
        # if len(self.data):
        #     print('\nminmax')
        #     print(np.min([t.cr for t in self.data]))
        #     print(np.max([t.cr for t in self.data]))
            # print(np.min([t.s.numpy() for t in self.data], 0))
            # print(np.max([t.s.numpy() for t in self.data], 0))
    
    def to_tensor(self):
        if self.tensored:
            return

        self.tensored = True
        for k in self.key_names:
            if isinstance(self[k][0], th.Tensor):
                self[k] = th.stack(self[k])
            else:
                self[k] = th.FloatTensor(self[k])
        # [th.FloatTensor(self[k]) for k in self.key_names]
    
    def iterate_once(self, batch_size, shuffle=True):
        '''
        Iterates over the data once in chunks of size `batch_size`
        Can optionally shuffle the data as well
        '''
        if shuffle:
            order = th.randperm(self.size())
        else:
            order = list(range(self.size()))

        self.to_tensor()

        for i in range(0, self.size(), batch_size):
            yield {k: self[k][order[i:i+batch_size]] for k in self.key_names}


class ObservedTuple(object):
    def __init__(self, s, a, r, ns):
        self.s = th.FloatTensor(s)
        self.a = th.FloatTensor(a)
        # self.r = r.float()
        self.r = r
        self.ns = th.FloatTensor(ns)
    
    def set_cum_rew(self, cum_reward):
        self.cr = cum_reward

# if __name__ == '__main__':
#     # test PPO with the Pendulum
#     import gym
#     from algorithms.senv import PointMass
    
#     writer = tensorboardX.SummaryWriter()
#     # env = gym.make('Pendulum-v0')
#     env = PointMass(randomize_goal=True, writer=writer, max_steps=100)

#     # ppo = PPO(env, gamma=0.9, hidden_layers=[4], writer=writer, running_norm=True, render=False)
#     ppo = PPO(
#         env, gamma=0.9, running_norm=True,
#         critic_layers=[8], actor_layers=[],
#         render=False, writer=writer,
#     )
#     ppo.sample_normalization(1000)
#     print(ppo.norm_rew.mean, ppo.norm_rew.std)
#     print(ppo.norm_state.mean, ppo.norm_state.std)
#     print(ppo.actor.net)
#     print(ppo.critic)
#     try:
#         # TODO: fix the name ov nb_iters, since it's not the nb_iters anymore
#         ppo.train(nb_iters=400, nb_max_steps=1000, nb_updates=20, batch_size=512)
#     finally:
#         env.close()


# th.optim.lr_scheduler
class MultiOptimScheduler(object):
    def __init__(self, scheduler_class, optims, *args, **kwargs):
        self.schedulers = [
            scheduler_class(optim, *args, **kwargs) for optim in optims
        ]
    
    def get_lr(self):
        return self.schedulers[0].get_lr()

    def step(self, epoch=None):
        for sched in self.schedulers:
            sched.step(epoch)
    
    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return self.__dict__

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)