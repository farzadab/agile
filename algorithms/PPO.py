import os
import torch as th
import numpy as np
import random
import tensorboardX
import copy

from algorithms.models import ActorNet
from nets import make_net
from dynamics import Data
from generators import inf_range
from algorithms.normalization import Stats


class PPO(object):
    SERIALIZER_VERSION = 2

    def __init__(
            self, env,
            hidden_layer_size=16,
            nb_layers=1, nb_critic_layers=None,
            eps=0.2,
            init_lr=0.001,
            annealing_rate=0.99,
            gamma=0.99,
            gae_lambda=0.95,
            exploration_noise=-1,
            explore_ratio=1.0,
            exploration_anneal=None,
            anneal_eps=True,
            writer=None, render=False):
        '''
        Implements the Proximal Policy Optimization algorithm (PPO)

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
        @param gae_lambda: the λ parameter in GAE(λ) (scalar)
        @param exploration_noise: the amount of exploration noise (log σ) for the actor (scalar)
        @param explore_ratio: the percentage of time an exploratory action is performed
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
        self.gae_lambda = gae_lambda
        self.exploration_anneal = exploration_anneal
        self.explore_ratio = explore_ratio

        self.render = render
        self.writer = writer

        self.save_path = None
        if self.writer is not None and self.writer.get_logdir() is not None:
            self.save_path = os.path.join(self.writer.get_logdir(), 'models')

        if nb_critic_layers is None:
            nb_critic_layers = nb_layers

        self.actor = ActorNet(env, [hidden_layer_size] * nb_layers, log_std_noise=exploration_noise)

        self.critic = make_net(
            [env.observation_space.shape[0]] +
            [hidden_layer_size] * nb_critic_layers +
            [1],
            [th.nn.ReLU() for _ in range(nb_critic_layers)]
        )

        self.init_optims()

    def init_optims(self):
        self.actor_optim = th.optim.Adam(
            self.actor.net.parameters(), lr=self.init_lr,
            betas=(0.95, 0.999), eps=1e-06)
        self.critic_optim = th.optim.Adam(
            self.critic.parameters(), lr=self.init_lr,
            betas=(0.95, 0.999), eps=1e-06)
        
        self.scheduler = MultiOptimScheduler(
            th.optim.lr_scheduler.LambdaLR,
            [self.actor_optim, self.critic_optim],
            lr_lambda=lambda epoch: self.annealing_rate ** epoch  # exponential decay
        )
    
    def _value_function(self, s):
        '''
        Helper function that evaluates the value function at a given state.
        '''
        return float(self.critic(th.FloatTensor(s)))

    def train(self, nb_iters, batch_size, nb_epochs, mini_batch_size):

        for i_iter in range(nb_iters):
            print('\rIteration: %d' % (i_iter+1), end='')
            self.writer.set_epoch(i_iter)
            if self.exploration_anneal:
                self.actor.set_noise(self.exploration_anneal.get_coeff(i_iter / nb_iters))

            # FIXME: how to do this now?
            # if i_iter > nb_iters / 2:
            #     self.running_norm = False


            # self.norm_state.mean[:] = 0
            # self.norm_state.std[:] = 1
            # print(self.actor.net[-1].state_dict()['weight'])
            # print(self.norm_state.mean)

            mem = ReplayMemory(gamma=self.gamma, gae_lambda=self.gae_lambda)

            # TODO: fix name
            self.sample_episode(batch_size, mem)

            returns = mem.calculate_advantages(self._value_function)
            self.writer.add_scalar('Train/AvgReturn', np.mean(returns))

            # still not sure if this is needed or not
            mem['adv'] = (mem['adv'] - mem['adv'].mean()) / mem['adv'].std()

            # explained_var = 1 - (mem['vpred'] - mem['creward']).var() / mem['creward'].var()
            explained_var = 1 - (mem['vpred'] - mem['vtarg']).var() / mem['vtarg'].var()
            self.writer.add_scalar('Train/ExplVar', explained_var)

            old_actor = self.actor.make_copy()
            self.scheduler.step()

            for _ in range(nb_epochs * 5):
                self.update_critic(mem, mini_batch_size)

            returns = mem.calculate_advantages(self._value_function)
            mem['adv'] = (mem['adv'] - mem['adv'].mean()) / mem['adv'].std()

            for _ in range(nb_epochs):
                self.update_actor(mem, old_actor, mini_batch_size)

            if callable(getattr(self.env, 'visualize_solution', None)):
                self.env.visualize_solution(
                    policy=lambda s: self.actor.forward(th.FloatTensor(s)).detach(),
                    value_func=self._value_function,
                    i_iter=i_iter
                )

            self.save(self.save_path)  # always save the last model
            
            # save the models. we do it 10 times in the whole training cycle
            if self._time_to_save(i_iter, nb_iters):
                self.save(self.save_path, index=i_iter)
    

    def _time_to_save(self, i_iter, nb_iters):
        '''
        (Private to PPO.train)
        Gives the signal for saving the model
        Does it exactly `nb_save_instances` times in the whole training
        '''
        nb_save_instances = 10
        return (nb_iters - i_iter - 1) % int(nb_iters / nb_save_instances) == 0

    # FIXME: now that PPO doesn't do the normalization, it looks meaning less
    def sample_normalization(self, nb_steps):
        '''
        Samples data for normalization.
        Usually used for initial normalization, but the values can still change if `running_norm=True`
        '''
        done = True  # force reset
        
        for _ in range(nb_steps):
            if done:
                _ = self.env.reset()

            _, _, done, _ = self.env.step(self.env.action_space.sample())

    def sample_episode(self, batch_size, mem):
        '''
        Samples a certain number of steps from the environment. Always resets at the start.
        TODO fix name: doesn't just sample a single episode, it sample `batch_size` steps now!
        TODO fix outputs: I'm currently outputing multiple values for logging, but it's not the correct way to do it
        '''
        # maybe use an env wrapper?
        total_rew = 0
        acts = []

        done = True  # force reset
        # first = True
        rews = []
        num_episodes = -1

        termination_types = {'rew': 0, 'torso': 0, 'time': 0}

        for i_step in inf_range():
            if done:
                # ret = mem.calc_episode_targets(self._value_function)
                # print(ret)
                mem.record_new_episode()
                state = self.env.reset()
                num_episodes += 1
                # if not first:
                #     print('')
                # else:
                #     first = False

                if i_step > batch_size:
                    break

            # TODO: use a callback here instead
            if self.render:# and i_step % 100 == 0:
                self.env.render(mode='human')

            explore = (np.random.rand() < self.explore_ratio) or (i_step == 0)
            act = self.actor.get_action(th.FloatTensor(state), explore=explore).detach().numpy()

            acts.append(act)

            state_p, rew, done, extra = self.env.step(act)
            if 'rewards' in extra:
                if isinstance(extra['rewards'], dict):
                    rews.append(list(extra['rewards'].values()))
                else:
                    rews.append(extra['rewards'])
            # print('\r%5.2f' % _rew, end='')

            mem.record(state, act, rew, state_p, explore=explore)

            total_rew += rew
            state = state_p

            if done:
                termination_types[extra.get('termination', 'time')] += 1
        
        if self.writer:
            self.writer.add_scalar('Train/AvgReward', float(total_rew) / (i_step+1))
            self.writer.add_scalar('Train/AvgEpsLen', len(acts) / num_episodes)
            self.writer.add_scalar('Extra/Action/Avg', float(np.array(acts).mean()))
            self.writer.add_scalar('Extra/Action/Std', float(np.array(acts).std()))
            if rews:
                rews_t = np.mean(rews, axis=0)
                for i, rew in enumerate(rews_t):
                    self.writer.add_scalar('Rewards/Avg-%d' % i, rew)
            for k, v in termination_types.items():
                self.writer.add_scalar('Extra/TermT/%s' % k, v / num_episodes)

    

    def update_critic(self, mem, batch_size):
        loss = 0
        mean_v = 0
        mean_vhat = 0
        mean_loss = 0
        criterion = th.nn.MSELoss()
        for batch in mem.iterate_once(batch_size):
            # mem.calculate_td(sample)
            # loss += sample.td()
            # TODO: use TD(lambda) later
            # FIXME: hmm, I'm using an updated version of `critic` every time. is that alright?
            # v = sample.cr #sample.r + self.gamma * self.critic(sample.ns)
            self.critic.zero_grad()  # better way?
            # v = batch['creward'].unsqueeze(1)
            v = batch['vtarg'].unsqueeze(1)
            vhat = self.critic(batch['state'])
            loss = criterion(vhat, v)
            # print(loss)
            loss.backward()
            self.critic_optim.step()
            # print('backed!')

            mean_v += float(v.sum()) / mem.size()
            mean_vhat += float(vhat.sum()) / mem.size()
            mean_loss += float(loss.sum()) / mem.size()
        
        self.writer.add_scalar('Train/LossCritic', mean_loss)
        self.writer.add_scalar('Extra/Value/Target', mean_v)
        self.writer.add_scalar('Extra/Value/Current', mean_vhat)

        return mean_loss
    
    def update_actor(self, mem, old_policy, batch_size):
        mean_loss = 0

        eps = self.eps
        if self.anneal_eps:
            eps *= self.scheduler.get_annealing_factor()

        for batch in mem.iterate_once(batch_size):
            self.actor.zero_grad()
            ratio = self.actor.get_nlog_prob(batch['action'], batch['state']) - old_policy.get_nlog_prob(batch['action'], batch['state'])
            ratio = ratio.exp()
            losses = th.min(
                ratio * batch['adv'],
                th.clamp(ratio, 1 - eps, 1 + eps) * batch['adv']
            )
            loss = -1 * losses.mean()  # we can only do gradient descent not **ascent**
            loss.backward()
            self.actor_optim.step()
            mean_loss += float(losses.sum()) / mem.size()
        
        # self.writer.add_scalar('Train/LossActor', la / nb_updates, i_iter)
        # self.writer.add_scalar('Extra/Action/Adv', mem['adv'].mean(), i_iter)
            
        return mean_loss

    # The legacy (v1.0) version of pickling. Keeping it for future reference
    # def save_models(self, path, index=None):
    #     if path is None:
    #         return
    #     name = 'last' if index is None else str(index)
    #     os.makedirs(path, exist_ok=True)
    #     save_obj = dict(
    #         actor=self.actor.net.state_dict(),
    #         critic=self.critic.state_dict(),
    #         norm_rew=self.norm_rew,
    #         norm_state=self.norm_state
    #     )
    #     th.save(save_obj, os.path.join(path, '%s-ppo.pt' % name))
    #     # self.actor.save_model(os.path.join(path, '%s-actor.pt' % str(index)))
    #     # th.save(self.critic, os.path.join(path, '%s-critic.pt' % str(index)))

    def get_env(self):
        return self.env

    def save(self, path, index=None):
        if path is None:
            return
        name = 'last' if index is None else str(index)
        os.makedirs(path, exist_ok=True)
        th.save(self, os.path.join(path, '%s-ppo.pt' % name))
    
    @staticmethod
    def load(path, index=None):
        # name = 'last' if index is None else str(index)
        # return th.load(os.path.join(path, '%s-ppo.pt' % name))
        # FIXME: use index ....
        return th.load(path)
    
    def __getstate__(self):
        state = copy.copy(self.__dict__)
        # FIXME: how to handle these?
        state['writer'] = None
        state['save_path'] = None
        state['actor_optim'] = None
        state['critic_optim'] = None
        state['scheduler'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.init_optims()

    def __model_loader_v1(self, path, actor=True, critic=True, norm=True):
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
    
    def extract_linear_policy(self):
        '''
        Returns the actual policy, 
        Only works when you have a linear actor
        '''
        policy = self.actor.net.state_dict()
        norm_matrix = th.cat(
            (
                th.cat((th.diag(self.env.norm_stt.std), th.zeros((1, self.env.norm_stt.std.shape[0])))),
                th.cat((self.env.norm_stt.mean, th.FloatTensor([1]))).reshape(-1, 1),
            ),
            1
        )
        actor_matrix = th.cat((policy['0.weight'], policy['0.bias'].reshape(-1, 1)), 1)
        return norm_matrix.mm(actor_matrix.t())


class ReplayMemory(Data):
    def __init__(self, gamma, gae_lambda):
        super().__init__(['state', 'action', 'reward', 'nstate', 'exploratory', 'vpred', 'td', 'adv', 'creward', 'vtarg'])
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.episode_starts = []
        self.tensored = False

    def record(self, s, a, r, ns, explore=False):
        super().add_point(s, a, r, ns, explore, 0, 0, 0, 0, 0)
    
    def record_new_episode(self):
        self.episode_starts.append(self.size())

    def calculate_advantages(self, vfunc):
        '''
        Calculates the advantages and targets for the current episode

        @param vfunc: the value function (aka critic)
        '''
        
        self.to_tensor()

        episode_returns = []
        episode_ends = self.episode_starts[1:] + [self.size()]
        for start, end in zip(self.episode_starts, episode_ends):
            # if self.size() - self.episode_start_index == 0:
            #     return
            crew = 0
            adv = 0
            # TODO fix: this is wrong, either set to 0 or just use it when the agent is still alive ...
            vpred = vfunc(self['nstate'][-1])
            # print(self.episode_start_index, self.size())
            episode_return = 0
            for i in range(end-1, start-1, -1):
                episode_return += self['reward'][i]
                crew = self['reward'][i] + crew * self.gamma
                
                vnext = vpred
                vpred = vfunc(self['state'][i])
                ret_1step  = self['reward'][i] + self.gamma * vnext
                td_delta = ret_1step - vpred
                adv = td_delta + adv * self.gamma * self.gae_lambda

                # if i == self.size()-1:  # correcting for the last episode
                #     adv /= 1-self.gae_lambda

                self['td'][i] = td_delta
                self['adv'][i] = adv
                self['creward'][i] = crew
                self['vtarg'][i] = self['adv'][i] + vpred
                self['vpred'][i] = vpred
                # self['vnext'][i] = vnext

            episode_returns.append(episode_return)
            # lens = max(1, int((self.size() - self.episode_start_index) / 5))
            # print('\n-------------------- Episode length:  %d ---------------------' % (self.size() - self.episode_start_index))
            # td = th.FloatTensor(self['td'][self.episode_start_index:])
            # adv = th.FloatTensor(self['adv'][self.episode_start_index:])
            # vtarg = th.FloatTensor(self['vtarg'][self.episode_start_index:])
            # crew = th.FloatTensor(self['creward'][self.episode_start_index:])
            # print('td   : %7.3f  %7.3f  %7.3f' % ( td[:lens].mean(), td[2*lens:3*lens].mean(), td[-lens:].mean()))
            # print('adv  : %7.3f  %7.3f  %7.3f' % ( adv[:lens].mean(), adv[2*lens:3*lens].mean(), adv[-lens:].mean()))
            # print('vtarg: %7.3f  %7.3f  %7.3f' % ( vtarg[:lens].mean(), vtarg[2*lens:3*lens].mean(), vtarg[-lens:].mean()))
            # print('crew : %7.3f  %7.3f  %7.3f' % ( crew[:lens].mean(), crew[2*lens:3*lens].mean(), crew[-lens:].mean()))

            # print('---------------------------------------------')
            # print(th.FloatTensor(self['td'])[self.episode_start_index:self.episode_start_index+10])
            # print(th.FloatTensor(self['adv'])[self.episode_start_index:self.episode_start_index+10])
            # print(th.FloatTensor(self['reward'])[self.episode_start_index:self.episode_start_index+10])
            # print(th.FloatTensor(self['creward'])[self.episode_start_index:self.episode_start_index+10])
            # print(th.FloatTensor(self['vtarg'])[self.episode_start_index:self.episode_start_index+10])
            # print('...............')
            # print(th.FloatTensor(self['td'])[-10:])
            # print(th.FloatTensor(self['adv'])[-10:])
            # print(th.FloatTensor(self['reward'])[-10:])
            # print(th.FloatTensor(self['creward'])[-10:])
            # print(th.FloatTensor(self['vtarg'])[-10:])

        # print(self.episode_start_index, self.size())
        return episode_returns
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
        self.to_tensor()

        exp_actions = th.tensor([i for i, v in enumerate(self['exploratory']) if v > 0.5], dtype=th.long)
        exp_actions_len = len(exp_actions)

        if shuffle:
            order = th.randperm(exp_actions_len)
        else:
            order = list(range(exp_actions_len))

        order = exp_actions[order]

        for i in range(0, exp_actions_len, batch_size):
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
    def __init__(self, scheduler_class, optims, lr_lambda, *args, **kwargs):
        self.schedulers = [
            scheduler_class(optim, lr_lambda=lr_lambda, *args, **kwargs) for optim in optims
        ]
        self.epochs = -1
        self.lr_lambda = lr_lambda
    
    def get_annealing_factor(self):
        return self.lr_lambda(self.epochs)

    def step(self, epoch=None):
        self.epochs += 1
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