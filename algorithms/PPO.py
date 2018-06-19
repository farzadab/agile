import os
import torch as th
import numpy as np
import random
import tensorboardX

from algorithms.models import ActorNet
from nets import make_net
from algorithms.normalization import Stats


class PPO(object):
    def __init__(self, env, gamma, actor_layers, critic_layers, eps=0.2, writer=None, render=False, running_norm=False):
        '''
        @param env: the environment (Gym env)
        @param gamma: discount factor parameter (scalar)
        @param actor_layers: hidden layers to use for the actor network (list of integers)
        @param critic_layers: hidden layers to use for the critic network (list of integers)
        @param eps: the epsilon parameter for clipping the "surrogate" objective in PPO
        @param writer: writer for logging training curves and images(tensorboardX.SummaryWriter)
        @param render: whehter to call the `env.render(mode='human')` or not. If yes, done at every step (bool)
        @param running_norm: whether or not to use a running average for normalization
        '''
        self.eps = eps
        self.env = env
        self.gamma = gamma
        
        self.render = render
        self.writer = writer
        self.running_norm = running_norm

        self.save_path = None
        if self.writer is not None:
            self.save_path = os.path.join(self.writer.file_writer.get_logdir(), 'models')

        self.actor = ActorNet(env, actor_layers, log_std_noise=-1)
        self.critic = make_net([env.observation_space.shape[0]] + critic_layers + [1], [th.nn.ReLU() for _ in critic_layers])
        self.init_optims()

        self.norm_state = Stats(env.observation_space.shape[0])
        self.norm_rew = Stats(1)

    def init_optims(self):
        self.actor_optim = th.optim.SGD(self.actor.net.parameters(), lr=0.01, weight_decay=0.0003)
        self.critic_optim = th.optim.SGD(self.critic.parameters(), lr=0.001, weight_decay=0.0003)

    def train(self, nb_iters, nb_max_steps, nb_updates, batch_size):

        for i_iter in range(nb_iters):
            print('\rIteration: %d' % (i_iter+1), end='')

            mem = ReplayMemory(gamma=self.gamma)

            # TODO: for GAE and TD(lambda) may need to assume that mem stores a single episode
            # and requires all the observations to be added sequentially and in order
            self.sample_episode(nb_max_steps, mem, i_iter)

            old_actor = self.actor.make_copy()
            lc = 0
            la = 0
            sum_v = 0
            sum_vhat = 0
            sum_adv = 0
            for _ in range(nb_updates):
                batch = mem.sample(batch_size)

                # TODO: better logger
                # TODO: get rid of the TD error

                lcn, v, vhat = self.update_critic(batch)
                lc += lcn
                sum_v += v
                sum_vhat += vhat

                lan, adv = self.update_actor(batch, old_actor)
                la += lan
                sum_adv += adv

            if self.writer:
                self.writer.add_scalar('Train/LossCritic', lc / nb_updates, i_iter)
                self.writer.add_scalar('Train/LossActor', la / nb_updates, i_iter)
                self.writer.add_scalar('Extra/Value/Next', sum_v / nb_updates, i_iter)
                self.writer.add_scalar('Extra/Value/Current', sum_vhat / nb_updates, i_iter)
                self.writer.add_scalar('Extra/Action/Adv', sum_adv / nb_updates, i_iter)

            if callable(getattr(self.env, 'visualize_solution')):
                self.env.visualize_solution(
                    policy=lambda s: self.actor.forward(th.FloatTensor(s)).detach(),
                    value_func=lambda s: float(self.critic(th.FloatTensor(s))),
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
            

    def sample_episode(self, nb_max_steps, mem, i_episode):
        '''
        Samples a certain number of steps from the environment. Always resets at the start.
        TODO fix name: doesn't just sample a single episode, it sample `nb_max_steps` steps now!
        TODO fix outputs: I'm currently outputing multiple values for logging, but it's not the correct way to do it
        '''
        # maybe use an env wrapper?
        total_rew = 0
        acts = []

        done = True  # force reset
        # first = True

        for i_step in range(nb_max_steps):
            if done:
                mem.calc_episode_rewards()
                _state = self.env.reset()
                nstate = self.norm_state.normalize(_state)
                # if not first:
                #     print('')
                # else:
                #     first = False
                if self.running_norm:
                    self.norm_state.observe(_state)

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
        
        mem.calc_episode_rewards()
        # print('')
        
        if self.writer:
            self.writer.add_scalar('Train/AvgReward', float(total_rew) / (i_step+1), i_episode)
            self.writer.add_scalar('Extra/Action/Avg', float(np.array(acts).mean()), i_episode)
            self.writer.add_scalar('Extra/Action/Std', float(np.array(acts).std()), i_episode)
    

    def update_critic(self, batch):
        loss = 0
        self.critic.zero_grad()  # better way?
        sum_v = 0
        sum_vhat = 0
        for sample in batch:
            # mem.calculate_td(sample)
            # loss += sample.td()
            # TODO: use TD(lambda) later
            # FIXME: hmm, I'm using an updated version of `critic` every time. is that alright?
            # TODO: use MSEloss
            v = sample.cr #sample.r + self.gamma * self.critic(sample.ns)
            vhat = self.critic(sample.s)
            loss += (vhat - v).pow(2).sum()

            sum_v += v
            sum_vhat += vhat

        loss /= len(batch)
        loss.backward()
        self.critic_optim.step()
        return float(loss), sum_v / len(batch), sum_vhat / len(batch)
    
    def update_actor(self, batch, old_policy):
        loss = 0
        self.actor.zero_grad()
        sum_adv = 0
        for sample in batch:
            # TODO: use GAE
            # FIXME: hmm, I'm using an updated version of `critic` every time. is that alright?
            # # never mind for now, not using TD style update
            # adv = sample.r + self.gamma * self.critic(sample.ns) - self.critic(sample.s)
            adv = sample.cr - self.critic(sample.s)
            ratio = self.actor.get_nlog_prob(sample.a, sample.s) - old_policy.get_nlog_prob(sample.a, sample.s)
            ratio = ratio.exp()
            loss += th.min(
                ratio * adv,
                th.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
            )
            sum_adv += adv
        loss /= -1 * len(batch)  # we can only do gradient descent not **ascent**
        loss.backward()
        self.actor_optim.step()
        return float(loss), sum_adv / len(batch)

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

    def load_models(self, path):
        loaded_obj = th.load(path)
        
        self.actor.net.load_state_dict(loaded_obj['actor'])
        self.critic.load_state_dict(loaded_obj['critic'])
        self.norm_rew = loaded_obj['norm_rew']
        self.norm_state = loaded_obj['norm_state']
        # self.actor.load_model(os.path.join(path, '%s.actor' % str(index)))
        # self.critic.load_state_dict(th.load(os.path.join(path, '%s.critic' % str(index))))
        self.init_optims()


# TODO: merge with dynamics.Data
class ReplayMemory(object):
    def __init__(self, gamma):
        self.data = []
        self.gamma = gamma
        self.episode_start_index = 0

    def size(self):
        return len(self.data)

    def record(self, s, a, r, ns):
        self.data.append(ObservedTuple(s, a, r, ns))

    def calc_episode_rewards(self):
        rew = 0

        for t in reversed(self.data[self.episode_start_index:]):
            rew = t.r + self.gamma * rew
            t.set_cum_rew(rew)

        self.episode_start_index = len(self.data)
        # if len(self.data):
        #     print('\nminmax')
        #     print(np.min([t.cr for t in self.data]))
        #     print(np.max([t.cr for t in self.data]))
            # print(np.min([t.s.numpy() for t in self.data], 0))
            # print(np.max([t.s.numpy() for t in self.data], 0))

    # def extend(self, other):
    #     for k in self.key_names:
    #         self[k].extend(other[k])
    #     self._size += other.size()

    def sample(self, sample_size=None):
        if sample_size is None or sample_size > self.size():
            sample_size = self.size()
        return random.sample(self.data, sample_size)

    # def get_all(self):
    #     return self.data


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