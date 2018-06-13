import torch as th
import numpy as np
import random
import tensorboardX

from algorithms.models import ActorNet
from algorithms.plot import ScatterPlot, QuiverPlot, Plot
from nets import make_net


class PPO(object):
    def __init__(self, env, gamma, hidden_layers, eps=0.2, writer=None):
        self.xlim = [-20,20]
        self.ylim = [-20,20]
        plot = Plot(1,2)
        self.splot = ScatterPlot(parent=plot, xlim=[-20,20], ylim=[-20,20], value_range=[-10,10])
        self.qplot = QuiverPlot(parent=plot, xlim=[-20,20], ylim=[-20,20])
        self.eps = eps
        self.env = env
        self.gamma = gamma
        self.writer = writer
        self.actor = ActorNet(env, hidden_layers)
        self.critic = make_net([env.observation_space.shape[0]] + hidden_layers + [1], [th.nn.ReLU() for _ in hidden_layers])
        self.actor_optim = th.optim.SGD(self.actor.net.parameters(), lr=0.001, weight_decay=0.0003)
        self.critic_optim = th.optim.SGD(self.critic.parameters(), lr=0.0001, weight_decay=0.0003)
        # pass
    def train(self, nb_episodes, nb_max_steps, nb_updates, batch_size):
        for i_episode in range(nb_episodes):
            print('Iteration %d' % (i_episode+1))
            mem = ReplayMemory(gamma=self.gamma)
            # TODO: for GAE and TD(lambda) may need to assume that mem stores a single episode
            # and requires all the observations to be added sequentially and in order

            obs = self.env.reset()
            # maybe use an env wrapper?
            total_rew = 0
            acts = []
            for i_step in range(nb_max_steps):
                env.render(mode='human')
                act = self.actor.get_action(th.FloatTensor(obs), explore=True).detach().numpy()
                acts.append(act.numpy())
                obs_p, rew, done, _ = self.env.step(act)
                print('\r%5.2f' % rew, end='')
                mem.record(obs, act, rew, obs_p)
                total_rew += rew
                obs = obs_p
                if done:
                    break
                # FIXME: maybe run more episodes if they end too soon?
            print('')

            mem.calc_cum_rews()

            if self.writer:
                self.writer.add_scalar('Train/AvgReward', float(total_rew) / (i_step+1), i_episode)
                self.writer.add_scalar('Extra/Action/Avg', float(np.array(acts).mean()), i_episode)
                self.writer.add_scalar('Extra/Action/Std', float(np.array(acts).std()), i_episode)

            
            old_actor = self.actor.make_copy()
            lc = 0
            la = 0
            sum_v = 0
            sum_vhat = 0
            sum_adv = 0
            for i_update in range(nb_updates):
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
                self.writer.add_scalar('Train/LossCritic', lc / nb_updates, i_episode)
                self.writer.add_scalar('Train/LossActor', la / nb_updates, i_episode)
                self.writer.add_scalar('Extra/Value/Next', sum_v / nb_updates, i_episode)
                self.writer.add_scalar('Extra/Value/Current', sum_vhat / nb_updates, i_episode)
                self.writer.add_scalar('Extra/Action/Adv', sum_adv / nb_updates, i_episode)
            
            if self.splot and self.qplot:
                x = np.linspace(self.xlim[0], self.xlim[1], 20)
                y = np.linspace(self.ylim[0], self.ylim[1], 20)
                points = np.array(np.meshgrid(x,y)).transpose().reshape((-1,2))
                v = np.ones(points.shape[0])
                d = np.ones((points.shape[0], 2))
                for i, p in enumerate(points):
                    v[i] = float(self.critic(th.FloatTensor(np.concatenate([p, [0,0,0,0]]))))
                    d[i] = self.actor.forward(th.FloatTensor(np.concatenate([p, [0,0,0,0]]))).detach()
                self.splot.update(points, v)
                self.qplot.update(points, d)

    
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

        
# TODO: merge with dynamics.Data
class ReplayMemory(object):
    def __init__(self, gamma):
        self.data = []
        self.gamma = gamma

    def size(self):
        return len(self.data)

    def record(self, s, a, r, ns):
        self.data.append(ObservedTuple(s, a, r, ns))
    
    def calc_cum_rews(self):
        rew = 0
        for t in reversed(self.data):
            rew = t.r + self.gamma * rew
            t.set_cum_rew(rew)

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

if __name__ == '__main__':
    # test PPO with the Pendulum
    import gym
    from algorithms.senv import PointMass
    
    # env = gym.make('Pendulum-v0')
    env = PointMass(randomize_goal=False)

    # TODO: normalization
    writer = None#tensorboardX.SummaryWriter()
    ppo = PPO(env, gamma=0.9, hidden_layers=[4], writer=writer)
    print(ppo.actor.net)
    print(ppo.critic)
    try:
        ppo.train(nb_episodes=500, nb_max_steps=400, nb_updates=10, batch_size=256)
    finally:
        env.close()