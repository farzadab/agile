# Vocab
  - `-->` indicates the value/modification of hyper-parameters or design decisions
  - `+++` indicates addition of new feature
  - `🔍🔍🔍` missing saved data
  - `✖` indicates bad result (or failure) in an experiment
  - `✔` indicates good result (or success) in an experiment
  - `⛳` indicates a really good result in an experiment (a point of comparison)
  - `???` question or wondering: can be the basis of the next experiments
  - `Res` indicates a response (or answer) to a question, most likely an answer found later down the line
  - `⚫⚫⚫` misc
  - `>` description or explanation about the run
  - `⇒⇒⇒` overal description

# Report

## 11/06/2018 - Commit 554ca445dbe046c8a91f754c60793f35ce628fc6
--> actor_optim: SGD(lr=0.001, weight_decay=0.0003)
--> value_optim: SGD(lr=0.0001, weight_decay=0.0003)

Using the `PointMass` environment
--> gamma=0.95, hidden_layers=[4], nb_episodes=500, nb_max_steps=400, nb_updates=10, batch_size=256
✖ Jun11_18-34-29 : no idea if it is working


## 12/06/2018 - Commit d2e5a88af8037aa219c961fe2b6dbb51ed332005

--> hidden_layers=[16]
--> randomize_goal=False
✖ Jun12_11-32-51 : `PointMass` env with randomize_goal=False to make the task simpler

+++ added plot for evaluating the value function

--> gamma=0.9
✔ 🔍🔍🔍 : worked really well, yay! 

✖ Jun12_15-59-53 : the same experiment as the last one (recreate the success), but doesn't seem to learn anything :(



+++ added plot for evaluating the policy

+++ found bugs in the environment: torque is not correctly clipped (fixed) + velocity cost is wrong at corners!
an experiment actually exploited the uncorrect torque clipping!
??? may need to change the reward. Res: distance is not the answer, but fixing the velocity at corners might be
✖ 🔍🔍🔍 : distance penalty doesn't work well!

??? is the sign of the loss used in updating actor correct? Res: yes
✔ 🔍🔍🔍 : flipped the sign of loss and it was obviously increasing the cost

??? the current sampling and update scheme is not so good, may need better memory structure or ...
??? need more exploration?

## 13/06/2018 - Commit 81985cb285a5c64b3063cba6661f92800a4cf4b5 (almost stable version of PPO - v0.5)

+++ refactored code + save plots in tensorboard

## 14/06/2018 - Commit 635fa869827f209612d0273074e3a944cd3c9f73

+++ added normalization
??? the learning is really slow! increasing lr (need annealing)
--> divided lr by 10: actor lr=0.01 value lr=0.001

Jun14_13-30-38 & Jun14_14-55-43 & Jun14_15-11-40 - running_average=False: 100 steps to avg reward of 2.5 (best is around 3), policy was almost perfect at 50 steps (noise maybe too high?)
Jun14_14-40-29 & Jun14_14-55-08 & Jun14_15-14-19 - running_average=True: seems to work slightly better (or just lucky init?)

??? why does the value function look so wrong??

Res: effect of running average: better learning at the start, but plateaus sooner (maybe nothing left to learn?). A lot better for the critic loss
??? maybe need to stop doing running average after a while?

??? possible bug: may need to apply the normalization after the episodes are done since the normalization changes within a single update
??? better visualization of policy: average over multiple velocity directions?

--> running_norm=True

## 14/06/2018 - Commit 11a95535f8c2aa8874fe25175f2dec5801c8e433

+++ added separate architecture configs for actor and critic networks

--> running_norm=True, actor_layers=[], critic_layers=[4] (same as before)
Jun14_16-08-58: bad init, but still (almost) perfect policy after 50
--> critic_layers=[8]
Jun14_15-50-36: learned good enough policy after 35 iters, (almost) perfect after 60
⚫⚫⚫ took a picture of this, but the other one (Jun14_16-08-58) was actually better! :)

--> distance^2 cost instead of velocity reward
Jun14_16-46-38: 
Jun14_17-10-41: same as above but scaled reward (mostly for visualzation). it had a really lucky initialization :)
Jun14_17-12-09: struggling to learn :(
??? TD might help a lot here


--> randomize_goal = True
Jun15_15-27-31: randomize_goal and velocity reward work perfectly! it took around 150 to get max reward

+++ added save/load functionality

Jun15_16-45-08: randomize_goal and distance squared works too. Not sure how well, need a validation criteria


## 18/06/2018 - Commit 62ecde1ebda63e652fe86cad27a9bb814d244391

Jun18_18-02-41 python3 -m algorithms.main --env_reward_style='velocity' --desc 're-running Jun15_15-27-31 since it was removed'
> almost perfect!
> Policy is just a simple PD-controller:
[[ -9.6279,  -0.5611],
 [ -0.7100,  -8.4666],
 [ 11.5908,   1.1154],
 [  0.5532,  11.2685],
 [ -0.1617,  -0.0322],
 [ -0.0042,  -0.1058],
 [  0.0314,  -0.0777]]


Jun18_18-02-16: python3 -m algorithms.main --env_reward_style='distsq' --desc 're-running Jun15_16-45-08 since it was removed'
> makes a weird circular motion when it gets close to the target, needs investagation (TODO)

Jun19_11-30-53: python3 -m algorithms.main --env_randomize_goal false --env_reward_style='distsq' --desc 'distsq works poorly, why? making the task simpler'
> still not so good
Jun19_12-25-34: python3 -m algorithms.main --env CircularPointMass --env_reward_style='distsq' --desc 'see if distsq works well with follow circle task'
> it's actually better, it was still getting better, need to investigate how farther it can go
Jun19_14-05-27: python3 -m algorithms.main --env CircularPointMass --env_reward_style='distsq' --load_path runs/Jun19_12-25-34_farzad-desktop/models/399-ppo.pt --desc 'starting from already learned policy, to see how much better it can get'
> was getting better, but not much
+++ hmm, maybe starting point at random is not such a good idea, added a "start-at-goal" feature: CircularPointSAG
Jun19_14-38-00: python3 -m algorithms.main --env CircularPointSAG --env_reward_style='distsq' --desc '1st try after adding start-at-goal feature'
Jun19_14-50-05: python3 -m algorithms.main --env CircularPointSAG --env_reward_style='velocity' --desc 'see how much faster it learns with velocity reward'

### CircularPointMass

Jun18_18-27-02: python3 -m algorithms.main --env CircularPointMass --env_randomize_goal false --env_reward_style velocity --env_max_steps 200  --desc 'circular point-mass 1st trial'

+++ found bug in the CircularPointMass environment

Jun18_18-42-16: python3 -m algorithms.main --env CircularPointMass --env_randomize_goal false --env_reward_style velocity --env_max_steps 200  --desc 'circular after goal-bugfix'
> almost perfect!
> Policy is just a simple PD-controller:
[[-10.5300,   1.3206],
 [ -1.1382,  -9.4609],
 [  8.8153,   0.5839],
 [ -1.3964,   8.1384],
 [  0.6020,   0.0905],
 [ -0.0276,   0.7987],
 [  0.0507,   0.0119]]


## 19/06/2018 - Commit 02ab398636558af19e3521a156666e4ee893a894

??? Michiel suggests fixes for the suboptimal performance of "distsq": better integration, running_norm=False, increasing critic size and maybe actor?
+++ better integration: V2
Jun19_18-42-03: python3 -m algorithms.main --env PM2 --env_reward_style='distsq' --desc 'PointMassV2 env: see if the integration matters'
> awful ... and the results are comparable/worse than Jun18_18-02-16

+++ increasing critic size from [8] to [16,16]

Jun19_17-16-33: python3 -m algorithms.main --env PM2 --env_reward_style='distsq' --desc 'PointMassV2 env: see if the integration matters'
> a little better, but still awful :(
Jun20_10-46-04: python3 -m algorithms.main --env PM2 --running_norm false --env_reward_style='distsq' --desc 'distsq with PM2 and no running_norm: see if disabling running_norm works'
> not good
Jun20_10-49-19: python3 -m algorithms.main --env PM2 --running_norm false --env_reward_style='distsq' --desc 'distsq with a non-linear critic: [8] instead of []'
> a little better but still not good (may need more training)Res: seems like the critic loss is better with a larger critic network, but the performance hasn't changed much

Jun20_10-54-51: python3 -m algorithms.main --env PM2 --running_norm false --env_reward_style='distsq' --env_max_steps 300 --nb_max_steps 1200 --desc 'distsq with PM2: increasing max_steps'
> (this is probably unfair, but let's see if it matters)
> but it was awful
Jun20_13-37-15: python3 -m algorithms.main --env PM2 --running_norm false --env_reward_style='distsq' --env_max_steps 300 --nb_max_steps 3000 --desc 'distsq with PM2: increasing max_steps'

Jun20_14-37-43: python3 -m algorithms.main --env PM2 --running_norm false --env_reward_style='distsq+g' --desc 'see if the goal reward matters'

Jun20_14-55-08: python3 -m algorithms.main --env NSPM --env_reward_style='distsq' --desc 'seeing if increasing control-step (4x) matters too much'

+++ bug (fixed): maybe we terminate the last episode too soon so the cumulative reward is wrong ...

+++ bug (monkey-patched): shifting reward for normalization is wrong :(

+++ bug (fixed) in ReplayMemory.calc_episode_targets wrong index: start episode was not updated

+++ bug (fixed, visualization): did not use to normalize

--> the holy fix: using Adam instead of SGD :'(

--> gae_lambda=0.8
Jun25_19-27-38: python3 -m algorithms.main --env_reward_style distsq  --nb_max_steps 10000 --running_norm false --desc 'Adam, no normalization and gae_lambda=0.8'
> so larger λ values don't hurt now
Jun25_19-38-55: python3 -m algorithms.main --env_reward_style 'distsq'  --running_norm true --nb_max_steps 10000 --desc 'test if normalization is bad or not'
> not it's good to have
Jun25_19-46-49: python3 -m algorithms.main --env_reward_style 'distsq' --nb_max_steps 10000 --desc 'see if normalization of advantage helps or not'
> not sure, but doesn't seem to hurt either (note: both λ=0.2 and λ=0.8 seem to have worked on this problem)


## 25/05/2018 - Commit 10bb09cd472da8f94c4f8b074db5de9e39171f04

--> turning off running norm halfway through the training, since it can be detrimental after a while

Jun26_12-52-28: python3 -m algorithms.main --env CPSAG --env_reward_style 'distsq' --nb_max_steps 5000 --desc 'test fixed code with moving target'
> great! almost perfect after 40 iterations
Jun26_12-57-49: python3 -m algorithms.main --env CPSAG --env_reward_style 'distsq' --nb_max_steps 1000 --desc 'how much does the # steps matter?'
> still works well enough after 40 iterations. maybe this env is too easy?


## 26/06/2018 - Commit 6824bd5fb9f8e75ec8db84eb210ade43d1d5b53f
> try stuff on Gym envs
Jun26_13-32-50: python3 -m algorithms.main --env 'Pendulum-v0' --net_nb_layers 1 --nb_max_steps 5000
> Pendulum works great after 120 * 5000 episodes
Jun26_13-25-43: python3 -m algorithms.main --env 'MountainCarContinuous-v0' --net_nb_layers 1 --nb_max_steps 5000
> doesn't work ... maybe increasing noise will help ...
Jun26_14-43-12: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 1 --net_nb_critic_layers 2 --nb_max_steps 20000 --desc 'Hopper\!'
> doesn't work: doesn't go beyond the first hop

## 26/06/2018 - Commit b1d4040560dcc7ae9bc2f027ac1918fd6418030a
+++ better logging + logs return

--> vpred = vfunc(self['nstate'][-1])
Jun26_18-36-26: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 1 --net_nb_critic_layers 2 --nb_max_steps 20000 --desc "see if vpred = vfunc(self['nstate'][-1]) works"
> this should not really work anyway, only apply it when the agent is still alive
Jun26_18-52-22: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 2 --net_nb_critic_layers 3 --nb_max_steps 50000 --nb_iters 500
> γ is probably too low: seems to work better, but still doesn't go beyond the first hop

## 27/06/2018 - Commit 48e5b3b70259b916a083054847b21230b60a8f69

--> γ: 0.9 -> 0.99 | λ: 0.8 -> 0.95 | lr: 1e-3 -> 3e-4 | critic_lr: lr * 5 | batch_size: 100 -> 512*8 | nb_epochs: 20 -> 10

Jun27_11-48-04: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 2 --net_nb_critic_layers 3
> learns to stand still! maybe more exploration will help

Jun27_12-22-35: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 2 --net_nb_critic_layers 3 --noise 0.5
> too much noise, not much learning
Jun27_12-32-26: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 2 --net_nb_critic_layers 3 --noise 0
> better but still has too much noise

--> manually added exploration noise annealing: LinearAnneal(-1, -1, 1, 1, -1, -1, 0, 0, -2)
Jun27_13-12-50: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 2 --net_nb_critic_layers 3
> noise of more than 0 is just too much, the reward plumets down to less than zero!
--> manually added exploration noise annealing: LinearAnneal(-1, -1, 0, 0, -1, -1, 0, 0, -2)
Jun27_13-26-56: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 32 --net_nb_layers 2 --net_nb_critic_layers 3
--> manually added exploration noise annealing: LinearAnneal(-0.7, -1.6)
--> increased forward progress reward by 10x
Jun28_16-25-15: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 64 --net_nb_layers 2
> doesn't seem to learn to hop, just dives forward
(⛳⛳) Jun28_16-34-49: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 64 --net_nb_layers 2 --gamma 0.995 --gae_lambda 0.97
> ✔✔✔ It hops, yay! gamma 0.995 gae_lambda 0.97 are great!
--> γ: 0.99 -> 0.995 | λ: 0.95 -> 0.97 (✔✔✔)

> Let's see if the 10x progress reward is necessary or not. Res: yes, it is!
Jun28_17-34-45: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 64 --net_nb_layers 2 --gamma 0.995 --gae_lambda 0.97
> no 10x reward and it doesn't want to move at all
Jun28_18-08-37: python3 -m algorithms.main --env 'HopperBulletEnv-v0' --net_layer_size 64 --net_nb_layers 2 --gamma 0.995 --gae_lambda 0.97
> with 10x reward it soon learn to hop after ~120 iterations and gets really good after ~160 iterations (same as Jun28_16-34-49)


## Circular PointMass with Phase
Jun29_cphase_large_net: default hyper-params
> works well but if run for longer horizons, can get behind
--> using value function as end-state as prediction
Jun29_cphase_vpred_end: using value function as end-state as prediction
> works better!
Jun29_cphase_vpred+40xcritic_lr: increasing critic lr from 5x to 40x
> works even better!

## Walker2D
--> γ=0.995
(1 ) Jun28_lambda_0.97  : λ=0.97 γ=0.995
--> 10x progress reward
(4 ) Jun29_lambda_0.99  : λ=0.99 γ=0.995
(6 ) Jun29_lambda_0.95  : λ=0.95 γ=0.995
(3 ) Jun29_lambda_0.92  : λ=0.92 γ=0.995
(3 ) Jun29_lambda_0.92_2: λ=0.92 γ=0.995
(7 ) Jun29_lambda_0.90  : λ=0.90 γ=0.995
(8 ) Jun29_lambda_0.70  : λ=0.70 γ=0.995
(5 ) Jun29_gamma_0.999  : λ=0.97 γ=0.999
(9 ) Jun29_lambda_0.97  : λ=0.97 γ=0.995  > it was a re-run, but got really different result (did I do something different?)
(10) Jun29_gamma_0.99   : λ=0.97 γ=0.99
--> 1x progress reward
(-1) Jun29_16-37-59: learns to stand still, so lots of reward ...
--> 3x progress reward
(1.5) Jun29_17-31-35
(0 ) Jun29_17-33-58
> just two steps
--> 7x progress reward
(3.5) Jun29_17-38-02
> one step jump
--> 15x progress reward
(8.5) Jun29_17-46-06
> idiot
--> 4x progress reward
(  ) Jun29_19-07-37
> tries to take a step + sometimes just stays still
--> 2x progress reward
(  ) Jun29_19-08-13
> mostly just stays still
--> 5x progress reward + better logging + removed LinearAnneal (fixed exploration)
(  ) Jul03_15-13-14
> learns an interesting heel walk but can't use it for long + stays still or drops
--> relaxed early termination: z: 0.8 -> 0.68 | pitch: 1.0 -> 5.0
(  ) Jul03_16-22-00
--> early termination: pitch: 5.0 -> 1.0
(  ) Jul03_16-33-14
--> stale cost: if sum((s - s') ** 2) < 1e-1 then penalize by -0.5
(  ) Jul03_16-45-48
--> early termination: z: 0.68 -> 0.70
(  ) Jul03_16-53-24
> takes two steps and then stays still
> the stale cost seemed to work: at first it was just staying still at start but was driven away from that solution
--> early termination: z: 0.70 -> 0.80 + increased stale cost to -1 instead of -0.5
(  ) Jul03_17-38-24
--> added multi_step option
(  ) Jul03_18-16-02: multi_step of 5
> just stays still
--> try to use the same reward as OpenAI Gym MuJoCo envs
(  ) Jul04_10-51-26
(  ) Jul04_10-52-03: λ=0.9
--> decreased critic lr from 40x to 1x + added LinearAnneal back
(  ) Jul04_11-04-08: same hyper-params as paper λ=0.95 | γ=0.99 | batch_size=2048 | mini_batch_size=64
--> just the progress reward ...
(  ) Jul04_11-33-18
(  ) Jul04_11-47-39
+++ found (and fixed) a bug I guess: scaling the reward with a running norm is not a good idea after all :(
--> 4x progress reward
(  ) Jul04_14-06-59
--> decreased max episode length for Walker2D (200)
--> do 5 times more critic updates
(  ) Jul04_14-45-14: batch_size=8000
> ✔ not bad! but maybe the episodes end too fast?
> None of them can take more than two steps

## 04/07/2018 - Commit c28cf8bfe84467b360cd0a834785932b58097c5f

--> increased max episode length for Walker2D to the original (1000)
(✔ ) Jul04_15-36-19: batch_size=8000  | γ=0.99 | λ=0.95
> It's walking, yay!
Zhaoming ----- increase network size!
(✔ ) Jul04_16-00-57: batch_size=8000  | γ=0.99 | λ=0.95 | nb_layers=3 | net_layer_size=80
> Even faster than Jul04_15-36-19!
(✖ ) Jul04_16-03-14: batch_size=10000 | γ=0.99 | λ=0.95 | nb_layers=3 | net_layer_size=80
> somehow this didn't turn out too well, maybe just random noise?
(⛳⛳) Jul05_10-59-53: no advantage normalization (o.w. same as Jul04_16-00-57)
> worked better than Jul04_16-00-57!
(  ) Jul05_11-01-46: higher gamma  γ=0.995      (o.w. same as Jul04_16-00-57)
> doesn't work as well, but may be due to higher variance of total return (larger γ means larger discounted return)
(  ) Jul05_11-02-38: higher lambda λ=0.97       (o.w. same as Jul04_16-00-57)
> seems to work worse, but may be due to noise only

Glen    ----- fixed reward normalization: 1 / (1-γ)


## 05/07/2018 - Commit 3981724c80923960dfec50305d072a8acbe7b1f9 (Serialization V2)
⇒⇒⇒ This is a hard fork that can't open the previously saved files. In order to open the previously saved files do something like:
```bash
git checkout serialization_v1  # v1 compatible branch
python3 -m algorithms.main --env 'Walker2DBulletEnv-v0' --net_layer_size 80 --net_nb_layers 3 --replay_path runs/Jul05_11-02-38_farzad-desktop/models/last-ppo.pt
git checkout master   # go back to the master branch
```
(  ) Jul05_15-57-48: batch_size=4000 (test new code, but went wrong)
(  ) Jul05_16-08-31: same as Jul04_16-00-57 (test new code)
(  ) Jul05_16-09-00: test new code with Hopper
> learns a brittle hop (toe hop) and falls after 3-4 cycles
--> 1x progress reward
(✖ ) Jul05_16-14-31: see if 1x progress reward is enough
> pretty much confirms that the 1x progress reward is NOT enough
--> 4x progress reward
Zhaoming ----- change the order of critic and actor updates
(  ) Jul06_12-10-44: same as Jul05_16-08-31 but with the order of updates changed (bad experiment: no advantage normalization)
(  ) Jul06_13-26-40: same as above (bad experiment: no advantage normalization)
> both worked really bad
(  ) Jul06_15-12-32: same as above but with advantage normalization
> something weird happened: accidentally I killed this experiment and had to re-run it from the start (Jul06_17-03-39)
> this new version seems to work a lot better!
(  ) Jul06_15-13-11: just re-running Jul05_16-08-31 to see if it still works
> well, it does, maybe the advantage normalization is crucial!

## Phase-based kinematic motion

## 09/07/2018 - Commit 789d92c28515e09f8d3aed5bf4eb699d506a3cda
## 09/07/2018 - Commit 93eb868232c10e6d8f9af1c88dc1534fe21d1b91

⇒⇒⇒ Phase-based motion for PointMass with different envs (linear actor)
⚫⚫⚫ python3 -m algorithms.main --env <env> --env_reward_style distexp --env_max_steps 500 --batch_size 8000
Jul06_16-54-05: CPhase2
> not bad, but maybe a linear actor is not enough
Jul06_17-14-15: LPhase1
> almost perfect tracking, but sacrifices left side a little bit
Jul09_17-14-14: LPhase2
> not bad, but could be better, doesn't track perfectly and overshoots
Jul09_17-14-46: LPhase3
> really bad, just goes right (doesn't understand what to do at discontinuity)
Jul09_17-15-16: SqPhase1
> not bad, but follows a circle like path
Jul09_17-15-28: SqPhase2
> not bad! follows a triangle path at the discontinuity

⇒⇒⇒ the same but with more powerful actors (`net_nb_layers=2`):
⚫⚫⚫ python3 -m algorithms.main --env <env> --env_reward_style distexp --net_nb_layers 2 --env_max_steps 500 --batch_size 8000
(⛳) Jul06_17-29-33: CPhase2
(⛳) Jul09_19-21-44: LPhase2
(⛳) Jul09_19-22-10: LPhase3
(⛳) Jul09_19-22-24: SqPhase1
(⛳) Jul09_19-22-35: SqPhase2
(⛳) Jul10_10-40-14: StepsPhase1
(⛳) Jul10_10-41-55: StepsPhase2
> non-linear actors are all better (using these as reference and making a video)
⇒⇒⇒ trying harder environments
(  ) Jul10_12-40-43: PhaseRN1
> not good, just barely follows the bottom part of the path
(  ) Jul10_12-41-04: PhaseRN1_NC
> not good but interesting: just tries to get the stationary point right and gets decent reward! (+ maybe the reward scheme is not good)

(  ) Jul10_14-15-14: PhaseRN2
(  ) Jul10_14-15-34: PhaseRN2_NC

⇒⇒⇒ other env reward styles
(  ) Jul10_14-42-12: distsq
(  ) Jul10_14-57-19: velocity
(⛳) Jul10_15-03-26: distsq+e


## Reference based motion with Walker2D

⇒⇒⇒ tried using Ben's walker environments, but it seems like the power is not tuned correctly (jump too high)
Jul11_15-42-25: power=0.40
Jul11_16-22-36: power=0.01
Jul11_16-30-43: power=0.02
Jul11_16-31-10: power=0.04
Jul11_17-16-04: power=0.08
⇒⇒⇒ giving up, maybe it's better to use the PyBullet original envs
⚫⚫⚫ python3 -m algorithms.main --env 'Walker2DEnv-v0' --net_layer_size 80 --net_nb_layers 3  --batch_size 8000
Jul12_11-35-39: just a re-run of Jul05_16-08-31 and Jul05_10-59-53 (used 4x progress reward + power=0.40)