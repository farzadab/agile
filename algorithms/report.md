# Vocab
  - `-->` indicates the value/modification of hyper-parameters or design decisions
  - `+++` indicates addition of new feature
  - `🔍🔍🔍` missing saved data
  - `✖` indicates bad result (or failure) in an experiment
  - `✔` indicates good result (or success) in an experiment
  - `???` question or wondering: can be the basis of the next experiments
  - `Res` indicates a response (or answer) to a question, most likely an answer found later down the line
  - `⚫⚫⚫` misc
  - `>` description or explanation about the run

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