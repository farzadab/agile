# Vocab
`-->` indicates the value/modification of hyper-parameters or design decisions
`+++` indicates addition of new feature
`🔍🔍🔍` missing saved data
`✖` indicates bad result (or failure) in an experiment
`✔` indicates good result (or success) in an experiment
`???` question or wondering: can be the basis of the next experiments
`Res` indicates a response (or answer) to a question, most likely an answer found later down the line

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

## 14/06/2018 - Commit ...

+++ added normalization
??? the learning is really slow! increasing lr (need annealing)
--> divided lr by 10: actor lr=0.01 value lr=0.001

Jun14_13-30-38 & Jun14_14-55-43 & Jun14_15-11-40 - running_average=False: 100 steps to avg reward of 2.5 (best is around 3), policy was almost perfect at 50 steps (noise maybe too high?)
Jun14_14-40-29 & Jun14_14-55-08 & Jun14_15-14-19 - running_average=True: seems to work slightly better (or just lucky init?)

Res: effect of running average: better learning at the start, but plateaus sooner. A lot better for the critic loss