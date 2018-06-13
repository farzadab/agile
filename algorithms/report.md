# Vocab
`-->` indicates the value/modification of hyper-parameters or design decisions
`+++` indicates addition of new feature
`🔍🔍🔍` missing saved data
`✖` indicates bad result (or failure) in an experiment
`✔` indicates good result (or success) in an experiment
`???` question or wondering: can be the basis of the next experiments
`Res` indicates a response (or answer) to a question

# Report

## 11/06/2018 - Commit 554ca445dbe046c8a91f754c60793f35ce628fc6

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

