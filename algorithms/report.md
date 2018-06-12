# Vocab
`-->` indicates the value/modification of hyper-parameters or design decisions

# Report

## 11/06/2018 - Commit 554ca445dbe046c8a91f754c60793f35ce628fc6

Using the `PointMass` environment
--> gamma=0.95, hidden_layers=[4], nb_episodes=500, nb_max_steps=400, nb_updates=10, batch_size=256
Jun11_18-34-29 : no idea if it is working


## 12/06/2018 - Commit ...

--> hidden_layers=[16]
--> randomize_goal=False
Jun12_11-32-51 : `PointMass` env with randomize_goal=False to make the task simpler
--> gamma=0.9

## 12/06/2018 - Commit ...

added plot for evaluating the value function, guess what, it works now, yay! :O


