
## Commit ad60aed6195493b8db4ecfa57aae27d73f3d15b9

** note: there is no test set, so the comparisons might not be just, but since we collect new random data, it shouldn't be too bad
** but mpc and non-mpc comparison is a lot different

env: Crab2DCustomEnv
fits new data, then extends the old and fits sample of 4 times the new data
(doesn't go up, best is good, end is good)

inits:
optim1: optim.Adam(net.parameters(), lr=0.001, weight_decay=0.1)
optim2: optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
hidden_layers = [256]

(-xx) Jun01_16-32-19 - Adam 128: goes down at first, but then goes up and plateaus
(+++) Jun01_16-45-43 - SGD  512: goes down slower but seems to be stable (always going down), also it maybe that it has plateaued at the end
(---) Jun01_17-00-31 - Adam 512: goes down really fast at first, but then goes up agan and plateaus (almost the same as Adam 128: two rows above)
(x-x) Jun01_17-14-55 - Adam 512 but fits whole data every single time (almost the same as Adam 512 row above, but worse)
(x+-) Jun01_17-23-49 - Adam 512 but only mpc: goes lower than all the rest and faster, but goes back up
(-++) Jun01_17-31-30 - Adam 1024: almost the same as above, but a less variation: doesn't go up as high in the end
(xxx) Jun01_17-35-58 - Adam 512  lr=0.003 (original lr=0.001): goes down really fast at first but quickly (faster than others) goes up high
(+-+) Jun01_17-46-04 - Adam 1024 lr=0.0003: goes down constant
(+--) Jun01_18-17-04 - Adam 512  lr=0.0003: doesn't go up, but ends just short of Adam 512
(+--) Jun01_17-52-03 - SGD  1024: starts high goes down constantly but it still ends high
(+++) Jun01_17-55-53 - SGD  1024 lr=0.003 (original lr=0.001): starts high but goes down steadily (finishes almost the same as Adam 1024 lr=0.0003)
(+xx) Jun01_18-00-39 - SGD  1024 lr=0.0003: starts high
(+++) Jun01_18-04-12 - SGD 512 but only mpc: constant decrease and the best result (almost the same as SGD 512 but lower)!

high-level results:
  - Adam tends to go up unless conservative lr and batch_size are chosen (Adam 1024 lr=0.0003)
    - only 1024 almost works. haven't tried only lr=0.0003 
  - "only MPC" is better than randomizing on both Adam and SGD
    - not sure if it's better overall or just in on-policy trajs (but our MPC is not too on-policy anyway)
  - "fits whole data every time" doesn't seem to work, but maybe more conservative lr might help: doesn't need as much data


==================================================================================================================================

## Commit ....

Idea: SGD was not using L2 weight decay penalty, so maybe that's the problem?

(+xx) Jun04_10-52-02 - SGD 512 mpc: awful performance! (parent: Jun01_18-04-12)
(+++) Jun04_11-05-12 - Adam 1024 lr=0.0003 weight_decay=0.03: fantastic! straight down and far better than others (parent: Jun01_17-46-04)
(+++) Jun04_11-18-52 - Adam 1024 lr=0.0003 weight_decay=0.001: even better, had a sudden jump, but I think  it was just a random thing (parent: Jun04_11-05-12)
(+++) Jun04_11-27-24 - Adam 1024 lr=0.0003 weight_decay=0.0003: the **best**! (parent: Jun04_11-05-12)
(+++) Jun04_11-20-33 - Adam 1024 lr=0.0003 no weight_decay: great, almost the same as weight_decay=0.001 (parent: Jun04_11-05-12)
(+++) Jun04_11-25-05 - Adam 1024 lr=0.0003 weight_decay=0.001 only mpc: better than non-mpc **3rd best** (parent: Jun04_11-18-52)
(+++) Jun04_11-45-40 - Adam 1024 lr=0.0003 weight_decay=0.0003 only mpc: not as good as non-mpc **2nd best** (parent: Jun04_11-27-24 and Jun04_11-25-05)

high-level results:
  - Adam works great with lower weight_decay!
    - best was weight_decay=0.0003 and not just mpc

* thoughts: maybe we can play with lr and batch_size later on, but not now
