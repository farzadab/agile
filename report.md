
## Commit ad60aed6195493b8db4ecfa57aae27d73f3d15b9

Reason: awful learning curves, need hyper-parameter tuning (choose between Adam and SGD, lr, batch_size, ...)

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

## Commit 02f46fd4d7ca786cbeac29a4707e5f7c83e91237

Reason: SGD was not using L2 weight decay penalty, so maybe that's the problem?

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


## Commit eacc95bf048360ca89b27a0435b0e336e3872390

Reason: network size and layers

starting from "Adam 1024 lr=0.0003 weight_decay=0.0003 not just mpc" and hidden_layers = [256]

(2) Jun04_12-27-48 - hidden_layers = 1 * [512] :  (parent: Jun04_11-27-24)
(1) Jun04_12-27-26 - hidden_layers = 2 * [256] :  (parent: Jun04_11-27-24) **best**
(2) Jun04_12-34-08 - hidden_layers = 3 * [256] :  (parent: Jun04_11-27-24)


## Commit 2ab9b1522afb5f0d0b2d551831be0465e083b293

env: HalfCheetahNew-v0

Reason: maybe HalfCheetah is easier?

Jun04_14-00-01
Jun04_15-00-20
Jun04_16-12-35

Learning is not bad, but finishes higher than the Crab environment, but still the resulting MPC does not produce good motions --- going back to the basics as Michiel suggested.

## 06/06/2018 - Commit c5b4698ed7e1a0d1f294024bf5e73ad575d78920

Added evaluations for longer time horizons
Jun06_12-49-35 - NN [256, 256]: evaluations for MPC and horizon 1-16 seem to work well, but for 32 the result is garbage. Also for RandCtrl and 1-16, we still get resonable decrease (high variance), but still for 32 it's awful
Jun06_16-42-40 - MOE [64] [64] with nb_total_steps=100: starts really high (not sure what the total_steps for the past expertiment was) goes down a lot (but still high) and goes up again! :((
Jun06_16-53-45 - same as above with nb_total_steps=1000: still not too good :(


## 06/06/2018 - Commit e84d6b659e7f047e7414a11f8317cec5bd17b479

Experimenting with manual control of NABi robot (PD without feet). Can't really move it too much without falling. Mostly just leaning to left and right or going from a tall stance to a wide stance (can almost jump).
Also changed the robot a little bit.