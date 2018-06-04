#!/bin/bash
sudo pip3 install -r requirements.txt

if [ ! -e ben ]; then
    git clone https://github.com/belinghy/pybullet-custom-envs .bens_envs
    # TODO: don't do this
    ln -s .bens_envs/cust_envs .
fi