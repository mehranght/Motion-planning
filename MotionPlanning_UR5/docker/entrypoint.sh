#!/usr/bin/env bash

set -ex

mount -a # mount fstab entries

groupadd -r -g $HOST_GROUP_ID user
useradd -r -u $HOST_USER_ID -g $HOST_GROUP_ID -G sudo -m user
gpasswd -a user video
echo "user:pass" | chpasswd

source /etc/profile.d/conda.sh
conda activate mp

exec gosu user "$@"
