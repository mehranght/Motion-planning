#!/usr/bin/env bash

# create non-root user, and run as that user
groupadd -r -g $HOST_GROUP_ID user
useradd -r -u $HOST_USER_ID -g $HOST_GROUP_ID -G sudo user
gpasswd -a user video
echo -e "pass\npass" | passwd user

source /opt/ros/noetic/setup.bash

exec /usr/local/bin/su-exec user "$@"
