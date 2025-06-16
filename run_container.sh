./docker2/gui-docker --rm -it motion-planning /bin/bash

#docker run \
#  -i \
#  --rm \
#  --privileged \
#  -v /home/aleksey/code/motion-planning/ws_moveit:/src/ws_moveit \
#  -v /home/aleksey/:/host_home/ \
#  -v /tmp/.X11-unix/:/tmp/.X11-unix/:rw \
#  -e DISPLAY \
#  -e XAUTHORITY=/host_home/.Xauthority \
#  -e HOST_USER_ID=`id -u` \
#  -e HOST_GROUP_ID=`id -g` \
#  --runtime=nvidia \
#  --net host \
#  -t motion-planning \
#  /bin/bash
