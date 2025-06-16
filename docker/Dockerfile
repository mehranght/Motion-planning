FROM moveit/moveit:noetic-release

RUN apt-get update && apt-get install -y \
        python3-catkin-tools \
        python3-osrf-pycommon \
        ros-noetic-catkin \
        ros-noetic-trac-ik-kinematics-plugin \
        sudo \
        tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN bash -c "source /opt/ros/noetic/setup.bash && \
             rosdep update"
RUN apt-get update && \
    apt-get dist-upgrade

COPY docker/su-exec.amd64 /usr/local/bin/su-exec
RUN chmod +x /usr/local/bin/su-exec

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
