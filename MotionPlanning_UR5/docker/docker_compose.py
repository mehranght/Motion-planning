#!/usr/bin/env python3

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Docker image repo")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Docker image name")
    parser.add_argument(
        "--tag", type=str, required=True,
        help="Docker image tag")
    return parser.parse_args()


def main(repo, image, tag):

    image_str = f"{image}:{tag}"

    if repo is not None:
        image_str = f"{repo}/{image_str}"

    print(f"""{{
    "version": "2.3",
    "services": {{
        "motion-planning-worker": {{
            "container_name": "motion-planning",
            "image": "{image_str}",
            "runtime": "nvidia",
            "network_mode": "host",
            "ipc": "host",
            "privileged": true,
            "logging": {{
                "driver": "journald"
            }},
            "security_opt": [
                "apparmor:unconfined"
            ],
            "environment": [
                "HOST_USER_ID",
                "HOST_GROUP_ID",
                "PYTHONPATH=./"
            ]
        }}
    }}
}}""")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
