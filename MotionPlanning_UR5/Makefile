SHELL := /bin/bash
.ONESHELL:

REPO ?= alexnet.lan:5000

CURR_REV=$(shell git rev-parse  HEAD)

IMAGE=motion-planning
TAG:=$(CURR_REV)

ALEXNET_LIB_DIR=/alexnet/libs

# Tar command constructed to ensure we get reproducible tarballs. This ensures
# that if we have multiple versions of the repo checked out our docker builds
# will re-use the same cached images
#
# Somewhat stolen from http://h2.jaguarpaw.co.uk/posts/reproducible-tar/
#
# 2021-01-01 chosen for date since it's a semi-recent fixed date
TAR_CMD = tar \
	--sort=name \
	--mtime="2021-01-01" \
	--owner=0 \
	--group=0 \
	--numeric-owner \
	--mode="go-rwx,u-rw" \
	--no-recursion

.PHONY: build-docker-compose
build-docker-compose:
	@if [ -z "${REPO}" ]; then \
		python3 docker/docker_compose.py \
			--image ${IMAGE} \
			--tag ${TAG} > docker/docker-compose.json;
	else \
		python3 docker/docker_compose.py \
			--repo ${REPO} \
			--image ${IMAGE} \
			--tag ${TAG} > docker/docker-compose.json;
	fi

.PHONY: build-docker-image
build-docker-image:
	@mkdir -p docker/libs
# Package requirements separately from the rest of the repo to avoid re-installing
# deps on minor script changes
	@git ls-files --recurse-submodules libs | \
		$(TAR_CMD) -cf docker/libs/requirements.tar -T - requirements.txt scripts/setup.sh
# Package repo files into a tarball for consistency between builds. Tarball
# should be binary identical for the same file contents
	@git ls-files --recurse-submodules | \
		grep -v ^docker | \
		grep -v ^libs | \
		$(TAR_CMD) -cf docker/libs/code.tar -T -

	@docker build \
		--tag ${IMAGE}:${TAG} \
		--build-arg=libdir=libs \
		docker/

.PHONY: deploy-docker-image
deploy-docker-image: build-docker-image
	docker tag \
		${IMAGE}:${TAG} \
		${REPO}/${IMAGE}:${TAG}
	docker push ${REPO}/${IMAGE}:${TAG}
