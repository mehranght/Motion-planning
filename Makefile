SHELL := /bin/bash
.ONESHELL:

.PHONY: build
build:
	docker build -t motion-planning --file docker/Dockerfile .

.PHONY: run
run:
	./docker/gui-docker --rm -it motion-planning /bin/bash
