# Phase 2A AMD Docker migration

The AMD environment entrypoint is now shared by Docker and Apptainer. Use
[`phase2-amd-environment.md`](phase2-amd-environment.md) for the current
workflow, persistent paths, assets, cache, GPU target handling and offline
Apptainer setup.

The old `docker/phase2a-amd.sh` command remains as a compatibility wrapper and
prints a deprecation notice. This document intentionally does not duplicate the
environment instructions.
