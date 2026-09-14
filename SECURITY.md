# Security

Please report suspected security issues privately to `unnamets@gmail.com` with
a concise description, affected monorepo revision, and reproduction details.
Do not open a public issue or patch containing credentials, private hostnames or
paths, scheduler/account identifiers, model or dataset assets, bags, traces,
container credentials, or site-specific logs.

GPU ROS does not ship runtime images, credentials, model weights, datasets, or
external checkout contents. Reports involving ROS, CUDA, ROCm, ONNX Runtime,
NVIDIA/AMD drivers, model providers, or datasets may also need to be sent to
the responsible upstream provider; include only the minimum public details in
the report to this project.

When sharing a reproduction, use a fresh external state directory and redact
`OVG_*` values that identify a private deployment. Preserve the one-repository
contract and do not weaken provider, digest, permission, device, or asset gates
to make a run succeed. See `CONTRIBUTING.md` for ordinary changes and
`docs/experiments/amd-runtime-contract.md` for the public runtime boundary.
