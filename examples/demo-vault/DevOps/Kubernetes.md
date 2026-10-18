# Kubernetes

Kubernetes runs containers across a cluster of machines. You describe the
desired state (three replicas of this image, behind this service) and the
control plane keeps reconciling reality towards it.

## Core pieces

- **Pods** are the smallest unit: one or more containers sharing a network
  namespace. The scheduler places pods onto nodes based on resource requests,
  affinity rules and taints.
- **Deployments** manage rolling updates of pods.
- **etcd** stores the whole cluster state. It is a small, strongly consistent
  key-value store, and if it loses quorum the control plane stops accepting
  changes.

## Gotchas

Resource requests drive scheduling but limits drive throttling, so set both.
Liveness probes that are too aggressive cause restart loops under load.
