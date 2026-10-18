# Docker

Docker packages an application and everything it needs into an **image**: a
stack of read-only layers built from a `Dockerfile`. Running an image gives you
a container, which is just a process with its own filesystem view, network
namespace and resource limits.

## Why it matters

- The same image runs on a laptop, in CI and in production, so "works on my
  machine" stops being an excuse.
- Layers are cached, so rebuilding after a small code change is fast as long
  as the slow steps (installing dependencies) come first in the Dockerfile.
- Containers share the host kernel. They are much lighter than virtual
  machines, but the isolation boundary is weaker.

## Notes to self

Keep images small: multi-stage builds, a slim base image, no build tools in the
final stage. Pin base image digests for reproducible builds.
