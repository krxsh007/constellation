# CI CD Pipelines

Continuous integration means every push is built and tested automatically, so
broken changes are caught minutes after they are written instead of days.
Continuous delivery extends the pipeline all the way to a deployable artifact.

A typical pipeline for a service:

1. Lint and run the unit tests.
2. Build a container image and tag it with the commit SHA.
3. Push the image to a registry.
4. Deploy to staging by upgrading the service's Helm release with the new
   image tag; promote to production after smoke tests pass.

Keep the pipeline fast (under ten minutes) or people start batching changes,
which defeats the point.
