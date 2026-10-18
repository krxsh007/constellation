# Helm Charts

Helm is a package manager for Kubernetes. A chart is a folder of templated
YAML manifests plus a `values.yaml` with the knobs a user can turn.

- `helm install` renders the templates with your values and applies them as a
  named **release**.
- Every upgrade creates a new revision, so `helm rollback my-app 3` restores
  the manifests of revision 3 in one command.
- Templates use Go templating. Keep logic out of them; deep `if` / `range`
  nesting makes charts hard to review.

I use one chart per service and an umbrella chart per environment.
