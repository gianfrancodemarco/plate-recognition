## Prometheus custom Docker image

Prometheus doesn't allow to define targets based on environment variables.
This isn't convenient because changing the host of a target would require rewriting the Prometheus configuration.

To avoid this problem, a custom Prometheus image is created.
In the prometheus.yaml, in place of the target actual host there is a placeholder.

The custom Docker entrypoint runs a script that:
- resolves the hosts in the prometheus.yaml using environment variables
- starts the Prometheus service

## Environment variables
These envrionment variables are required to be set on the container:
- PLATE_RECOGNITION_APP_HOST

### TODOs
- Make the host target resolution dynamic instead of having to specify each host by hand
