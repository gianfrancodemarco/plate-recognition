## Grafana custom Docker image

Grafana would need persistent storage to save all of its configurations.
Since the infrastructure is run on CloudRun, which is effimeral by definition, no persistent storage is provided.

This custom image uses the provisioning function of Grafana to provision dashboards and datasources at startup.
When creating or editing a new dashboard or datasource, export its JSON representation into the correct folder, so that it will be included in the Grafana image.

## Environment variables
These envrionment variables are required to be set on the container:
- PROMETHEUS_SCHEMA
- PROMETHEUS_HOST
- PROMETHEUS_PORT