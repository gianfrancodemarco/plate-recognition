FROM prom/prometheus

COPY ./prometheus.yaml /etc/prometheus/prometheus.yaml
COPY ./resolve_targets.sh ./resolve_targets.sh 
COPY ./run_prometheus.sh ./run_prometheus.sh 

ENTRYPOINT [ "sh", "./run_prometheus.sh" ] 
