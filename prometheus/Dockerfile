FROM prom/prometheus

COPY ./prometheus.yaml /etc/prometheus/prometheus.yml
COPY ./substitute_target.sh ./substitute_target.sh 
COPY ./run_prometheus.sh ./run_prometheus.sh 

ENTRYPOINT [ "sh", "./run_prometheus.sh" ] 
#RUN sed -i '' -e "s/HOST/ASD/" /etc/prometheus/prometheus.yml
