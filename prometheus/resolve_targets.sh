echo "Resolving hosts in /etc/prometheus/prometheus.yml"
sed -i -e "s/PLATE_RECOGNITION_APP_HOST/$PLATE_RECOGNITION_APP_HOST/" /etc/prometheus/prometheus.yml
echo "Hosts resolved"
