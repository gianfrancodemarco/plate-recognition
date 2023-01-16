echo "Resolving hosts in /etc/prometheus/prometheus.yaml"
echo "PLATE_RECOGNITION_APP_HOST: $PLATE_RECOGNITION_APP_HOST"
sed -i -e "s#PLATE_RECOGNITION_APP_HOST#$PLATE_RECOGNITION_APP_HOST#" /etc/prometheus/prometheus.yaml
sed -i -e "s#TELEGRAM_BOT_HOST#$TELEGRAM_BOT_HOST#" /etc/prometheus/prometheus.yaml
echo "Hosts resolved"
