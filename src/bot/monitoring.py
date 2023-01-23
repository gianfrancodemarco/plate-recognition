import prometheus_client as prom

MESSAGES_COUNTER = prom.Counter('number_of_messages', 'The total number of messages received by the bot')
PHOTOS_COUNTER = prom.Counter('number_of_photos', 'The total number of photos received by the bot')

def start_monitoring():
    prom.start_http_server(8080)

def increase_messages_counter():
    MESSAGES_COUNTER.inc(1)

def increase_photos_counter():
    PHOTOS_COUNTER.inc(1)
