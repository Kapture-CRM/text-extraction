# gunicorn.conf.py
from prometheus_client import multiprocess

def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)

bind = "127.0.0.1:8500"
workers = 10
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
raw_env = ["FORWARDED_ALLOW_IPS=*"]

# Logging
accesslog = "/var/log/text-extraction/access.log"
errorlog = "/var/log/text-extraction/error.log"
loglevel = "info"