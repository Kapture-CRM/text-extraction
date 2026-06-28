# gunicorn.conf.py

bind = "127.0.0.1:8000"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5

# Logging
accesslog = "/var/log/your-service/access.log"
errorlog = "/var/log/your-service/error.log"
loglevel = "info"