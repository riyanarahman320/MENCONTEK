import os

# Gunakan 'uvicorn' sebagai worker
worker_class = "uvicorn.workers.UvicornWorker"

# Jumlah worker. '1' adalah awal yang aman untuk paket Basic/Pro
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))

# Port yang akan didengarkan. DigitalOcean akan mengatur ini.
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Timeout untuk request yang lama (penting untuk model ML)
timeout = 120  # 120 detik
