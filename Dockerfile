FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Use uvicorn CLI directly — avoids __main__ block entirely.
# OpenEnv evaluator controls PORT via environment variable.
CMD ["sh", "-c", "uvicorn inference:app --host 0.0.0.0 --port ${PORT:-7860}"]