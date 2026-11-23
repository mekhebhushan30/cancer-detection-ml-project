cat << 'EOF' > Dockerfile
# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Industry Standard: Mount AWS credentials securely in memory (Zero trace in image/history)
RUN --mount=type=secret,id=aws_key \
    --mount=type=secret,id=aws_secret \
    AWS_ACCESS_KEY_ID=$(cat /run/secrets/aws_key) \
    AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/aws_secret) \
    AWS_DEFAULT_REGION=eu-north-1 \
    python development/main.py

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF
