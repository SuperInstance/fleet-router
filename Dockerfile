FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8100
CMD ["python3", "-m", "fleet_router.cli", "--host", "0.0.0.0", "--port", "8100"]
