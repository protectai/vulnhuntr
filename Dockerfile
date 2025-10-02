FROM python:3.10-bookworm

WORKDIR /usr/src/vulnhuntr
COPY . .
RUN pip install --no-cache-dir --timeout 600 --retries 10 .

ENTRYPOINT [ "vulnhuntr" ]
