FROM python:3.10-bookworm
LABEL org.opencontainers.image.source="https://github.com/protectai/vulnhutr"

WORKDIR /usr/src/vulnhuntr
COPY . .
RUN pip install --no-cache-dir .

ENTRYPOINT [ "vulnhuntr" ]
