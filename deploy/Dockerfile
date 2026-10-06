FROM mnemory:base
WORKDIR /app
COPY pyproject.toml README.md ./
COPY mnemory/ mnemory/
RUN pip install --no-deps --no-cache-dir .
CMD ["mnemory"]
