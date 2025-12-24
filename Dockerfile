# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Configure a non-root user (Required for Hugging Face Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy requirements file first to leverage Docker cache
COPY --chown=user ./requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose the port expected by HF Spaces
EXPOSE 7860

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
