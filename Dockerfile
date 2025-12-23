# 1. Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 4. Install Python dependencies
COPY requirements.txt .
# We remove 'numpy' from the pip install here because requirements.txt handles it now
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy source code
COPY src/ ./src/

# 6. CRITICAL FIX: Add 'src' to Python Path so imports work
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 7. Expose and Run
EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]