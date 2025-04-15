# Sử dụng image Python nhẹ
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements.txt vào container
COPY requirements.txt .

# Cài đặt các package cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Chạy lệnh mặc định
CMD ["python", "train.py"]
