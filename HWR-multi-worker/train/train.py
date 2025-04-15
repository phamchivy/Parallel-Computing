import numpy as np
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
import time
from datetime import datetime
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

np.set_printoptions(precision=2)

# ✅ Load biến TF_CONFIG từ Docker environment
tf_config = os.getenv("TF_CONFIG")

if tf_config:
    try:
        tf_config = json.loads(tf_config)  # Chuyển từ chuỗi JSON sang dict
        print("Using distributed training with TF_CONFIG:", tf_config)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    except json.JSONDecodeError:
        print("⚠️ Warning: Invalid TF_CONFIG format, switching to single-worker mode")
        strategy = tf.distribute.get_strategy()
else:
    print("Using single-worker training")
    strategy = tf.distribute.get_strategy()

# ✅ Load bộ dữ liệu MNIST
(X_full, y_full), (X_test, y_test) = mnist.load_data()

# ✅ Tiền xử lý: flatten + normalize
X_full = X_full.reshape(-1, 28*28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28*28).astype("float32") / 255.0

# Chuyển y về shape (n, 1) cho đồng bộ
y_full = y_full.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Bước 1: Tách ra tập Test trước (20% dữ liệu)
X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Bước 2: Từ phần còn lại, tách tiếp ra Validation (20% của 80% còn lại = 16%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print("✅ MNIST loaded thành công!")
print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ✨ CHUYỂN train data sang tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# ✨ Shard theo số worker hiện có
num_workers = strategy.num_replicas_in_sync
task_id = strategy.cluster_resolver.task_id

train_dataset = train_dataset.shard(num_shards=num_workers, index=task_id)

print(f"🔧 Worker {task_id}/{num_workers} — Dataset size after shard: {len(list(train_dataset.as_numpy_iterator()))}")


# ✨ Shuffle, batch và prefetch
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


tf.random.set_seed(1234) # for consistent results

# ✅ Định nghĩa model trong `strategy.scope()`
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),     
        tf.keras.layers.Dense(25, activation='relu', name="L1"),
        tf.keras.layers.Dense(15, activation='relu', name="L2"),
        tf.keras.layers.Dense(10, activation='linear', name="L3")
    ], name="my_model")

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],  # ✅ thêm dòng này để lấy accuracy
    )

model.summary()

start_time = time.time()

# ✅ Train model với `verbose=2` để giảm log
history = model.fit(train_dataset, epochs=40, validation_data=(X_val, y_val), verbose=2)

end_time = time.time()
elapsed_time = end_time - start_time

start_dt = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
end_dt = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

# ✅ Hiển thị kết quả dự đoán với tập test để đánh giá khách quan

# Default tên ảnh
worker_name = "worker_unknown"

if tf_config:
    try:
        task = tf_config.get("task", {})
        worker_type = task.get("type", "worker")
        worker_index = task.get("index", 0)
        worker_name = f"{worker_type}_{worker_index}"
    except:
        pass

# Đường dẫn ảnh đầu ra riêng biệt
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f"predictions_{worker_name}.png")

# ✅ Đánh giá độ chính xác trên tập test
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
accuracy_percent = test_accuracy * 100
print(f"🎯 Độ chính xác trên tập test: {accuracy_percent:.2f}%")

# ✅ Hiển thị kết quả dự đoán
m_test = X_test.shape[0]
fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])

for i, ax in enumerate(axes.flat):
    # Chọn ngẫu nhiên ảnh từ tập test
    random_index = np.random.randint(m_test)
    
    X_random_reshaped = X_test[random_index].reshape((28, 28)).T
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Dự đoán
    prediction = model.predict(X_test[random_index].reshape(1, 784), verbose=0)
    prediction_p = tf.nn.softmax(prediction).numpy()
    yhat = np.argmax(prediction_p)
    
    # Hiển thị nhãn thực và nhãn dự đoán
    ax.set_title(f"{y_test[random_index, 0]},{yhat}", fontsize=10)
    ax.set_axis_off()

fig.suptitle(f"Label (true), yhat (pred) — Accuracy: {accuracy_percent:.2f}%", fontsize=14)
plt.savefig(output_path)

print(f"✅ Worker lưu ảnh vào: {output_path}%")

print(f"🧠 Worker {worker_name} huấn luyện từ {start_dt} đến {end_dt} mất {elapsed_time:.2f} giây")


