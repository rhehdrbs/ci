import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 데이터셋 경로 설정
dataset_path = "D:\\3학년\\공동연구\\실습파일\\Citrus\\Leaves"

# 이미지 크기 설정
img_size = (150, 150)

# 이미지 파일과 라벨을 저장할 리스트 초기화
data = []
labels = []

# 이미지 파일 경로와 라벨을 추출하여 리스트에 추가
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        data.append(img)
        labels.append(label)

# LabelEncoder를 사용하여 라벨을 정수값으로 매핑
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# 리스트를 넘파이 배열로 변환
data = np.array(data, dtype='float32')
labels = np.array(labels)

# 데이터셋을 훈련용과 검증용으로 나누기
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 이미지 데이터 전처리
train_data = train_data / 255.0
val_data = val_data / 255.0

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['accuracy'])

# 모델 훈련
history = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))

'''
#모델 평가
loss, accuracy = model.evaluate(val_data, val_labels)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

#손실, 정확도 값 가져오기
history = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
loss = history.history['loss']
accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# 손실 그래프
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 정확도 그래프
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''

#모델 저장
model.save("D:\\3학년\\공동연구\\실습파일\\model.h5")
