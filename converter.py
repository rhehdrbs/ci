import tensorflow as tf

# 모델 파일 경로
model_path = "H:/3학년/공동연구/실습파일/model.h5"

# 모델 로드
model = tf.keras.models.load_model(model_path)

# 변환할 TFLite 모델 파일 경로
tflite_model_path = "H:/3학년/공동연구/실습파일/model.tflite"

# 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 모델 변환
tflite_model = converter.convert()

# 변환된 모델을 파일로 저장
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

