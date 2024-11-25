import os
import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# GPU 설정 확인
device = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"

# Step 1: 이미지 분류 모델 생성
def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # 출력 차원 설정
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])  # from_logits=False 설정
    return model

# Step 2: 이미지 데이터 전처리 및 학습
def train_model(base_dir):
    # 데이터 전처리
    image_size = (128, 128)
    batch_size = 32
    # 데이터 증강기 설정
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(base_dir, target_size=image_size, batch_size=batch_size, subset='training', class_mode='sparse')
    val_data = datagen.flow_from_directory(base_dir, target_size=image_size, batch_size=batch_size, subset='validation', class_mode='sparse')

    # num_classes가 정확히 지정되어 있는지 확인
    num_classes = len(train_data.class_indices)

    model = create_model(input_shape=(128, 128, 3), num_classes=num_classes)

    # 모델 확인
    model.summary()

    # 학습
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[early_stopping])

    return model, train_data

# Step 3: 예측 결과를 바탕으로 중요도 계산
def calculate_importance(model, test_data):
    importance_list = []
    
    import os
import cv2
import numpy as np

def calculate_importance(model, train_data):
    importance_list = []
    
    # train_data에서 이미지 경로 추출
    for img_path in train_data.filepaths:
        # 이미지 경로 출력 (디버깅)
        print(f"Loading image: {img_path}")
        
        # 경로가 문자열인지 확인
        if not isinstance(img_path, str):
            print(f"Error: img_path is not a string: {img_path}")
            continue
        
        # 경로가 실제 파일인지 확인
        if not os.path.exists(img_path):
            print(f"Error: Image file does not exist: {img_path}")
            continue
        
        # 이미지 읽기
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error loading image: {img_path}")
            continue  # 이미지가 없으면 넘어감

        image = cv2.resize(image, (128, 128))
        image = np.expand_dims(image, axis=0) / 255.0  # 정규화

        # 예측 수행
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        predicted_prob = predictions[0][predicted_class]  # 예측 확률
        
        # 중요도 계산 (a=3, e=3, c=2, b,d=1, 연속적 배정)
        if predicted_prob >= 0.9:
            importance = 3  # 예측 확률이 0.9 이상이면 중요도 3
        elif predicted_prob <= 0.1:
            importance = 3  # 예측 확률이 0.1 이하이면 중요도 3
        elif 0.1 < predicted_prob <= 0.2:  # b 구간
            importance = 1 + (predicted_prob - 0.1) * 5  # 선형 변화
        elif 0.2 < predicted_prob <= 0.4:  # b 구간 계속 선형 변화
            importance = 1 + (predicted_prob - 0.2) * 2.5  # 선형 변화
        elif 0.4 < predicted_prob <= 0.6:  # c 구간
            importance = 2  # c 구간은 중요도 2
        elif 0.6 < predicted_prob <= 0.8:  # d 구간 계속 선형 변화
            importance = 1 + (0.8 - predicted_prob) * 2.5  # 선형 변화
        elif 0.8 < predicted_prob < 1:  # d 구간 계속 선형 변화
            importance = 1 + (0.9 - predicted_prob) * 5  # 선형 변화
        else:
            importance = 3  # e 구간

        importance_list.append((img_path, importance))
    
    return importance_list


# Step 4: Importance 라벨 추가
def assign_importance_labels(base_dir, importance_list):
    try:
        # 엑셀 파일을 저장할 경로 지정
        output_path = os.path.join("./exel", 'image_importance_labels.xlsx')
        
        # 디렉토리가 존재하는지 확인
        if not os.path.exists(base_dir):
            print(f"Error: Directory does not exist - {base_dir}")
            return

        # 엑셀 파일 저장
        importance_df = pd.DataFrame(importance_list, columns=['Image Path', 'Importance'])
        importance_df.to_excel(output_path, index=False)
        print(f"Importance labels saved to {output_path}")
    
    except Exception as e:
        print(f"Error saving Excel file: {e}")

# 학습 전에 데이터 차원 확인
def check_data_dimensions(train_data, val_data):
    # 훈련 데이터 차원 확인
    for data_batch, label_batch in train_data:
        print(f"Training batch data shape: {data_batch.shape}")
        print(f"Training batch label shape: {label_batch.shape}")
        break

    # 검증 데이터 차원 확인
    for data_batch, label_batch in val_data:
        print(f"Validation batch data shape: {data_batch.shape}")
        print(f"Validation batch label shape: {label_batch.shape}")
        break

# 데이터 차원 확인
base_dir = './animal_images/animals/train'
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(base_dir, target_size=(128, 128), batch_size=32, subset='training', class_mode='sparse')
val_data = datagen.flow_from_directory(base_dir, target_size=(128, 128), batch_size=32, subset='validation', class_mode='sparse')

check_data_dimensions(train_data, val_data)

# 모델 학습
model, train_data = train_model(base_dir)

# 중요도 계산
importance_list = calculate_importance(model, train_data)

# 중요도 라벨 저장
assign_importance_labels(base_dir, importance_list)
