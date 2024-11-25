import os
import numpy as np
import cv2
import torch 
from ultralytics import YOLO
import tensorflow as tf
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.neighbors import KernelDensity


# GPU 설정 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: 물체 인식 (YOLOv8 모델 사용)
def object_recognition(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    # 모델 로드
    model = YOLO('yolov8n.pt')
    model.to(device)  # GPU로 이동
    
    # 모델 실행
    results = model(image_path)
    
    if len(results[0].boxes) == 0:
        print(f"No objects detected in {image_path}")
        return np.array([]), [], None
    
    # YOLO 결과에서 이미지의 원본 크기 얻기
    orig_shape = results[0].orig_shape  # 결과에서 원본 이미지 크기 가져오기
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 박스 좌표 정보 가져오기
    
    # 원본 이미지를 읽어서 반환
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 읽으므로 RGB로 변환
    image = image / 255.0  # 0-255 범위의 값을 0-1 범위로 정규화

    # 정규화 후 float64 타입이 되므로, 이를 uint8로 변환
    image = (image * 255).astype(np.uint8)

    # orig_shape가 튜플 형태인지 체크하고, 튜플로 반환되지 않으면 수정
    if isinstance(orig_shape, tuple):
        height, width = int(orig_shape[0]), int(orig_shape[1])
    else:
        # 만약 단일 값으로 반환되면 이를 높이와 너비로 처리
        height, width = int(orig_shape), int(orig_shape)
    
    return (height, width), boxes, image  # 이미지도 반환

# Step 2: 물체 크기 측정 및 색, 윤곽선 추출
def extract_features(image, boxes):
    features = []
    height, width = image.shape[0], image.shape[1]  # 이미지 크기 사용
    
    for box in boxes:
        top_left = (max(0, int(box[0])), max(0, int(box[1])))  # 괄호 수정
        bottom_right = (min(width, int(box[2])), min(height, int(box[3])))  # 이미지의 높이와 너비 사용
        object_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        if object_region.size == 0:
            print("Empty object region detected. Skipping.")
            continue

        try:
            hsv_image = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)  # 색상 영역
            hue_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
            saturation_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
            value_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

            gray_image = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0

            features.append([ 
                hue_hist.mean(),  # 색상
                saturation_hist.mean(),  # 채도
                value_hist.mean(),  # 밝기
                bottom_right[0] - top_left[0],  # 가로 길이
                bottom_right[1] - top_left[1],  # 세로 길이
                edge_density  # 윤곽선
            ])
        except Exception as e:
            print(f"Error extracting features: {e}")
    return features


# Step 4: 샘플 가중치 계산
def get_sample_weights(importance_scores):
    weights = []
    for score in importance_scores:
        # 1~0 범위를 3~1로 변환
        normalized_score = (score * 2) + 1
        
        if normalized_score > 2.7:
            weights.append(3)
        elif normalized_score > 2.2:
            weights.append(2.5)
        elif normalized_score > 1.7:
            weights.append(2)
        elif normalized_score > 1.2:
            weights.append(1.5)
        else:
            weights.append(1)
    return np.array(weights)

# Step 5: 학습 데이터 처리 및 모델 학습 (엑셀 기반 초기 중요도 + 가중치 변화량 반영)
def process_animal_images(base_dir, excel_file_path):
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    try:
        excel_data = pd.read_excel(excel_file_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

    # 클래스 폴더들 찾기
    animal_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    if len(animal_folders) == 0:
        raise ValueError(f"No animal class folders found in directory: {base_dir}")

    # 모델 정의
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Dense(128, activation='relu', input_shape=(6,), name="dense_layer_1"),
        tf.keras.layers.BatchNormalization(),  # 배치 정규화 추가
        tf.keras.layers.Dropout(0.2),  # 드롭아웃 추가 (20% 비율로 드롭)
        
        tf.keras.layers.Dense(64, activation='relu', name="dense_layer_2"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu', name="dense_layer_3"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(16, activation='relu', name="dense_layer_4"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(8, activation='relu', name="dense_layer_5"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(4, activation='relu', name="dense_layer_6"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # 다중 클래스 분류를 위한 소프트맥스 출력층
        tf.keras.layers.Dense(len(animal_folders), activation='softmax', name="output_layer"),  
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )

    features_list = []
    labels_list = []
    sample_weights_list = []
    class_names = {folder: idx for idx, folder in enumerate(animal_folders)}

    # 엑셀 저장을 위한 리스트
    all_features_with_labels = []
    
    # 첫 번째 반복에서 엑셀 데이터를 사용하여 중요도를 설정
    importance_scores = {}

    # 데이터 확인: 학습 데이터 무결성 확인
    for animal in animal_folders:
        animal_dir = os.path.join(base_dir, animal)
        print(f"Processing images in: {animal_dir}")

        for img_file in os.listdir(animal_dir):
            img_path = os.path.join(animal_dir, img_file)
            if img_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):  # 이미지 파일 확인
                try:
                    orig_shape, boxes, image = object_recognition(img_path)
                    if image is None:
                        print(f"No image loaded for {img_path}, skipping.")
                        continue
                    if len(boxes) == 0:
                        print(f"No objects detected in {img_path}, skipping.")
                        continue
                    features = extract_features(image, boxes)  # 원본 이미지 전달

                    # 동물 라벨과 함께 features 저장
                    features_with_labels = [[class_names[animal]] + feature for feature in features]
                    all_features_with_labels.extend(features_with_labels)  # 엑셀에 저장될 리스트에 추가

                    if len(features) == 0:
                        print(f"No features extracted from {img_path}, skipping.")
                        continue

                    # 엑셀에서 해당 이미지 경로에 맞는 Importance 값을 찾기
                    image_path = os.path.join(base_dir, animal, img_file)  # 경로 구분자 통일

                    # 이미지 경로 문제 확인
                    if not os.path.exists(img_path):
                        print(f"Image file does not exist: {img_path}")

                    image_importance = excel_data[excel_data['Image Path'] == image_path]['Importance'].values

                    if len(image_importance) > 0:
                        importance_scores[img_path] = image_importance[0]  # 중요도 저장
                    else:
                        importance_scores[img_path] = 0  # 찾을 수 없으면 기본값 0

                    sample_weights = get_sample_weights([importance_scores[img_path]] * len(features))
                    features_list.extend(features)
                    labels_list.extend([class_names[animal]] * len(features))
                    sample_weights_list.extend(sample_weights)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    sample_weights_array = np.array(sample_weights_list)

    if features_array.shape[0] == 0:
        raise ValueError("No valid features extracted for training.")

    # 모델 학습
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(features_array, labels_array, sample_weight=sample_weights_array, validation_split=0.2, epochs=50, callbacks=[early_stopping])

    # 중요도 계산 및 가중치 갱신 반복
    for epoch in range(1, 51):  # 학습 반복
        print(f"Epoch {epoch}")

        # 가중치 변화량 기록 (예시로 간단히 weight_changes 리스트에 추가)
        weight_changes = []  # 가중치 변화량 기록
        for img_file in os.listdir(base_dir):
            img_path = os.path.join(base_dir, img_file)
            if img_path in importance_scores:
                # 각 데이터의 가중치 변화량을 계산 (변화량 추적)
                weight_changes.append(importance_scores[img_path])  # 여기서 가중치 변화량 계산 필요

        if len(weight_changes) > 0:  # weight_changes가 비어 있지 않은지 확인
            # 가중치 변화량 합에 대한 분포도 계산
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(weight_changes).reshape(-1, 1))
            densities = np.exp(kde.score_samples(np.array(weight_changes).reshape(-1, 1)))

            # 밀도를 1~3 범위로 스케일링
            min_density = np.min(densities)
            max_density = np.max(densities)
            scaled_importance = 1 + 2 * ((densities - min_density) / (max_density - min_density))

            # 밀도가 높은 변화량을 가진 부분을 새롭게 중요도로 설정
            for img_file in os.listdir(base_dir):
                img_path = os.path.join(base_dir, img_file)
                if img_path in importance_scores:
                    # 변화량에 대한 밀도 기반 중요도 갱신
                    importance_scores[img_path] = scaled_importance  # 밀도에 맞는 중요도 설정

        else:
            print(f"No weight changes recorded for epoch {epoch}, skipping KDE.")

        # 새로 계산된 중요도 기반으로 가중치 갱신
        new_sample_weights = get_sample_weights(list(importance_scores.values()))
        model.fit(features_array, labels_array, sample_weight=new_sample_weights, validation_split=0.2, epochs=50, callbacks=[early_stopping])

    # 엑셀 파일로 저장
    save_features_to_excel(all_features_with_labels, filename='./exel/features_with_labels.xlsx')

    return model



# Save features with labels to Excel function
def save_features_to_excel(features, filename='./exel/features_with_labels.xlsx'):
    # Add headers: Animal Label, Hue Mean, Saturation Mean, Value Mean, Width, Height, Edge Density
    features_df = pd.DataFrame(features, columns=['Animal Label', 'Hue Mean', 'Saturation Mean', 'Value Mean', 'Width', 'Height', 'Edge Density'])
    features_df.to_excel(filename, index=False)
    print(f"Features with labels saved to {filename}")

def evaluate_and_predict(model, test_dir):
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory does not exist: {test_dir}")

    test_features = []
    test_labels = []
    class_names = {folder: idx for idx, folder in enumerate(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, folder))}

    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):  # 이미지 파일 확인
                    try:
                        orig_shape, boxes, image = object_recognition(img_path)
                        if image is None:
                            print(f"No image loaded for {img_path}, skipping.")
                            continue
                        if len(boxes) == 0:
                            print(f"No objects detected in {img_path}, skipping.")
                            continue
                        features = extract_features(image, boxes)  # 원본 이미지 전달
                        if len(features) == 0:
                            print(f"No features extracted from {img_path}, skipping.")
                            continue
                        test_features.extend(features)
                        test_labels.extend([class_names[folder]] * len(features))
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

    test_features_array = np.array(test_features)
    test_labels_array = np.array(test_labels)

    if test_features_array.shape[0] == 0:
        raise ValueError("No valid test features extracted.")

    loss, accuracy = model.evaluate(test_features_array, test_labels_array, verbose=1)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    predictions = model.predict(test_features_array, verbose=1)

    # predicted_labels = np.argmax(predictions, axis=1)
    # print("\nPrediction results:")
    # for i, prediction in enumerate(predicted_labels):
    #     print(f"Predicted: {prediction}, Actual: {test_labels_array[i]}")

# 실행
base_dir = './animal_images/animals/train'
test_dir = './animal_images/animals/test'
model = process_animal_images(base_dir, "./exel/image_importance_labels.xlsx")
evaluate_and_predict(model, test_dir)  # test_dir 인자를 추가하여 함수를 호출합니다.
