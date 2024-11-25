import os
import shutil
import random

def split_data(base_dir, train_dir, test_dir, test_size=0.2):
    # 각 숫자 폴더에 대해 작업
    animal_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for animal in animal_folders:
        animal_folder = os.path.join(base_dir, animal)

        # train, test 폴더 생성 (동물별로 train/test 폴더를 생성)
        animal_train_dir = os.path.join(train_dir, animal)  # 동물별 train 폴더 경로
        animal_test_dir = os.path.join(test_dir, animal)    # 동물별 test 폴더 경로

        # 폴더가 없다면 생성
        os.makedirs(animal_train_dir, exist_ok=True)
        os.makedirs(animal_test_dir, exist_ok=True)

        # 클래스 내의 이미지 파일 목록 가져오기
        image_files = []

        # os.walk()로 하위 폴더까지 탐색하여 이미지 파일들 찾기
        for root, dirs, files in os.walk(animal_folder):
            # 이미지 파일만 필터링 (확장자에 맞는 파일들만)
            image_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])

        if not image_files:
            print(f"Warning: No images found in '{animal_folder}'. Skipping this folder.")
            continue  # 이미지가 없으면 해당 동물을 건너뜀

        # 이미지 파일을 랜덤하게 섞기
        random.shuffle(image_files)

        # 테스트 데이터의 개수 계산
        num_test = int(len(image_files) * test_size)
        
        # 테스트 데이터와 트레이닝 데이터 분리
        test_files = image_files[:num_test]
        train_files = image_files[num_test:]

        # 테스트 파일을 test 폴더로 복사
        for file in test_files:
            shutil.copy(file, animal_test_dir)

        # 트레이닝 파일을 train 폴더로 복사
        for file in train_files:
            shutil.copy(file, animal_train_dir)

        # 복사된 파일 확인 (디버깅)
        print(f"Class '{animal}': {len(train_files)} training images, {len(test_files)} test images")
        print(f"Train folder '{animal_train_dir}' contains {len(os.listdir(animal_train_dir))} images.")
        print(f"Test folder '{animal_test_dir}' contains {len(os.listdir(animal_test_dir))} images.")

# 기본 디렉토리 경로
base_dir = './dataset_drawNum'  # 원본 이미지 디렉토리
train_dir = './dataset_drawNum/train'   # 트레이닝 데이터 저장 디렉토리
test_dir = './dataset_drawNum/test'     # 테스트 데이터 저장 디렉토리

# 트레이닝 이미지와 테스트 이미지로 분리
split_data(base_dir, train_dir, test_dir, test_size=0.2)
