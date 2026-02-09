사용해본 증강: v2의 RandomHorizontalFlip(수평전환, 상하), RandomVerticalFlip(수직전환, 좌우), RandomAffine(이미지 각도, 위치 조정 등), RandomPhometricDistort(ColorJitter 기반), ColorJitter(색상관련: 대비, 밝기 등), RandomResize(사이즈조정), Normalize(정규화)

+RandomApply 활용해봄(몇 %확률로 적용할지)

사용한 코드:

import torchvision.transforms.v2 as T

train_transform = T.Compose([
    T.ToDtype(torch.float32, scale=True),        # [0,255] → [0,1]
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomApply([
        T.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            scale=(0.9, 1.3),
            shear=5,
    )], p=0.5),
    T.RandomApply([
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.02,
        )
    ], p=0.5),
    T.RandomPhotometricDistort(
        brightness=(0.95, 1.05),
        contrast=(0.8, 1.2),
        saturation=(0.9, 1.1),
        hue=(-0.02, 0.02),
        p=0.5,
    ),
    T.RandomResize(min_size=640, max_size=800),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

test_transform = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Resize(640),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

p = 이 부분이 확률 결정함(0.5면 50% 확률로 적용)

색상 관련: RandomPhometricDistor, ColorJitter 크게 흔들면 성능 많이 떨어짐. 작게 적용해도 성능이 좋지 못했음

위치, 각도, 방향 조정: Random Horizontal/Vertical Flip, RandomAffine 적용하면 오히려 성능 떨어짐

사이즈 조정: Kaggle에 올리면 0점이 나와 제외함

—> 정규화 제외 모두 적용하지 않기로 결정함