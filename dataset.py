import os
import math
import json
import warnings

import cv2
import albumentations as A
from PIL import Image
from skimage.util import random_noise # Add noise : peper method

import numpy as np
from torch.utils.data import Dataset
from shapely.geometry import Polygon

# Calculate the Euclidean distance
def cal_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


''' [Ln 24 ~ 59]
- Text area의 모서리를 안쪽으로 수축시킴
- 사각형의 두 점을 이동시켜서 경계를 축소함
'''
def move_points(vertices, index1, index2, r, coef):
    """Move two points to shrink the edge of a text region.

    Args:
        vertices (np.ndarray): Text region vertices of shape (8,)
                             Format: [x1,y1, x2,y2, x3,y3, x4,y4]
        index1 (int): Index of first point to move (0-3)
        index2 (int): Index of second point to move (0-3)
        r (list): Shrink ratios for each corner [r1,r2,r3,r4]
        coef (float): Global shrink coefficient (0-1)

    Returns:
        np.ndarray: Modified vertices array

    Raises:
        ValueError: If input parameters are invalid
    """
    index1, index2 = index1 % 4, index2 % 4
    x1_index, x2_index = index1 * 2 + 0, index2 * 2 + 0
    y1_index, y2_index = index1 * 2 + 1, index2 * 2 + 1

    r1, r2 = r[index1], r[index2]

    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])

    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y

    return vertices


''' [Ln 68 ~ ]
- 텍스트 영역을 나타내는 4개의 꼭짓점 좌표를 입력받아 내부로 축소하는 함수입니다
- 각 모서리에서 가장 짧은 변의 길이를 기준으로 축소 비율을 적용합니다
- 긴 변을 기준으로 offset을 결정하여 축소 순서를 정합니다
'''
def shrink_poly(vertices, coef=0.3):
    """
    텍스트 영역을 축소하는 함수

    Args:
        vertices (list): [x1,y1,x2,y2,x3,y3,x4,y4] 형식의 텍스트 영역 꼭짓점 좌표
        coef (float): 축소 비율 (기본값: 0.3)

    Returns:
        list: 축소된 텍스트 영역의 꼭짓점 좌표
    """
    if len(vertices) != 8:
        raise ValueError("vertices must contain exactly 8 coordinates (x1,y1,x2,y2,x3,y3,x4,y4)")
    if not 0 < coef < 1:
        raise ValueError("coefficient must be between 0 and 1")

    # 꼭짓점 좌표 추출
    corners = [(vertices[i], vertices[i+1]) for i in range(0, 8, 2)]
    x1,y1, x2,y2, x3,y3, x4,y4 = [coord for point in corners for coord in point]

    # 각 변의 최소 길이 계산
    def cal_distance(x1, y1, x2, y2):
        return ((x2-x1)**2 + (y2-y1)**2)**0.5

    r = [
        min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4)),  # r1
        min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3)),  # r2
        min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4)),  # r3
        min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))   # r4
    ]

    # 긴 변 판별을 위한 offset 계산
    long_edges_1 = cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4)
    long_edges_2 = cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4)
    offset = 0 if long_edges_1 > long_edges_2 else 1

    def move_points(v, p1_idx, p2_idx, r, coef):
        """
        두 점 사이의 거리를 축소하는 보조 함수
        """
        p1_idx = p1_idx % 4 * 2
        p2_idx = p2_idx % 4 * 2

        # 두 점 사이의 벡터 계산
        dx = v[p2_idx] - v[p1_idx]
        dy = v[p2_idx + 1] - v[p1_idx + 1]

        # 거리에 따른 이동량 계산
        move_x = dx * coef * r[p1_idx//2] / cal_distance(v[p1_idx],v[p1_idx+1], v[p2_idx],v[p2_idx+1])
        move_y = dy * coef * r[p1_idx//2] / cal_distance(v[p1_idx],v[p1_idx+1], v[p2_idx],v[p2_idx+1])

        # 좌표 이동
        v[p1_idx] += move_x
        v[p1_idx + 1] += move_y
        v[p2_idx] -= move_x
        v[p2_idx + 1] -= move_y
        return v

    # 각 변을 순서대로 축소
    v = vertices.copy()
    for i in range(4):
        v = move_points(v, i + offset, (i + 1) % 4 + offset, r, coef)

    return v


# theta 값이 양수라면, 시계 방향으로 회전된 것을 의미함.
def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


''' [Ln 145 ~ ]
- "Process" : vertices(8,) -> v(4, 2) -> 회전 -> v(4, 2) -> vertices(8,)
- theta : 회전할 각도 (radian)
- anchor : 회전의 중심점. None=첫 번째 곡짓점
'''
def rotate_vertices(vertices, theta, anchor=None):
    """
    텍스트 영역의 꼭짓점들을 주어진 각도로 회전시킵니다.

    Args:
        anchor (numpy.ndarray, optional): 회전의 중심점.
            None인 경우 첫 번째 꼭짓점을 기준으로 함

    Returns:
        numpy.ndarray: 회전된 꼭짓점들의 좌표 (8,)

    Raises:
        ValueError: vertices가 올바른 형태가 아닌 경우
    """
    # 입력 검증
    if vertices.size != 8:
        raise ValueError("vertices must contain exactly 8 coordinates")


    v = vertices.reshape((4, 2)).T # vertice(8,) -> v(2, 4)

    if anchor is None:
        anchor = v[:, :1]  # shape: (2, 1)

    rotate_mat = get_rotate_mat(theta)  # rotation matrix(2, 2)


    res = np.dot(rotate_mat, v - anchor) + anchor

    # 원래 형태로 변환하여 반환 (8,) 형태의 1차원 배열
    return res.T.reshape(-1)


# vertices를 입력받았을 때, BBox의 x/y 좌표를 돌려줌
def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


# err : Text BBox의 "이상적인 값 - 실제 관측값"
def cal_error(vertices):
    x_min, x_max, y_min, y_max = get_boundary(vertices) # x, y의 꼭짓점
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


''' [Ln 209 ~ 262]
- 텍스트 영역을 감싸는 최소 면적의 직사각형을 만드는 최적의 회전 각도를 찾습니다.
Process:
    1. -90도에서 90도까지 각도를 바꿔가며 회전된 영역의 경계 상자 면적 계산
    2. 면적이 작은 상위 n개의 각도에 대해 방향 오차 검사
    3. 방향 오차가 가장 작은 각도 선택
'''
def find_min_rect_angle(vertices: np.ndarray,
                       angle_interval: int = 1,
                       rank_num: int = 10) -> float:
    '''
    Args:
        - angle_interval (int): 각도 탐색 간격 (기본값: 1도)
        - rank_num (int): 검토할 상위 후보 각도 수 (기본값: 10)

    Returns: 최적의 회전 각도 (radian)
    '''
    # 입력 검증
    if vertices.size != 8:
        raise ValueError("vertices must contain exactly 8 coordinates")
    if angle_interval <= 0:
        raise ValueError("angle_interval must be positive")
    if rank_num <= 0:
        raise ValueError("rank_num must be positive")

    def calculate_rotated_area(theta_deg: float) -> float:
        """주어진 각도로 회전했을 때의 경계 상자 면적 계산"""
        theta_rad = np.deg2rad(theta_deg)
        rotated = rotate_vertices(vertices, theta_rad)

        # x, y 좌표 분리
        x_coords = rotated[::2]
        y_coords = rotated[1::2]

        # 경계 상자 크기 계산
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        return width * height

    # 각도 목록 생성 (-90도 ~ 90도) -> 목록별 면적 계산
    angle_list = np.arange(-90, 90, angle_interval)
    area_list = [calculate_rotated_area(theta) for theta in angle_list]

    # 면적 기준 정렬된 인덱스 (작은 순)
    sorted_indices = np.argsort(area_list)

    # 상위 후보들에 대해 방향 오차 검사
    min_error = float('inf')
    best_angle = 0

    for idx in sorted_indices[:rank_num]:
        theta_rad = np.deg2rad(angle_list[idx])
        rotated = rotate_vertices(vertices, theta_rad)
        temp_error = cal_error(rotated)

        if temp_error < min_error:
            min_error = temp_error
            best_angle = theta_rad
    return best_angle


# Crop area가 text area와 부분적으로 겹치는지 확인
def is_cross_text(start_loc,
                 length: int,
                 vertices: np.ndarray,
                 intersection_threshold = (0.01, 0.99)) -> bool:
    """
    Args:
        start_loc (Tuple[int, int]): 자르기 영역의 좌상단 좌표 (w, h)
        length (int): 자르기 영역의 한 변의 길이 (정사각형 기준)
        vertices (np.ndarray): [x1,y1,x2,y2,x3,y3,x4,y4]
        intersection_threshold (Tuple[float, float]):
            교차 판정을 위한 (최소, 최대) 면적 비율 (기본값: (0.01, 0.99))

    Returns: Area(text), Area(crop)이 겹치는 경우를 근거로 판단함.
        - True : '일부만' 겹치는 경우 (0.01 ~ 0.99)
        - False : 전혀 겹치지 않거나 완전히 포함되는 경우 (~ 0.01, 0.99 ~)
    """
    # 입력 검증
    if length <= 0:
        raise ValueError("length must be positive")
    if vertices.size == 0:
        return False
    if not (0 <= intersection_threshold[0] < intersection_threshold[1] <= 1):
        raise ValueError("Invalid intersection threshold range")

    def create_square_polygon(start_w: int, start_h: int, size: int) -> Polygon:
        """주어진 시작점과 크기로 정사각형 폴리곤 생성"""
        corners = np.array([
            [start_w, start_h],                    # 좌상
            [start_w + size, start_h],             # 우상
            [start_w + size, start_h + size],      # 우하
            [start_w, start_h + size]              # 좌하
        ])
        return Polygon(corners).convex_hull

    # 자르기 영역의 정사각형 폴리곤 생성
    start_w, start_h = start_loc
    crop_polygon = create_square_polygon(start_w, start_h, length)

    min_ratio, max_ratio = intersection_threshold

    # 각 텍스트 영역에 대해 교차 검사
    for vertex in vertices:
        # 텍스트 영역의 폴리곤 생성
        text_polygon = Polygon(vertex.reshape((4, 2))).convex_hull

        try:
            # 교차 영역의 비율 계산
            intersection_area = crop_polygon.intersection(text_polygon).area
            intersection_ratio = intersection_area / text_polygon.area

            # 부분적 겹침 확인 (너무 작거나 너무 큰 겹침은 제외)
            if min_ratio <= intersection_ratio <= max_ratio:
                return True

        except Exception as e:
            # 기하학적 연산 오류 처리
            print(f"Warning: Geometric operation failed: {e}")
            continue

    return False


''' [Ln 332 ~ 365]
- Crop image가 Text area를 적당히 포함하는지 확인하고, 만약 그렇지 않다면 vertex를 적당히 조절함.
- Image resizing, random crop 사용 : Text Intersection 방지 목적
'''
def crop_img(img, vertices, labels, length):
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w, ratio_h = img.width / w, img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h, remain_w = img.height - length, img.width - length
    flag, cnt = True, 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices

    pass  # Actual implementation as shown in original code


# rotation matrix를 입력받고, 모든 pixel에 rotation 적용시킴.
def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x, y = np.arange(length), np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin, y_lin = x.reshape((1, x.size)), y.reshape((1, x.size))

    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])

    rotated_x, rotated_y = rotated_coord[0, :].reshape(x.shape), rotated_coord[1, :].reshape(y.shape)

    return rotated_x, rotated_y


# w, h 중에서 큰 값을 기준으로 정사각형 이미지를 만들고, 나머지 부분은 padding 처리
def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to augment data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    # vertices에 ratio_h 적용
    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices


# 이미지 내의 텍스트 영역을 잘라내고, 잘라낸 이미지를 정사각형으로 만들기 위해 padding 처리
def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x, center_y = (img.width - 1) / 2, (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)  # -10 ~ +10
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)

    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))

    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0) # 다각형(square)를 그려주는 명령어
    return mask


''' [Ln 461 ~ 474]
- If) 'drop_under'='ignore_under'=0 : filter 실행 X
- Else) vertices filtering
'''
def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels


# Add noise using pepper method
def add_pepper(image, p):
    if np.random.random() < p:
        image = np.array(image)
        noise_img = random_noise(image, mode='pepper', amount=0.08)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        return Image.fromarray(noise_img)
    return image


def random_choice_augmentations(propability):
    random_choice = A.OneOf([
        A.RandomShadow(num_shadows_lower=2, num_shadows_upper=5, always_apply=True),
        A.RandomGravel(number_of_patches=20, always_apply=True),
        A.RandomRain(blur_value=1, always_apply=True),
        A.RandomBrightnessContrast(always_apply=True)
    ], p=propability)
    return random_choice

class SceneTextDataset(Dataset):
    def __init__(self, root_dir,
                 split='train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_tags=[],
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 augmentation=False,
                 binarization=False,
                 color_jitter=False,
                 normalize=False):

        # Load dataset : ufo 폴대 내부에 여러 개의 json file을 불러옴
        with open(os.path.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = os.path.join(root_dir, 'img/train')

        # Set hyperparameter
        self.image_size = image_size
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.binarization = binarization
        self.color_jitter = color_jitter
        self.normalize = normalize

        self.ignore_tags = ignore_tags

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        # 1. Load Image and annotation(json)
        image_fname = self.image_fnames[idx]
        image_fpath = os.path.join(self.image_dir, image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            word_tags = word_info['tags']

            # 2. Text area filtering
            ignore_sample = any(elem for elem in word_tags if elem in self.ignore_tags)
            num_pts = np.array(word_info['points']).shape[0]

            if ignore_sample or num_pts > 4:
                continue

            vertices.append(np.array(word_info['points']).flatten())
            labels.append(int(not word_info['illegibility']))
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        # 3. Text area filtering
        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        # 4. 이미지 전처리
        image = cv2.imread(image_fpath)
        image = Image.fromarray(image[:, :, ::-1])
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        # exception
        if self.augmentation and any([self.binarization, self.color_jitter, self.normalize]):
            warnings.warn("Only one of augmentation and others should be declared.")
            raise ValueError

        # 5. (optional) If augmentation is True -> add_pepper
        funcs = []
        if self.augmentation:
            funcs.append(random_choice_augmentations(1.0))  # random choice augmentations
        else:
            image = add_pepper(image, 0.5)  # add salt & pepper

        if self.binarization:
            _, image = cv2.threshold(image, cv2.THRESH_OTSU)  # binarization
        if self.color_jitter:
            funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.89, 0.88, 0.88), std=(0.16, 0.17, 0.17)))

        transform = A.Compose(funcs)
        image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask