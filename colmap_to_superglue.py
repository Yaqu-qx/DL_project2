#!/usr/bin/env python3
"""
COLMAP to SuperGlue格式转换脚本
从COLMAP的cameras.txt和images.txt生成SuperGlue所需的图像对文件
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# def normalize_intrinsics_for_scannet(K, original_width=1707, original_height=1280):
#     """将内参归一化到ScanNet标准尺寸 (1296x968)"""
#     target_width = 640
#     target_height = 480
#
#     scale_x = target_width / original_width
#     scale_y = target_height / original_height
#
#     # 创建缩放矩阵
#     scale_matrix = np.array([
#         [scale_x, 0, 0],
#         [0, scale_y, 0],
#         [0, 0, 1]
#     ])
#
#     # 应用缩放
#     K_normalized = K @ scale_matrix
#
#     print(f"Original K: fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}, cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}")
#     print(
#         f"Normalized K: fx={K_normalized[0, 0]:.2f}, fy={K_normalized[1, 1]:.2f}, cx={K_normalized[0, 2]:.2f}, cy={K_normalized[1, 2]:.2f}")
#
#     return K_normalized

def parse_cameras(cameras_file):
    """解析cameras.txt文件"""
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            # 构建内参矩阵
            if model == 'SIMPLE_RADIAL':
                # SIMPLE_RADIAL: f, cx, cy, k
                f, cx, cy, k = params
                K = np.array([
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]
                ])
            elif model == 'PINHOLE':
                # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = params
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            else:
                print(f"Warning: Unsupported camera model {model}, using default")
                K = np.array([
                    [1000, 0, width / 2],
                    [0, 1000, height / 2],
                    [0, 0, 1]
                ])

            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'K': K,
                'params': params
            }

    return cameras


def parse_images(images_file):
    """解析images.txt文件"""
    images = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue

            parts = line.split()
            if len(parts) < 10:
                i += 1
                continue

            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])  # 四元数 (qw, qx, qy, qz)
            tx, ty, tz = map(float, parts[5:8])  # 平移向量
            camera_id = int(parts[8])
            name = parts[9]

            # 构建从世界到相机的变换矩阵
            # 四元数转旋转矩阵
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # 构建变换矩阵 [R | t]
            T = np.eye(4)
            T[:3, :3] = rotation
            T[:3, 3] = [tx, ty, tz]

            images[image_id] = {
                'name': name,
                'camera_id': camera_id,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'T_world_to_cam': T  # 世界坐标系到相机坐标系的变换
            }

            i += 2  # 跳过下一行（特征点数据）

    return images


def compute_relative_pose(T1, T2):
    """计算从相机1到相机2的相对变换"""
    # T1: 世界到相机1的变换
    # T2: 世界到相机2的变换
    # 相对变换: T1_to_2 = T2 @ inv(T1)
    T1_inv = np.linalg.inv(T1)
    T1_to_2 = T2 @ T1_inv

    return T1_to_2


def validate_relative_pose(T_AB, img1_name, img2_name):
    """验证相对位姿是否合理"""
    rotation = T_AB[:3, :3]
    translation = T_AB[:3, 3]

    # 检查旋转矩阵的性质
    det = np.linalg.det(rotation)
    is_orthogonal = np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)

    print(f"Pair {img1_name} -> {img2_name}:")
    print(f"  Rotation det: {det:.6f} (should be ~1.0)")
    print(f"  Is orthogonal: {is_orthogonal}")
    print(f"  Translation norm: {np.linalg.norm(translation):.3f}")

    return is_orthogonal and abs(det - 1.0) < 0.01

def select_image_pairs(images, num_pairs=10):
    """选择图像对，优先选择视角相近的图像"""
    image_items = list(images.items())
    selected_pairs = []

    # 简单的策略：按图像ID顺序选择相邻的图像对
    sorted_images = sorted(image_items, key=lambda x: x[1]['name'])

    for i in range(0, len(sorted_images), 2):
        img1_id, img1_data = sorted_images[i]
        img2_id, img2_data = sorted_images[i + 1]

        selected_pairs.append((img1_data, img2_data))

    # 如果还需要更多对，选择间隔一个的图像
    if len(selected_pairs) < num_pairs:
        for i in range(min(num_pairs - len(selected_pairs), len(sorted_images) - 2)):
            img1_id, img1_data = sorted_images[i]
            img2_id, img2_data = sorted_images[i + 2]

            selected_pairs.append((img1_data, img2_data))

    return selected_pairs[:num_pairs]


def format_matrix_for_output(matrix):
    """将矩阵格式化为一行数字"""
    return ' '.join(f'{x:.6f}' for x in matrix.flatten())


def select_manual_pairs(images, pair_list):
    """根据手动指定的图像对列表选择图像对"""
    # 创建文件名到图像数据的映射
    name_to_image = {img_data['name']: img_data for img_data in images.values()}

    selected_pairs = []

    # 每两个文件名组成一对
    for i in range(0, len(pair_list), 2):
        if i + 1 >= len(pair_list):
            print(f"Warning: 忽略不成对的图像名: {pair_list[i]}")
            break

        name1 = pair_list[i] + ".jpg"
        name2 = pair_list[i + 1] + ".jpg"

        if name1 not in name_to_image:
            print(f"Warning: 图像 {name1} 不存在于COLMAP重建结果中")
            continue
        if name2 not in name_to_image:
            print(f"Warning: 图像 {name2} 不存在于COLMAP重建结果中")
            continue

        img1_data = name_to_image[name1]
        img2_data = name_to_image[name2]
        selected_pairs.append((img1_data, img2_data))

    return selected_pairs


def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP output to SuperGlue format')
    parser.add_argument('--colmap_path', type=str, required=True,
                        help='Path to COLMAP output directory (containing cameras.txt and images.txt)')
    parser.add_argument('--output_file', type=str, default='superglue_pairs.txt',
                        help='Output file name')
    parser.add_argument('--num_pairs', type=int, default=10,
                        help='Number of image pairs to generate')
    parser.add_argument('--set-pairs', nargs='+', type=str,
                        help='Manually specify image pairs. Example: --set-pairs 01 02 03 04')

    args = parser.parse_args()

    colmap_path = Path(args.colmap_path)
    cameras_file = colmap_path / 'cameras.txt'
    images_file = colmap_path / 'images.txt'

    if not cameras_file.exists() or not images_file.exists():
        print(f"Error: Could not find cameras.txt and/or images.txt in {colmap_path}")
        return

    print("Parsing COLMAP files...")
    cameras = parse_cameras(cameras_file)
    images = parse_images(images_file)

    print(f"Found {len(cameras)} cameras and {len(images)} images")

    # 选择图像对
    if args.set_pairs:
        print(f"Using manually specified pairs: {args.set_pairs}")
        pairs = select_manual_pairs(images, args.set_pairs)
    else:
        print(f"Automatically selecting {args.num_pairs} image pairs")
        pairs = select_image_pairs(images, args.num_pairs)

    print(f"Generating {len(pairs)} image pairs...")

    # 生成输出文件
    with open(args.output_file, 'w') as f:
        for img1, img2 in pairs:
            # 获取相机内参
            K1 = cameras[img1['camera_id']]['K']
            K2 = cameras[img2['camera_id']]['K']
            # K1 = normalize_intrinsics_for_scannet(cameras[img1['camera_id']]['K'], cameras[img1['camera_id']]['width'], cameras[img1['camera_id']]['height'])
            # K2 = normalize_intrinsics_for_scannet(cameras[img2['camera_id']]['K'], cameras[img2['camera_id']]['width'], cameras[img2['camera_id']]['height'])

            # 计算相对位姿
            T_relative = compute_relative_pose(img1['T_world_to_cam'], img2['T_world_to_cam'])

            # 验证相对位姿
            if not validate_relative_pose(T_relative, img1['name'], img2['name']):
                print(f"Warning: Invalid relative pose for pair {img1['name']} -> {img2['name']}")

            # 格式化输出行
            # 格式: image1 image2 0 0 K1(9) K2(9) T_relative(16)
            line_parts = [
                img1['name'], img2['name'],
                '0', '0',  # 旋转角度（通常为0）
                format_matrix_for_output(K1),
                format_matrix_for_output(K2),
                format_matrix_for_output(T_relative)
            ]

            f.write(' '.join(line_parts) + '\n')

    print(f"Successfully generated {args.output_file} with {len(pairs)} image pairs")


if __name__ == '__main__':
    main()