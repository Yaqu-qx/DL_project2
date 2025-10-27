import json
import numpy as np
import os

def convert_to_superglue_format(input_json_path, output_file):
    """
    将位姿数据转换为SuperGlue要求的格式
    
    格式:
    path_image_A path_image_B exif_rotationA exif_rotationB [KA_0 ... KA_8] [KB_0 ... KB_8] [T_AB_0 ... T_AB_15]
    """
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # 构建内参矩阵 K
    K = np.array([
        [data['fl_x'], 0, data['cx']],
        [0, data['fl_y'], data['cy']],
        [0, 0, 1]
    ])
    K_flat = K.flatten().tolist()
    
    # 处理图像对
    frames = data['frames']
    pairs_data = []
    
    # 创建连续帧对 (相邻帧匹配)
    for i in range(0, len(frames) - 1, 10):
        frame_A = frames[i]
        frame_B = frames[i + 1]
        
        # 图像路径
        path_A = frame_A['file_path']
        path_B = frame_B['file_path']
        
        # EXIF旋转 (默认为0，无旋转)
        exif_rotationA = 0
        exif_rotationB = 0
        
        # 计算从A到B的变换矩阵 T_AB
        T_A = np.array(frame_A['transform_matrix'])
        T_B = np.array(frame_B['transform_matrix'])
        
        # T_AB = T_B * inv(T_A)
        T_A_inv = np.linalg.inv(T_A)
        T_AB = np.dot(T_B, T_A_inv)
        T_AB_flat = T_AB.flatten().tolist()
        
        # 构建行数据
        line_data = [
            path_A, path_B,
            exif_rotationA, exif_rotationB,
            *K_flat,  # KA
            *K_flat,  # KB (假设相同相机)
            *T_AB_flat  # T_AB
        ]

        pairs_data.append(line_data)
    
    # 写入文件
    with open(output_file, 'w') as f:
        for line_data in pairs_data:
            # 将所有元素转换为字符串并连接
            line = ' '.join(map(str, line_data))
            f.write(line + '\n')
    
    print(f"SuperGlue格式数据已保存到: {output_file}")
    print(f"生成了 {len(pairs_data)} 个图像对")
    return output_file

# 使用示例
if __name__ == "__main__":
    input_json = "./dataset-advance/bdaibdai___MatrixCity/small_city/aerial/pose/block_A/transforms_test.json"  # 替换为你的输入文件路径
    output_file = "./dataset-advance/bdaibdai___MatrixCity/small_city/aerial/pose/block_A/superglue2.txt"
    convert_to_superglue_format(input_json, output_file)