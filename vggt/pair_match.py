import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import random

def extract_vggt_features(model, images, device):
    """
    使用VGGT提取图像特征
    """
    with torch.no_grad():
        images_batch = images.unsqueeze(0)
        aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
        point_map, point_conf = model.point_head(aggregated_tokens_list, images_batch, ps_idx)
        return point_map, point_conf

def match_features(desc1, desc2, max_matches=500):
    """
    使用BFMatcher进行特征匹配
    """
    print("?")
    desc1_np = desc1.cpu().numpy().reshape(-1, desc1.shape[-1]).astype(np.float32)
    desc2_np = desc2.cpu().numpy().reshape(-1, desc2.shape[-1]).astype(np.float32)
    print("??")
    # L2归一化
    desc1_np /= np.linalg.norm(desc1_np, axis=1, keepdims=True) + 1e-6
    desc2_np /= np.linalg.norm(desc2_np, axis=1, keepdims=True) + 1e-6
    print(desc1_np , desc2_np )

    if desc1_np.shape[0] == 0 or desc2_np.shape[0] == 0:
        print("⚠️ 转换后特征为空，跳过匹配")
        return [], desc1_np, desc2_np

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    print("????")
    matches = bf.match(desc1_np, desc2_np)
    matches = sorted(matches, key=lambda x: x.distance)

    num_to_keep = min(max_matches, len(matches))
    good_matches = matches[:num_to_keep]

    return good_matches, desc1_np, desc2_np

def generate_keypoints(desc, H, W):
    """
    将特征描述符映射为图像关键点坐标
    """
    keypoints = []
    for idx in range(H * W):
        y = idx // W
        x = idx % W
        keypoints.append(cv2.KeyPoint(float(x), float(y), 2.0))  # size=2.0，线会更粗
    return keypoints

def visualize_matches(image1_path, image2_path, matches, keypoints1, keypoints2, output_path="matches_visualization.png", max_lines=30):
    """
    可视化匹配点
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # 随机均匀采样匹配线
    if len(matches) > max_lines:
        matches_to_draw = random.sample(matches, max_lines)
    else:
        matches_to_draw = matches

    match_img = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2,
        matches_to_draw, None,
        matchColor=(0,255,0),
        singlePointColor=(255,0,0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"VGGT Feature Matches ({len(matches_to_draw)} matches)")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 匹配可视化已保存: {output_path}")

def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"使用设备: {device}")

    # 初始化模型
    model = VGGT()
    local_model_path = "./model.pt"
    if os.path.exists(local_model_path):
        state_dict = torch.load(local_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        print("无法加载模型，请确保模型文件存在")
        return
    model.eval()

    # 图像路径
    image_paths = [
        "../dataset-advance/bdaibdai___MatrixCity/aerial_street_fusion/aerial/0060.png",
        "../dataset-advance/bdaibdai___MatrixCity/aerial_street_fusion/street/test/0042.png",
    ]

    # 加载和预处理图像
    images = load_and_preprocess_images(image_paths).to(device)

    # 提取特征
    print("🔍 正在提取VGGT特征...")
    point_map, point_conf = extract_vggt_features(model, images, device)

    # 假设输出特征图为 [B, num_images, H, W, C]
    H, W = point_map.shape[2], point_map.shape[3]
    desc1 = point_map[0, 0]  # 第一张图
    desc2 = point_map[0, 1]  # 第二张图

    print(f"✅ 每张图使用 {H*W} 个特征点进行匹配")
    matches, desc1_np, desc2_np = match_features(desc1, desc2, max_matches=500)
    print(matches, desc1_np, desc2_np)
    
    if len(matches) < 20:
        print(f"⚠️ 匹配数太少: {len(matches)}")
        return

    keypoints1 = generate_keypoints(desc1, H, W)
    keypoints2 = generate_keypoints(desc2, H, W)

    print("🔗 正在进行特征匹配...")
    visualize_matches(image_paths[0], image_paths[1], matches, keypoints1, keypoints2, output_path="../advance-output/vggt-3/60-42.png", max_lines=30)

    # 打印匹配距离统计
    distances = [m.distance for m in matches]
    print(f"📊 匹配距离统计: 最小={min(distances):.4f}, 最大={max(distances):.4f}, 平均={np.mean(distances):.4f}")
    print(f"✅ 成功提取并匹配 {len(matches)} 对特征点")

if __name__ == "__main__":
    main()
