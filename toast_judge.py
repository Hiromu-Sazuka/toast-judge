"""
toast_judge.py
食パンの焼き加減判定スクリプト
CLIP + HSV 色分析による 5 段階評価
Google Colaboratory 上で動作します。
"""

# ============================================================
# 1. 依存パッケージのインストール（Colab 上で実行）
# ============================================================
# !pip install ftfy regex tqdm
# !pip install git+https://github.com/openai/CLIP.git

# ============================================================
# 2. ライブラリのインポート
# ============================================================
import torch
import clip
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ============================================================
# 3. デバイス設定
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================================
# 4. Google Drive のマウント（Colab 上で実行）
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# 5. CLIP モデルの読み込み
# ============================================================
model, preprocess = clip.load("ViT-B/32", device=device)

# ============================================================
# 6. テキストプロンプトの定義（5 段階）
# ============================================================
prompt_groups = [
    [
        "white bread with almost no browning",
        "bread that is barely toasted and still white"
    ],
    [
        "lightly toasted bread with pale golden color",
        "bread with light golden surface"
    ],
    [
        "evenly toasted bread with golden brown surface",
        "uniform golden brown toast"
    ],
    [
        "toast with dark brown areas and uneven browning",
        "heavily toasted bread with dark patches"
    ],
    [
        "burnt toast with black charred surface",
        "toast that is overcooked and blackened"
    ]
]

scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
labels = ["very_light", "light", "perfect", "heavy", "burnt"]

prompts = [p for group in prompt_groups for p in group]
text_tokens = clip.tokenize(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ============================================================
# 7. 色分析関数の定義
# ============================================================

def get_brown_distribution(image_path):
    """茶色ピクセルの割合と分布カバレッジを返す"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 赤みがかった焦げ茶もカバー（H: 5〜28）
    brown_mask = (H >= 5) & (H <= 28) & (S >= 40) & (V > 50)
    brown_ratio = brown_mask.sum() / brown_mask.size

    # 4×4 ブロックで分布を評価
    h, w = brown_mask.shape
    block_h, block_w = h // 4, w // 4
    filled_blocks = 0
    for i in range(4):
        for j in range(4):
            block = brown_mask[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            if block.mean() > 0.1:
                filled_blocks += 1
    coverage = filled_blocks / 16

    return brown_ratio, coverage


def get_brown_intensity(image_path):
    """茶色ピクセルの平均彩度と明度を返す"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    brown_mask = (H >= 5) & (H <= 28) & (S >= 40) & (V > 50)
    if brown_mask.sum() == 0:
        return 0.0, 0.0

    mean_S = S[brown_mask].mean()
    mean_V = V[brown_mask].mean()
    return mean_S, mean_V


def get_dark_pixel_ratio(image_path):
    """黒に近いピクセルの割合を返す（焦げ検出用）"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_mask = gray < 40
    return dark_mask.sum() / dark_mask.size

# ============================================================
# 8. 焼き加減の判定関数
# ============================================================

def predict_doneness(image_path):
    """
    画像のパスを受け取り、焼き加減スコアと確率分布を返す。

    Returns:
        score (float): 0.0〜1.0 の焼き加減スコア
        probs (np.ndarray): 各ラベルの確率（長さ 5）
    """
    # --- CLIP スコアの計算 ---
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze()
    similarity = similarity / 0.07
    probs_all = similarity.softmax(dim=0).cpu().numpy()

    probs = []
    idx = 0
    for group in prompt_groups:
        size = len(group)
        probs.append(probs_all[idx:idx + size].mean())
        idx += size
    probs = np.array(probs)
    probs = probs / probs.sum()

    clip_score = (probs * scores).sum()

    # --- 色分析 ---
    brown_ratio, coverage = get_brown_distribution(image_path)
    dark_ratio = get_dark_pixel_ratio(image_path)
    mean_S, mean_V = get_brown_intensity(image_path)

    brown_ratio, coverage = get_brown_distribution(image_path)
    dark_ratio = get_dark_pixel_ratio(image_path)
    mean_S, mean_V = get_brown_intensity(image_path)
    
    print(
        f"brown_ratio={brown_ratio:.3f}, "
        f"coverage={coverage:.3f}, "
        f"dark_ratio={dark_ratio:.3f}, "
        f"mean_S={mean_S:.1f}, "
        f"clip_score={clip_score:.3f}"
    )

    # --- 補正ルールの適用 ---

    # 黒ピクセルが多い → burnt
    if dark_ratio > 0.20:
        probs = np.array([0.0, 0.0, 0.0, 0.2, 0.8])
        return 0.95, probs

    # 茶色ピクセルがほぼない → very_light
    if brown_ratio < 0.10:
        probs = np.array([0.8, 0.2, 0.0, 0.0, 0.0])
        return (probs * scores).sum(), probs

    # 彩度が低く茶色も薄い → light
    if mean_S < 145 and brown_ratio < 0.6:
        probs = np.array([0.05, 0.70, 0.20, 0.05, 0.0])
        return (probs * scores).sum(), probs

    return clip_score, probs



# ============================================================
# 9. 単一フォルダ内の画像を一括処理して表示
# ============================================================

def predict_folder(folder_path):
    """フォルダ内の全画像に対して判定を実行し、結果を表示する"""
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, fname)
            score, probs = predict_doneness(image_path)
            pred_label = labels[np.argmax(probs)]

            print(f"\n=== {fname} ===")
            print(f"Score: {score:.3f} [{pred_label}]")

            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{fname}  Score: {score:.3f} [{pred_label}]")
            plt.show()

# ============================================================
# 10. バッチ処理 → CSV 出力
# ============================================================

def batch_predict_to_csv(folder_path, output_csv="toast_results.csv"):
    """フォルダ内の全画像を判定して結果を CSV に保存する"""
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, fname)
            score, probs = predict_doneness(path)
            pred_label = labels[np.argmax(probs)]
            results.append({
                "filename": fname,
                "score": float(score),
                "pred_label": pred_label,
                "very_light": float(probs[0]),
                "light": float(probs[1]),
                "perfect": float(probs[2]),
                "heavy": float(probs[3]),
                "burnt": float(probs[4])
            })

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved to {output_csv}")
    return results


