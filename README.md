# 🍞 Toast Doneness Classifier

**CLIP と色分析を組み合わせた食パンの焼き加減自動判定システム**

ポップアップトースターを持っていなくても、パンの焼き色を AI が自動で判定してくれるスクリプトです。  
Google Colaboratory 上で動作します。

---

## 🎯 できること

画像を渡すだけで、焼き加減を 5 段階で判定します。

| スコア | ラベル | 状態 |
|--------|--------|------|
| 0.00 | `very_light` | ほぼ焼けていない |
| 0.25 | `light` | 薄いきつね色 |
| 0.50 | `perfect` | ちょうどよい焼き色 |
| 0.75 | `heavy` | やや焼きすぎ |
| 1.00 | `burnt` | 焦げている |

---

## 🔧 使用技術

- **[CLIP](https://github.com/openai/CLIP)** （OpenAI）― 画像とテキストの照合モデル
- **OpenCV** ― HSV 色空間による茶色ピクセル分析
- **Google Colaboratory** ― 実行環境（GPU 対応）

---

## 📂 ファイル構成

```
toast-judge/
├── README.md          # このファイル
├── toast_judge.py     # メインスクリプト
├── requirements.txt   # 依存パッケージ
└── .gitignore
```

---

## 🚀 使い方

### 1. Google Colab で開く

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. 依存パッケージのインストール

```python
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
```

### 3. Google Drive をマウント

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. 画像フォルダのパスを設定

```python
folder_path = "/content/drive/MyDrive/あなたのフォルダ名/bread"
```

### 5. 実行

```python
batch_predict_to_csv(folder_path, "toast_results.csv")
```

結果は CSV ファイルにも保存されます。

---

## 📊 判定ロジック（概要）

```
入力画像
  │
  ├─ CLIP スコア計算
  │     テキストプロンプト（5段階）との類似度を計算
  │
  ├─ 色分析（HSV 空間）
  │     茶色ピクセルの割合・分布・彩度・明度を分析
  │
  └─ 補正ルール適用
        ・黒ピクセルが多い → burnt に補正
        ・茶色ピクセルが少ない → very_light に補正
        └ 最終スコアを出力
```

詳細なフローチャートは下の図を参照してください。

---

## 🧪 判定結果のサンプル

| filename | score | pred_label |
|----------|-------|------------|
| toast1.jpg | 0.313 | light |
| toast4.jpg | 0.515 | perfect |
| toast10.jpg | 0.586 | heavy |

---

## ⚠️ 既知の課題と今後の展望

- CLIP はテキストプロンプトの表現に強く依存するため、赤みがかった焦げ茶（toast10 のような画像）の判定が難しい
- CNN ベースの学習済みモデルによる精度向上を検討中
- リアルタイム動画判定への対応も視野に入れている

---

## 📝 背景

授業の課題として、ChatGPT とのバイブコーディングで作成。  
「パンを焼くたびに取り出して確認する手間をなくしたい」という個人的な動機がきっかけです。

---

## 📄 ライセンス

MIT License
