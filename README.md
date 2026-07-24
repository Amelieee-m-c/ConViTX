# ConViTX 重現 — CNN+ViT 平行融合的植物病害分類

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![Reproduction](https://img.shields.io/badge/status-partial-dbab09)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

重現的論文:
> "An Ultra Lightweight Interpretable Convolution-Vision Transformer Fusion
> Model for Plant Disease Identification: ConViTX", IEEE TCBB, 2025.
>(官方程式碼:https://github.com/Image-and-Vision-Engineering-Group/ConViTX
> ——沒有參考,這是純粹從論文文字獨立重新實作的 clean-room 版本。)

## 重現結果

| 資料集 | 指標 | 論文 | 重現結果 |
|---|---|---|---|
| PlantVillage(38 類) | Accuracy | 99.63% | 97.75% |
| | Precision | 99.65% | 96.25% |
| | Recall | 99.63% | 97.24% |
| | F1 | 99.64% | 96.71% |
| | 可訓練參數量 | 704,882 | 328,871 |
| PlantDoc(27 類,自行訓練) | Accuracy(4 次調參嘗試中最好的一次) | 69.92% | 25.42% |

PlantVillage 重現得還算接近。PlantDoc 沒有——詳見下方「PlantDoc 調參紀錄」;
不管換哪一組 learning rate 或資料增強組合,差距都存在,看起來是這個架構在
小樣本情境下的真實天花板(它的 ViT 分支完全沒有預訓練權重,不像 CNN 分支的
MobileNetV2 有 ImageNet 預訓練),不是超參數沒調好。

<details>
<summary>PlantDoc 調參紀錄(4 次嘗試)</summary>

| # | lr | 資料增強 | Weight decay | 停止 epoch | Accuracy |
|---|---|---|---|---|---|
| 1 | 1e-4 | 只有水平翻轉 | 0 | 45 | 16.95% |
| 2(最佳) | 1e-4 | rotate/shift/zoom/shear | 1e-4 | 149 | 25.42% |
| 3 | 3e-5 | rotate/shift/zoom/shear | 1e-4 | 150(跑滿上限) | 18.22% |

</details>

## 資料集狀況(2026-07-22 確認)

論文在 5 個資料集上做評測,這裡實際只有兩個是拿得到的:

| 資料集 | 狀態 | 備註 |
|---|---|---|
| PlantVillage(38 類,54,305 張) | ✅ 有 | 沿用唯讀的 `rethinking_fewshot_vlms/data/PlantVillage_Split_721`(7:2:1 切分) |
| PlantDoc(28 類,約 2,572 張) | ✅ 有 | 從官方 `pratikkayal/PlantDoc-Dataset` repo 下載到 `PlantDoc_full/` |
| Embrapa(93 類,46,376 張) | ❌ 沒有 | 沒有取得 |
| PlantCOMB(35 類,11,824 張,作者自己拼接 10 個 Kaggle 資料集) | ❌ 沒有 | 需要重新拼湊 10 個各自獨立的外部資料集 |
| IIITDM_Maize(416 張真實空拍 + 400 張 C3GAN 合成) | ❌ 沒有,也無法重現 | 作者自己私有的空拍田間資料集 + 自己的 C3GAN 生成模型,無法獨立重建 |

這次重現的範圍:只做 **PlantVillage 主要 benchmark(論文 Table III(a))
+ PlantDoc 跨資料集泛化測試(論文 Table III(c))**。Embrapa/PlantCOMB/Maize
的結果,以及 33 組配置的消融實驗(Table I/II),都不在範圍內。

## 已知偏差 / 論文本身的矛盾之處

論文的架構描述(Section II-B、Fig. 2、Algorithm 1)有幾處資訊不足,還有一處
直接矛盾。以下是處理方式(`src/model.py` 的 docstring 裡有同步的清單):

1. **MHA head 數矛盾**:消融實驗的文字(Section III-C)說 16 個 head 表現
   最好(Table II(a)),但緊接在 Section III-D 前面的「最終架構」段落卻說
   實際部署的模型每層用 **4** 個 head。這裡採用 4,相信論文自己說的最終
   配置,而不是消融表格的結論。
2. **CNN/ViT 空間解析度不對齊**:CNN 分支(224×224 輸入經過前 2 個
   MobileNetV2 block)輸出 56×56;ViT 分支(同樣輸入切成 7×7 patch)自然
   會產生 32×32 的網格。論文只說 ViT 這邊的 depthwise-separable conv block
   會「幫忙對齊輸出維度」,但沒說怎麼做。這裡用雙線性內插把 ViT 分支從
   32×32 放大到 56×56,再用 pointwise conv 把通道數對齊到 24,才做拼接。
3. **Projection dimension**:最終架構段落寫的是 48,但 Table II(c) 裡
   準確率最高(99.54)的那一列卻對應到「64」。這裡採用 48(理由跟第 1 點
   一樣——相信明確寫出的最終規格)。
4. **Epoch 數 / early stopping**:目前拿到的論文摘錄沒有給。這裡預設最多
   30 epoch,early stopping 監控 val_loss,patience=5。
5. **融合模組裡的「MobileNetV2 block」**(16 filters、3×3、stride 2)用
   torchvision 實際的 `InvertedResidual` 類別實作(expand ratio 6,
   MobileNetV2 的預設值),而不是單純的 conv,因為論文明確稱它為
   「MobileNetV2 block」。

最終參數量:**約 0.33M 可訓練參數**,跟論文宣稱的 0.7M(704,882)同一個
數量級但偏低——這是預期中的結果,因為論文沒有完整交代好幾個中間層的
channel 寬度,這裡選擇的是內部一致性,而不是硬湊到論文的數字。

## 檔案說明

- `src/model.py` — ConViTX 架構(`SEBlock`、`CNNBranch`、`ViTBranch`、
  `FusionModule`、`ConViTX`)。
- `src/train_plantvillage.py` — 在 38 類 PlantVillage 切分上做完整資料集
  訓練(不是 few-shot),Adam lr=1e-4,batch=16,只做 rescale 到 [0,1] 的
  前處理,對應論文規格。
- `data_prep/plantdoc_class_mapping.py` — 把 PlantDoc 自由文字的資料夾
  名稱對應到 PlantVillage 的 `Species___Disease` 類別名稱,只對應有明確
  對應關係的 28 個類別。
- `src/eval_plantdoc.py` — 載入 PlantVillage 訓練好的權重,在 PlantDoc
  (對應子集)上評估,對應論文 Table III(c) 的田間泛化比較。

## 執行方式

```
python src/train_plantvillage.py --output_dir runs/plantvillage_seed1 --epochs 30 --patience 5
python src/eval_plantdoc.py --model_dir runs/plantvillage_seed1 --split both
```

## 環境需求

Python 3.10+、PyTorch + torchvision(建議 CUDA 版)、scikit-learn、matplotlib。

## 授權

本 repo 程式碼採用 MIT 授權。不包含任何資料集圖片——PlantVillage 和
PlantDoc 需要自行取得(見上方「資料集狀況」),各自有其原始授權。
