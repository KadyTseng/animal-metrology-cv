# 🐾 動物測量學電腦視覺系統 (Animal Metrology CV)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
---

## 專業測驗題目

本專案依據以下規格，將圖像分割（Image Segmentation）模型的輸出應用於測量學（Metrology）：

1. 使用 COCO 資料集，挑選至少包含兩隻動物的影像，動物種類不限。
2. 使用分割模型標示動物輪廓，並定位各動物的眼睛。
3. 量測每隻動物的雙眼距離。
4. 量測任意兩隻動物之間的右眼距離。
5. 說明使用的 AI 模型、選用原因及評估指標公式。
6. 說明量測方法、計算公式與驗證方式。

**實作範圍與驗證狀態：**動物輪廓由實例分割遮罩呈現；眼睛候選點由姿態關鍵點模型預測，並非眼睛區域分割。每個動物遮罩都會獨立裁切並執行 Pose，不以分割類別篩選。信心達標且落在遮罩內的點會用於遮罩範圍檢查後的距離計算；這不代表眼點解剖位置已由人工或真值驗證。原始候選點距離會另標示為未驗證估計。目前 CSV 示範使用非 COCO 影像，因此不代表已完成 COCO 多動物量測驗證。

## 📖 專案簡介 (Introduction)

**`animal-metrology-cv`** 是一套用於動物影像分析與形態距離計算的電腦視覺測量系統。

推論流程分為兩階段：**YOLOv8-seg** 取得各動物的實例遮罩與外接框；系統依遮罩裁切個體區域，將輪廓外像素填為輪廓內像素的中位色，再使用微調的 **YOLOv8-pose** 預測兩個眼睛候選點。Pose 分類結果不會拿來與分割類別比對。信心達標且位於同一個體遮罩內的兩點可計算遮罩範圍檢查後的距離；遮罩檢查不能證明點位確實是眼睛。未通過遮罩檢查的候選距離會在摘要視窗另行標示。

- **任務 3 (Task 3)**：單隻動物個體之**雙眼距離 (Interocular Distance)** 計算（像素轉實體毫米 mm）。
- **任務 4 (Task 4)**：畫面中任意兩隻動物間**右眼至右眼跨個體距離 (Cross-Individual Right-Eye Distance)** 計算。
- **數據管線與輸出**：提供即時 RESTful API 服務與結構化 CSV 批次報告輸出。
- **互動式前端**：使用者可上傳照片；滑過輪廓或量測線時，右側量測卡片會反白，文字不覆蓋影像。

輪廓模型使用 COCO 預訓練權重；眼睛關鍵點權重以貓狗資料訓練，但推論會套用到每個分割出的動物 ROI。非貓狗物種的預測可能不可靠。眼睛以關鍵點表示，不輸出眼睛區域遮罩。

---

## 🏗️ 系統架構圖 (System Architecture)

```mermaid
flowchart TD
    subgraph Input_Layer["1. 輸入層 (Input Layer)"]
        A1["原始影像 / 串流 (data/raw/)"]
        A2["REST API 上傳 (POST /predict)"]
        A3["互動式網頁上傳 (/ui/)"]
    end

    subgraph Preprocessing["2. 前處理與安全性檢驗 (Preprocessing & Security)"]
        B1["MIME 格式與大小限制檢查 (<=10MB)"]
        B2["記憶體安全解碼 (cv2.imdecode)"]
        B3["影像前處理"]
    end

    subgraph Deep_Learning["3. 兩階段推論 (Two-Stage Inference)"]
        C1["YOLOv8-seg：每隻動物的遮罩與外接框"]
        C2["依實例遮罩裁切個體並遮除輪廓外背景"]
        C3["YOLOv8-pose：逐一預測雙眼候選點，不比對分割類別"]
        C4["檢查信心與遮罩內有效性；未驗證候選分開呈現"]
    end

    subgraph Metrology_Engine["4. 幾何計量引擎 (Metrology Engine)"]
        D1["解剖特徵點解析 (左眼 / 右眼座標)"]
        D2["任務 3: 單隻動物雙眼距離 (Euclidean Distance)"]
        D3["任務 4: 跨個體右眼間距組合配對計算"]
        D4["比例尺轉換模組 (Pixel to mm: D = d_px * s)"]
    end

    subgraph Delivery_Layer["5. 輸出與交付層 (Delivery Layer)"]
        E1["FastAPI JSON 回應 (Swagger /docs)"]
        E2["批次數據輸出 (outputs/measurement_results.csv)"]
        E3["互動影像：個體輪廓色、有效眼點及未驗證候選"]
        E4["右側摘要：正式量測與未驗證候選距離分開呈現"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2 --> B3 --> C1
    C1 --> C2 --> C3 --> C4 --> D1
    D1 --> D2 & D3
    D2 & D3 --> D4
    D4 --> E1
    D4 --> E2
    D4 --> E3 --> E4
```

獨立架構圖來源：[`docs/system_architecture.mmd`](docs/system_architecture.mmd)。

每個被分割出的 COCO 動物個體都會送入 Pose 模型，不會因分割類別不是貓或狗而略過。姿態權重仍是以貓狗眼睛資料訓練，因此套用到其他物種時可能產生錯誤候選點。點位在遮罩外時會以紅色叉號顯示；遮罩內的點可參與距離計算，但「位於輪廓內」不等同於「已確認是眼睛」。右側摘要會把遮罩內估計與未通過遮罩檢查的候選分開標示。

---

## 🧰 技術棧說明 (Tech Stack)

| 領域 | 核心技術 / 套件 | 說明與用途 |
| :--- | :--- | :--- |
| **深度學習核心** | `ultralytics (YOLOv8-Seg + fine-tuned YOLOv8-Pose)` | 依各動物遮罩建立獨立 ROI；兩關鍵點姿態模型不受分割類別限制地產生候選點 |
| **影像處理** | `opencv-python`, `numpy` | 矩陣運算、幾何變換、畸變矯正、遮罩面積與視覺化標註繪製 |
| **後端服務** | `FastAPI`, `uvicorn[standard]` | REST API、靜態前端、OpenAPI / Swagger 文件 |
| **互動前端** | HTML, CSS, JavaScript Canvas | 照片上傳、輪廓/線段 hover 與右側量測卡片聯動 |
| **數據處理與輸出** | `pandas`, `pycocotools` | 形態測量數據整理、統計分析、COCO 評估格式與 CSV 匯出 |
| **容器化部屬** | `Docker` (Python 3.10-slim) | 隔離相依環境，滿足最小權限非 root 運行與資安合規 |

## 資料來源與適用範圍

- 動物實例分割使用 COCO 預訓練的 `yolov8n-seg.pt`，可分割其 COCO 類別中的多種動物。
- 在本專案檢視的 COCO 動物候選影像中，部分動物距離較遠、呈側面或雙眼不夠清楚，較適合用於輪廓分割，卻不利於雙眼關鍵點標註與訓練。因此，眼睛模型改以 Animal-Pose 資料集的 400 張貓狗影像訓練（每類 200 張）；缺失眼點標記為不可見。
- Animal-Pose 影像用於眼睛關鍵點模型的開發，不取代題目要求的 COCO 多動物測試案例。資料集可取得不代表影像或標註可自由再散布，使用前仍須確認適用授權。
- Pose 模型對所有分割出的動物 ROI 執行推論，但權重只以貓狗眼睛資料訓練；其他物種的候選點與候選距離可能不可靠。輪廓外的點不納入正式距離，並以未驗證候選形式顯示。
- 示範影像、訓練資料及模型權重不隨原始碼發佈；使用者須自行取得具合法使用權的影像及所需模型檔案。

---

## 📁 專案目錄結構 (Directory Structure)

```text
animal-metrology-cv/
├── data/
│   ├── raw/                  # 本機原始影像（不隨套件或 GitHub 發布）
│   └── animalpose_eye_yolo/  # 本機訓練資料（來源影像不隨套件發布）
├── docs/
│   └── system_architecture.mmd
├── examples/
│   └── sample_measurements.csv
├── web/
│   ├── index.html            # 互動式照片上傳與量測頁面
│   ├── app.js
│   └── styles.css
├── src/
│   ├── metrology.py          # 分割、眼點驗證與距離計算
│   ├── api.py                # FastAPI、/ui/ 與預測端點
│   ├── visualize.py          # CLI 單張/資料夾推論
│   └── fetch_coco_samples.py # COCO 範例資料下載工具
├── tests/
│   ├── test_metrology.py
│   └── test_api_security_headers.py
├── models/                   # 本機模型權重（不隨套件或 GitHub 發布）
├── .env.example              # 系統環境變數範本
├── .gitignore
├── .dockerignore
├── Dockerfile                # 輕量安全容器建置腳本
├── requirements.txt          # 專案相依 Python 套件清單
└── README.md                 # 使用、API、模型與量測說明
```

---

## 🚀 本地運行步驟 (Local Setup & Run)

### 步驟 1：建立並啟用虛擬環境

```bash
cd animal-metrology-cv

# 建立 Python 3.10 虛擬環境
python3 -m venv venv

# 啟用虛擬環境 (Linux / macOS)
source venv/bin/activate
# (Windows 請執行: venv\Scripts\activate)
```

### 步驟 2：安裝相依套件

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 步驟 3：設定環境變數

```bash
cp .env.example .env
mkdir -p models
cp /path/to/yolov8n-seg.pt models/yolov8n-seg.pt
cp /path/to/animal_eye_pose.pt models/animal_eye_pose.pt
```

原始碼發行版不包含模型權重、影像、訓練資料、推論輸出或 `.env`。執行推論前，須自行取得 YOLOv8-seg 權重及相容的雙眼關鍵點 Pose 權重，放入 `models/`，並確認其授權允許相應用途。`.env.example` 提供預設檔案路徑。

### 步驟 4：啟動互動式前端與推論

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

- 開啟互動前端：`http://127.0.0.1:8000/ui/`
- Swagger API 文件：`http://127.0.0.1:8000/docs`
- ReDoc API 文件：`http://127.0.0.1:8000/redoc`

在前端選取影像並設定比例尺後，按「開始偵測」。亦可使用 CLI 對單張影像或整個資料夾進行推論：

```bash
python src/visualize.py \
  --image path/to/your-image.jpg \
  --output-dir outputs/annotated/demo --device 0

python src/visualize.py \
  --input-dir path/to/your-image-folder \
  --output-dir outputs/annotated/seg_pose_400 --device 0
```

測試：

```bash
python -m unittest discover -s tests -v
```

---

## 🐳 Docker 部署指令 (Docker Deployment)

容器以非 root 使用者執行。模型權重需放在本機 `models/`，並以唯讀掛載提供給容器。

### 1. 建置 Docker 映像檔

```bash
docker build -t animal-metrology-cv:1.0.0 .
```

### 2. 啟動 Docker 容器

```bash
docker run -d \
  --name metrology-service \
  -p 127.0.0.1:8000:8000 \
  --env-file .env \
  -v "$PWD/models:/models:ro" \
  -e SEG_MODEL_WEIGHTS_PATH=/models/yolov8n-seg.pt \
  -e POSE_MODEL_WEIGHTS_PATH=/models/animal_eye_pose.pt \
  -e DEVICE=cpu \
  --restart unless-stopped \
  animal-metrology-cv:1.0.0
```

### 3. 查看容器日誌與狀態

```bash
docker logs -f metrology-service
curl -f http://127.0.0.1:8000/health
```

互動介面為 `http://127.0.0.1:8000/ui/`。預設使用 CPU；若要在 Docker 中使用 GPU，需另外安裝 NVIDIA Container Toolkit，並以支援 CUDA 的 PyTorch 映像與相依套件設定。

---

## 🔌 API 文件與測試資訊 (API Documentation & Testing)

啟動 API 後可使用以下頁面：

- 互動前端：`http://127.0.0.1:8000/ui/`
- Swagger / OpenAPI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`

### 測試帳號與金鑰資訊

本專案沒有使用者名稱或密碼帳號。開發測試可不帶 API key；如請求帶有 `X-API-Key`，其值必須符合執行環境中的 `API_KEY`。`.env.example` 的值只供本機示範，不可作為正式環境金鑰。此 API 目前允許未帶 key 的請求，請勿未加存取控制就公開部署。

### 端點清單

| 方法 | 端點 | 說明 |
| :--- | :--- | :--- |
| `GET` | `/ui/` | 互動式上傳、輪廓與距離 hover 檢視 |
| `GET` | `/health` | 健康檢查，確認 YOLO 模型載入狀態與當前比例尺 |
| `GET` | `/docs` | Swagger UI 互動式測試介面 |
| `POST` | `/predict` | 上傳動物圖片並以 JSON 回傳偵測結果、眼睛座標與測量數值 |
| `POST` | `/predict/visualize` | 回傳靜態標註 JPEG；互動效果請使用 `/ui/` |

若要直接在 Swagger 的 Response body 查看偵測結果，請在 `/docs` 展開 **POST `/predict`**、上傳圖片並執行。`/predict/visualize` 的回應是 JPEG 圖片；這個端點不會在 Response body 顯示 JSON 數值。

### cURL 呼叫範例

```bash
curl -X POST "http://127.0.0.1:8000/predict?pixel_to_mm_scale=0.15&save_to_csv=false" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your-image.jpg"
```
請將影像路徑替換為具合法使用權的本機圖片。

### 雙貓示範影像量測結果（非 COCO）

以下為非 COCO 雙貓示範影像的模型預測結果，使用 `0.15 mm/px` 範例比例尺：

| 測量項目 | 範例結果 |
| :--- | ---: |
| animal_1 雙眼距離 | 4.608 mm |
| animal_2 雙眼距離 | 4.519 mm |
| 兩隻動物右眼至右眼距離 | 42.411 mm |

完整 CSV 範例位於 [`examples/sample_measurements.csv`](examples/sample_measurements.csv)。數值為模型預測結果；`0.15 mm/px` 僅為示範值，未經影像中的實體標尺校正，不應視為可追溯的實際物理尺寸。

CSV 僅記錄影像識別資訊與量測值，不包含原始影像。此示範影像**不是 COCO 影像**，僅用於說明雙眼距離及跨個體右眼距離的計算，不能作為 COCO 多動物案例的驗證結果。

### COCO 資料集範圍與限制

- 專案的 YOLOv8-seg 使用 COCO 預訓練權重；本儲存庫未包含 COCO 原始影像。
- 眼睛模型是使用 Animal-Pose 貓狗資料微調的兩點 YOLOv8-Pose。推論時會對每個分割動物遮罩執行，不受分割類別限制；但非貓狗物種屬於訓練分布以外，預測只可視為未驗證候選。
- 正式量測要求兩個眼點信心達標且皆位於同一動物遮罩內。若原始候選點未通過遮罩檢查，右側摘要仍可顯示候選距離，但會明確標為「未驗證」，不納入正式測量結果。
- 篩選 COCO 影像時，部分候選動物較遠、非正面或無法同時辨識雙眼，因而不適合作為本眼睛關鍵點模型的訓練樣本；這是訓練資料選擇的考量，不是以其他資料取代 COCO 測試要求。
- 儲存庫中的 CSV 示範並非 COCO 影像推論結果，因此尚未示範符合題目條件的 COCO 多動物眼距案例。COCO 分割預訓練資料、Animal-Pose 關鍵點訓練資料及示範量測影像為不同資料來源，評估結果應分別說明。

---

## 🔬 核心申論題詳述 (Core Theoretical Analyses)

---

### 📝 第 5 題：YOLOv8-Pose / YOLOv8-Seg 模型選型理由與數學評估指標詳解

#### 1. 選擇 YOLOv8-Pose / YOLOv8-Seg 的核心原因

在動物形態計量學 (Animal Metrology) 的落地場景中，選擇基於 Ultralytics 的 YOLOv8-Pose / Seg 架構，主要基於以下四大工程與演算法優勢：

1. **分階段模型各司其職 (Segmentation + Cropped Pose)**：
   - 第一階段的 YOLOv8-seg 負責個體遮罩與外接框；第二階段依每個遮罩裁切個體 ROI，將輪廓外像素填為遮罩內像素的中位色後送入 YOLOv8-pose。Pose 類別不與分割類別比對；正式量測仍會驗證候選點是否位於個體遮罩內，未通過者只列為未驗證候選。
   - 這樣可以讓姿態模型專注於每隻動物的局部畫面；代價是每個貓狗個體都要額外執行一次姿態推論，整體速度取決於影像中的個體數量與硬體。
2. **免從頭訓練與強大的遷移學習能力 (Transfer Learning & Pre-trained Priors)**：
   - 特殊動物（如特定品種牛隻、羊隻、瀕危物種）的解剖影像標註極為昂貴且困難，難以收集數十萬張影像從零訓練 (From Scratch)。
   - 本專案以 COCO 預訓練的 `yolov8n-pose.pt` 為起點，將姿態輸出調整為左右眼兩個關鍵點，再以標註的貓狗影像微調。模型泛化能力仍須以獨立測試集及目標應用影像評估。
3. **Anchor-Free 設計與尺度適應性**：
   - 動物在攝像鏡頭前的姿態多變（站立、俯臥、側身）、體型大小跨度極大。YOLOv8 捨棄傳統複雜且僵硬的 Anchor Boxes，改採無錨點 (Anchor-Free) 迴歸中心點與關鍵點偏移，顯著減低對超參數預設的敏感度，改善關鍵點定位精度。

---

#### 2. 模型評估指標之數學公式推導

評估動物關鍵點檢測與實例分割的表現，業界標準採用 **OKS (Object Keypoint Similarity)** 與基於其建構的 **mAP (Mean Average Precision)**。

##### (A) 物件關鍵點相似度 (Object Keypoint Similarity, OKS)

OKS 衡量預測的解剖關鍵點與標註真值 (Ground Truth, GT) 之間的相似程度，其數學形式類似於目標檢測中的 IoU，值域落在 $[0, 1]$：

$$
OKS = \frac{\sum_{i} \exp\left(-\frac{d_i^2}{2 s^2 \sigma_i^2}\right) \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}
$$

**參數幾何與物理意涵解析：**
- $d_i = \sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}$：第 $i$ 個關鍵點之**預測座標 $(\hat{x}_i, \hat{y}_i)$ 與真值座標 $(x_i, y_i)$ 間的歐幾里得像素距離**。
- $s$：目標動物個體的**尺度因子 (Scale Factor)**，定義為個體分割面積或邊界框面積的平方根：$s = \sqrt{\text{Area}}$。
  - *意義*：引入 $s$ 能確保尺度不變性。對於遠處較小個體，較小的像素偏移量就會被嚴格扣分；反之對於近處較大個體，允許等比例較寬容的絕對像素偏差。
- $\sigma_i$：第 $i$ 個關鍵點的**歸一化標準差因子 (Keypoint Per-class Falloff Constant)**。
  - *意義*：不同解剖部位的標註不確定性不同。例如動物「眼睛 (Eye)」特徵清晰，人工標註變異極小，故設定較小的 $\sigma \approx 0.025$（稍有偏差即迅速衰減懲罰）；而對於「肩關節」或「臀部」，解剖定義較模糊，標註誤差自然較大，$\sigma \approx 0.089$。
- $v_i$：關鍵點真值可見度標籤（$v_i = 0$ 為未標註，$v_i = 1$ 為存在但被遮擋，$v_i = 2$ 為清晰可見）。
- $\delta(v_i > 0)$：狄拉克指示函數 (Kronecker Indicator)，僅針對影像中實際存在的關鍵點進行計算。

##### (B) 平均精確率 (Mean Average Precision, mAP)

在特定 OKS 門檻 $T$（如 $T = 0.5$ 或 $T = 0.75$）下，若某檢測結果滿足 $OKS \ge T$，則將該預測判定為 True Positive (TP)，否則為 False Positive (FP)。

1. **精確率 (Precision, $P$) 與召回率 (Recall, $R$)**：

   $$
   P(t) = \frac{TP(t)}{TP(t) + FP(t)}, \quad R(t) = \frac{TP(t)}{TP(t) + FN(t)}
   $$

2. **單一類別平均精確率 (Average Precision, AP)**：
   透過計算 Precision-Recall 曲線下的面積進行數值積分（通常採用 101 點插值法）：

   $$
   AP = \int_0^1 P_{\text{interp}}(r) \, dr
   $$

   其中 $P_{\text{interp}}(r) = \max_{\tilde{r} \ge r} P(\tilde{r})$。

3. **全類別平均精確率 (mAP)**：
   對所有目標類別集合 $C$ 取期望值：

   $$
   mAP = \frac{1}{|C|} \sum_{c \in C} AP_c
   $$

4. **綜合評估指標 $mAP@[0.5:0.95]$**：
   在 OKS 門檻由 0.50 遞增至 0.95（步長 0.05，共 10 個門檻階梯）下分別計算 mAP 並取平均值，代表模型在嚴苛幾何約束下的整體穩健度。

##### 本版本的保留測試集結果

最新貓狗 Pose 權重以 400 張資料重新切分（320 train、40 validation、40 test；每類測試 20 張）。保留測試集的 Pose `mAP50 = 0.963`、`mAP50-95 = 0.911`。測試集很小，這些指標只衡量關鍵點檢出，不代表像素級誤差或實體毫米量測的校正精度。

---

### 📏 第 6 題：歐幾里得距離、物理比例尺轉換邏輯與交叉驗證機制詳解

#### 1. 歐幾里得距離公式 (Euclidean Distance)

在 2D 數位影像平面上，任兩組解剖關鍵點座標 $P_1(x_1, y_1)$ 與 $P_2(x_2, y_2)$ 之空間幾何直線距離定義如下：

$$
d_{\text{pixel}} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

##### 任務 3：單隻動物個體之雙眼距離 (Interocular Distance)

設單隻動物個體經關鍵點檢測所得之左眼中心座標為 $P_{\mathrm{left}}(x_L, y_L)$，右眼中心座標為 $P_{\mathrm{right}}(x_R, y_R)$，則個體眼距為：

$$
d_{\text{intra-eye, px}} = \sqrt{(x_R - x_L)^2 + (y_R - y_L)^2}
$$

##### 任務 4：任意兩隻動物間右眼至右眼跨個體距離 (Inter-Animal Distance)

設個體 $A$ 的右眼中心為 $P_{A,R}(x_{A,R}, y_{A,R})$，個體 $B$ 的右眼中心為 $P_{B,R}(x_{B,R}, y_{B,R})$，兩者右眼間的跨個體距離為：

$$
d_{\text{inter-eye, px}} = \sqrt{(x_{B,R} - x_{A,R})^2 + (y_{B,R} - y_{A,R})^2}
$$

---

#### 2. 像素轉物理單位 (mm) 的比例尺轉換邏輯

影像中的距離單位為**像素 (Pixel)**，欲將其映射至實體物理世界之**毫米 (Millimeter, mm)**，其轉換架構與數學關係如下：

##### (A) 線性比例轉換模型
定義比例轉換係數 $s$（單位：$\text{mm/pixel}$）：

$$
s = \frac{D_{\text{known, mm}}}{d_{\text{known, px}}}
$$

任何由演算法測得之像素距離 $d_{\text{pixel}}$，其對應的物理真值換算公式為：

$$
D_{\text{physical, mm}} = d_{\text{pixel}} \times s
$$

##### (B) 光學成像模型與幾何物理條件
針孔相機模型 (Pinhole Camera Model) 表明：

$$
d_{\text{px}} = f \cdot \frac{D_{\text{real}}}{Z}
$$

其中 $f$ 為鏡頭焦距 (以像素計)，$Z$ 為物體距離感測平面的工作深度 (Depth)。比例尺 $s = \frac{Z}{f}$ 之有效性需依賴以下關鍵工程前提：
1. **共面假設 (Coplanar Assumption)**：
   - 測量目標（動物解剖點）與基準參考物必須位於同一景深平面 ($Z_{\text{target}} \approx Z_{\text{reference}}$)。
   - 若深度產生顯著落差 $\Delta Z$，透視縮放效應將導致尺寸估算產生 $\frac{\Delta Z}{Z}$ 的線性系統誤差。
2. **鏡頭畸變校正 (Lens Distortion Correction)**：
   - 廣角或普通定焦鏡頭邊緣存在徑向畸變 (Radial Distortion, $k_1, k_2$) 與切向畸變 (Tangential Distortion, $p_1, p_2$)。
   - 在執行像素測距前，系統必須利用標定棋盤格預先獲取相機內參矩陣 $\mathbf{K}$，並呼叫 `cv2.undistort` 將畸變像素點映射回無失真的理想針孔幾何平面。
3. **已知基準物校正 (Fiducial Marker Calibration)**：
   - 在牧場圍欄、量測地磅或動物標籤處固定標定板（例如 ArUco Marker、AprilTag 或已知寬度的標準色卡）。
   - 演算法即時偵測標記四角點並計算其像素寬度 $d_{\text{marker, px}}$，自動刷新當前畫面的動態比例尺 $s$。

---

#### 3. 交叉驗證機制 (Cross-Validation against Ground Truth & Known Markers)

為了確認測量系統的準確度、穩定性與工程可信度，建議採用**兩級交叉驗證流程**。目前專案沒有相機標定板自動偵測或游標卡尺真值資料；以下是部署前的驗證方法，不是已完成的實體量測驗證：

```mermaid
flowchart LR
    A["高精度驗證標準"] --> B["1. 畫面內已知參考標記物 (ArUco / 固定標尺)"]
    A --> C["2. 人工實體游標卡尺真值 (Ground Truth, GT)"]
    
    B --> D["驗證管線: 比例尺自洽性檢驗"]
    C --> E["驗證管線: 形態統計誤差檢驗"]
    
    D & E --> F["誤差統計指標計算: MAE / RMSE / RE%"]
    F --> G{"是否滿足誤差門檻 (如 MAE < 1.0mm)?"}
    G -- 是 --> H["通過驗證，進入正式計量"]
    G -- 否 --> I["發出警訊，觸發相機重新標定"]
```

##### (A) 核心誤差評估數學指標

設測試樣本集共包含 $N$ 組比對樣本，$D_{\text{pred}}^{(k)}$ 為系統測得之毫米距離，$D_{\text{gt}}^{(k)}$ 為高精度游標卡尺測得之真值：

1. **平均絕對誤差 (Mean Absolute Error, MAE)**：
   衡量系統整體平均偏差大小：
   $$MAE = \frac{1}{N} \sum_{k=1}^N \left| D_{\text{pred}}^{(k)} - D_{\text{gt}}^{(k)} \right|$$

2. **均方根誤差 (Root Mean Square Error, RMSE)**：
   對大誤差極度敏感，用以檢驗系統是否存在重大離群偏差：
   $$RMSE = \sqrt{\frac{1}{N} \sum_{k=1}^N \left( D_{\text{pred}}^{(k)} - D_{\text{gt}}^{(k)} \right)^2}$$

3. **相對誤差百分比 (Relative Error, RE%)**：
   衡量相對於量測目標尺度的相對準確度：
   $$RE\% = \frac{1}{N} \sum_{k=1}^N \left( \frac{\left| D_{\text{pred}}^{(k)} - D_{\text{gt}}^{(k)} \right|}{D_{\text{gt}}^{(k)}} \times 100\% \right)$$

##### (B) 兩級交叉驗證執行架構
1. **第一級：場景標記自洽驗證 (Online In-frame Marker Self-Audit)**：
   - 影像中同時擺放兩組不同位置的已知標記物（標記 A 用於計算比例尺，標記 B 用於閉環驗證）。
   - 若由標記 A 算出的比例尺估算標記 B 之實體尺寸誤差大於 $1.5\%$，系統立即發出相機角度偏斜或景深不一致的警訊。
2. **第二級：人機盲測交叉審查 (Offline Caliper Cross-Audit)**：
   - 由專業獸醫師或品管員以高精度游標卡尺隨機抽檢 $N \ge 30$ 隻動物標本之實測雙眼距離 $D_{\text{gt}}$。
   - 計算本系統輸出值與手動實測值之 MAE 與 RMSE。當系統滿足 $MAE \le 1.0\,\text{mm}$ 且 $RE\% \le 2.0\%$ 時，確認系統通過驗證標準。

---

## 📄 授權與第三方素材 (Licenses & Third-Party Assets)

本專案原始碼目前未附 `LICENSE`，因此尚未授予他人依特定開源條款使用、修改或再散布本專案程式碼的權利。公開程式碼前，應先由專案負責人決定並加入適用的程式碼授權。

下列第三方模型、資料集與影像適用各自的來源條款。將其納入發行、部署或公開發布前，應確認授權是否允許相應的使用與再散布；僅標示來源不代表已取得再散布權：

| 素材 | 專案中的檔案／來源 | 分發前應確認事項 |
| :--- | :--- | :--- |
| Ultralytics YOLO 權重 | `yolov8n-seg.pt`、`yolov8n-pose.pt` | 確認 Ultralytics 適用的 AGPL-3.0 或 Enterprise 授權條款，以及權重再散布方式。參考 [Ultralytics License](https://www.ultralytics.com/license) 與 [授權原文](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。 |
| 微調的貓狗眼睛模型 | 兩關鍵點 Pose checkpoint（不含於原始碼儲存庫） | 由預訓練權重及 Animal-Pose 訓練資料衍生；分發前應確認基礎模型及訓練資料條款允許散布衍生權重。 |
| COCO 影像與標註 | COCO 資料集影像及其來源 | COCO 影像可能各自沿用原始影像的授權；應依 COCO 資料集頁面及個別影像來源確認條款。參考 [COCO Dataset](https://cocodataset.org/)。 |
| Animal-Pose 訓練影像與標註 | Animal-Pose 來源資料及其轉換標註 | 確認原始資料集及各影像的使用、修改與再散布條款；轉換標註格式不會自動產生再散布權。 |
| 示範影像與其標註輸出 | 雙貓示範影像及其衍生標註圖 | 確認來源授權涵蓋複製與再散布；CSV 僅記錄影像參照與量測數值，不附原圖。 |

原始碼儲存庫不包含第三方影像、訓練資料、模型權重或推論影像。`.env` 可能含 API key 等機密設定，不得提交或轉交；環境設定範例請使用 `.env.example`，並避免填入有效憑證。
