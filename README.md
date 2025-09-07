# LLM‑based Weak Supervision Framework for Query Intent Classification in Video Search

> Phân loại **ý định truy vấn** (query intent) trong bối cảnh tìm kiếm video bằng **giám sát yếu** (weak supervision) dựa trên **LLM**.
> Tập trung vào **pipeline nghiên cứu** gọn nhẹ (chạy CPU), **demo web** đơn giản và **báo cáo tái lập**.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Điểm nổi bật](#điểm-nổi-bật)
- [Kiến trúc & Pipeline](#kiến-trúc--pipeline)
- [Taxonomy nhãn](#taxonomy-nhãn)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt & Thiết lập](#cài-đặt--thiết-lập)
- [Chạy nhanh (Quickstart)](#chạy-nhanh-quickstart)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cấu hình dự án](#cấu-hình-dự-án)
- [Huấn luyện & Đánh giá](#huấn-luyện--đánh-giá)
- [Demo web (Gradio) & API](#demo-web-gradio--api)
- [Mở rộng & Tùy biến](#mở-rộng--tùy-biến)
- [Kết quả mẫu & Báo cáo](#kết-quả-mẫu--báo-cáo)
- [Hạn chế, Quyền riêng tư & Đạo đức](#hạn-chế-quyền-riêng-tư--đạo-đức)
- [Lộ trình (Roadmap)](#lộ-trình-roadmap)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)
- [Trích dẫn](#trích-dẫn)
- [FAQ & Troubleshooting](#faq--troubleshooting)

---

## Tổng quan

Dự án xây dựng một **khung giám sát yếu** cho bài toán **phân loại ý định truy vấn** trong tìm kiếm video. Thay vì phụ thuộc vào tập nhãn lớn do con người gán, ta **kết hợp nhiều nguồn nhãn “yếu”**:

- **Luật/regex từ khóa** (VN/EN),
- **Zero‑shot NLI** (ví dụ `facebook/bart-large-mnli`) để suy nhãn trực tiếp,
- **LLM‑labeler** (prompt LLM → chọn 1 nhãn + lý do).

Các nguồn này được trộn bằng **Label Model** (Snorkel hoặc majority vote có trọng số) để tạo **nhãn xác suất**, sau đó huấn luyện một **End‑Model nhẹ** (ví dụ SetFit hoặc Logistic Regression trên sentence embeddings). Cuối cùng, một **demo web** hiển thị dự đoán, xác suất và “phiếu bầu” từ các nguồn nhãn.

**Phù hợp:** đề tài thực tập theo hướng **nghiên cứu + demo**; yêu cầu Python cơ bản, không cần GPU.

---

## Điểm nổi bật

- 🧩 **Giám sát yếu**: kết hợp rules + zero‑shot + LLM để tiết kiệm công gán nhãn.
- 🏷️ **Taxonomy 6 lớp** tối ưu cho bối cảnh video (KIS, How‑to, Music, News, Sports, Review).
- 🧪 **Ablation**: so sánh Rules‑only vs Zero‑shot vs Weak‑sup → End‑Model.
- 🧼 **Làm sạch nhãn (tùy chọn)** bằng cleanlab.
- 🖥️ **Demo web**: Gradio UI, kèm **REST API** (FastAPI) để tích hợp FE.
- 📝 **Báo cáo tái lập**: scripts, config, seed cố định.

---

## Kiến trúc & Pipeline

```
             +------------------------+
Data (text)  |  queries.csv / synthetic|  (MSR-VTT/TVR: chỉ dùng text mô tả/truy vấn;
    ───────▶ +------------------------+   dữ liệu demo: data/sample_queries.csv)
                     │
                     ▼
        +------------------------+      +---------------------------+
LFs --->| Keyword Rules (VN/EN) |      | Zero-shot NLI (BART-MNLI) |
        +------------------------+      +---------------------------+
                     │                           │
                     ├──────────────┬────────────┤
                     ▼              ▼            ▼
              +----------------------------------------+
              |     Weak Labels (per source)           |
              +----------------------------------------+
                               │
                               ▼
                    +----------------------+
                    |  Label Model (WS)    |  →  p(y|x)
                    | (Snorkel / Weighted) |
                    +----------------------+
                               │
                               ▼
                +-------------------------------+
                |  End-Model (SetFit / LogReg)  |
                +-------------------------------+
                               │
                               ▼
                +-------------------------------+
                | Evaluation & Error Analysis   |
                +-------------------------------+
                               │
                               ▼
                +-------------------------------+
                |   Demo Web (Gradio / API)     |
                +-------------------------------+
```

---

## Taxonomy nhãn

1. **KIS** (Known‑Item Search): Tìm đúng video/kênh/tập cụ thể
2. **How‑to / Tutorial**
3. **Music / Karaoke / Lyrics**
4. **News / Thời sự**
5. **Sports / Highlights / Live**
6. **Review / Unboxing / So sánh**

> Bạn có thể tùy biến taxonomy trong `config/taxonomy.yaml`.

---

## Yêu cầu hệ thống

- Python **3.10+**
- CPU đủ RAM \~4–8GB (không yêu cầu GPU)
- (Tùy chọn) Docker 24+

---

## Cài đặt & Thiết lập

### 1) Clone & tạo môi trường

```bash
git clone https://github.com/<your-username>/video-intent-weak-supervision.git
cd video-intent-weak-supervision

# Khuyến nghị: dùng venv/conda
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 2) Cài dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Nếu chưa có `requirements.txt`, bạn có thể bắt đầu với:

```bash
pip install snorkel transformers torch accelerate datasets setfit scikit-learn \
            cleanlab pandas numpy scipy pyyaml gradio fastapi uvicorn python-dotenv \
            rich tqdm
```

### 3) (Tuỳ chọn) Thiết lập biến môi trường

Tạo file `.env` nếu cần dùng API LLM:

```
# Ví dụ (tuỳ nhà cung cấp):
OPENAI_API_KEY=<your_key>
# hoặc các khóa model khác (nếu dùng)
```

> Mặc định pipeline có thể chạy **không cần LLM** (chỉ rules + zero‑shot).

---

## Chạy nhanh (Quickstart)

### 0) Dữ liệu mẫu

- `data/sample_queries.csv` (đã kèm): cột `text` chứa truy vấn.
- Bạn có thể thêm dòng của riêng bạn. Ví dụ:

```csv
text
"cách sửa lỗi wifi windows 11"
"đom đóm lyrics karaoke"
"MU vs Liverpool highlights"
"thời sự 19h hôm nay"
"Conan tập 100 vietsub"
"review iphone 15 vs s23"
```

### 1) Tạo nhãn yếu

```bash
python scripts/01_generate_weak_labels.py \
  --input data/sample_queries.csv \
  --output outputs/weak_labels.parquet \
  --config config/config.yaml
```

Kịch bản trên sẽ:

- Chạy **LFs từ khóa** (VN/EN),
- Chạy **zero‑shot** (`facebook/bart-large-mnli`),
- (Tuỳ chọn) gọi **LLM‑labeler** nếu `.env` có khoá,
- Lưu “phiếu bầu”/điểm của từng nguồn.

### 2) Hợp nhất bằng Label Model

```bash
python scripts/02_label_model.py \
  --weak outputs/weak_labels.parquet \
  --output outputs/pseudo_labeled.parquet \
  --config config/config.yaml
```

Kết quả: mỗi truy vấn có **nhãn xác suất** và **nhãn giả (pseudo‑label)**.

### 3) Huấn luyện End‑Model

```bash
python scripts/03_train_end_model.py \
  --train outputs/pseudo_labeled.parquet \
  --save_dir artifacts/end_model \
  --config config/config.yaml
```

### 4) Đánh giá trên tập test vàng (nếu có)

```bash
python scripts/04_evaluate.py \
  --model_dir artifacts/end_model \
  --gold data/gold_test.csv \
  --report outputs/report.json \
  --config config/config.yaml
```

### 5) Chạy demo

```bash
# Gradio UI
python app/gradio_app.py --model_dir artifacts/end_model --config config/config.yaml

# hoặc REST API (FastAPI)
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Cấu trúc thư mục

```
├── app/
│   ├── api.py                  # FastAPI: /predict, /explain
│   └── gradio_app.py           # Demo UI (chạy nhanh)
├── artifacts/                  # Model + tham số đã train (có thể check-in nhẹ, không nặng)
│   ├── end_model/              # weights, label encoder, tokenizer…
│   └── label_model/            # (MỚI) tham số/đồ thị LabelModel, ngưỡng p, v.v.
├── config/
│   ├── config.yaml             # cấu hình chung (models, thresholds, paths)
│   ├── taxonomy.yaml           # 6 nhãn + alias/từ khóa gợi ý
│   └── prompts/                # (MỚI, tùy chọn) prompt cho LLM-labeler (JSON/YAML)
├── data/
│   ├── raw/                    # (MỚI) dữ liệu gốc chỉ-đọc (caption/query text từ MSR-VTT/TVR, synthetic)
│   │   ├── msrvtt_captions.csv
│   │   ├── tvr_queries.jsonl   # (tùy chọn) nếu bạn thêm TVR sau
│   │   └── synthetic_queries.csv
│   ├── processed/              # (MỚI) dữ liệu sau tiền xử lý (pool, splits)
│   │   ├── unlabeled_pool.csv
│   │   └── gold_test.csv       # tập test vàng bạn tự gán nhãn
│   └── README.md               # mô tả nguồn dữ liệu & điều khoản sử dụng (text-only)
├── notebooks/                  # (tùy chọn) EDA, ablation, error analysis
│   └── 01_eda_intent.ipynb
├── outputs/                    # KẾT QUẢ CHẠY — không check-in nặng (có thể .gitignore)
│   ├── metrics/                # (MỚI) metrics_*.json, classification_report
│   ├── figures/                # (MỚI) confusion_matrix_*.png, biểu đồ
│   ├── predictions/            # (MỚI) predictions_*.csv
│   └── logs/                   # (MỚI) log, seed, thời gian chạy
├── scripts/
│   ├── 00_build_pool.py        # (MỚI) gom text MSR-VTT/TVR + synthetic → data/processed/unlabeled_pool.csv
│   ├── 01_generate_weak_labels.py
│   ├── 02_label_model.py
│   ├── 03_train_end_model.py
│   ├── 04_evaluate.py
│   └── 05_run_baselines.py     # (MỚI) Zero-shot & Rules-only + xuất metrics/CM/predictions
├── lfs/
│   ├── keyword_lfs.py          # labeling functions (regex/từ khóa)
│   └── patterns.yaml           # (MỚI, tiện maintain) danh sách từ khóa/regex theo lớp
├── src/                        # (MỚI) thay cho "utils/" để gọn gàng theo kiểu package
│   └── video_intent_ws/
│       ├── __init__.py
│       ├── data_io.py
│       ├── seed.py
│       ├── metrics.py
│       ├── models.py           # Zero-shot wrapper, SetFit/LogReg loader
│       └── label_model_utils.py
├── tests/                      # (MỚI, nhỏ gọn) kiểm thử tối thiểu
│   ├── test_lfs.py             # test regex khớp đúng lớp
│   └── test_config.py          # test load config/taxonomy không lỗi
├── .gitignore                  # (MỚI) bỏ qua outputs/, *.ckpt, .venv/, __pycache__/...
├── .env.example
├── pyproject.toml              # (MỚI, khuyến nghị) khai báo package + tool (ruff/pytest)
├── requirements.txt
├── Makefile                    # (MỚI, tiện chạy) make pool | baselines | ws | train | eval | demo
├── LICENSE
└── README.md

```

---

## Cấu hình dự án

**`config/config.yaml` (ví dụ):**

```yaml
seed: 42
labels:
  - KIS
  - How-to
  - Music
  - News
  - Sports
  - Review

zero_shot:
  enabled: true
  model_name: facebook/bart-large-mnli
  multi_label: false

llm_labeler:
  enabled: false # Bật nếu có API key
  provider: openai
  model: gpt-4o-mini
  temperature: 0.2
  max_tokens: 64

label_model:
  type: snorkel # hoặc weighted_majority
  min_confidence: 0.7

end_model:
  type: setfit # hoặc logreg
  base: sentence-transformers/paraphrase-mpnet-base-v2
  epochs: 2
  batch_size: 16

paths:
  taxonomy: config/taxonomy.yaml
```

**`config/taxonomy.yaml` (ví dụ rút gọn):**

```yaml
KIS:
  aliases:
    [known-item, episode, ep, season, trailer, vietsub, official mv, channel]
How-to:
  aliases: [how to, tutorial, hướng dẫn, cách, fix, sửa, mẹo]
Music:
  aliases: [lyrics, lyric, karaoke, mv, official audio, beat, nhạc]
News:
  aliases: [news, breaking, thời sự, bản tin, nóng]
Sports:
  aliases: [vs, highlight, full match, trận, bán kết, chung kết, premier league]
Review:
  aliases: [review, đánh giá, so sánh, vs, unboxing, giá]
```

---

## Huấn luyện & Đánh giá

### Thực nghiệm đề xuất

- **So sánh**:

  1. **Rules‑only**
  2. **Zero‑shot**
  3. **Rules + Zero‑shot**
  4. **+ LLM‑labeler**
  5. **Weak‑sup → End‑Model** (kết quả chính)

- **Chỉ số**: Macro‑F1, Accuracy, Confusion Matrix.

- **Làm sạch** (tuỳ chọn): dùng cleanlab tìm điểm nghi ngờ → duyệt thủ công một phần nhỏ.

### Tái lập

- Cố định `seed` trong `config.yaml`.
- Log tham số, phiên bản package (ghi trong `outputs/report.json`).
- Nếu dùng LLM‑labeler, lưu prompt & phiên bản model vào `outputs/metadata.json`.

---

## Demo web (Gradio) & API

### Gradio

```bash
python app/gradio_app.py --model_dir artifacts/end_model --config config/config.yaml
```

- Nhập truy vấn → xem:

  - Dự đoán **End‑Model**,
  - Zero‑shot (nhãn, xác suất),
  - “Phiếu bầu” từ các **LFs** (đếm số luật khớp).

### FastAPI (REST)

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**

- `POST /predict`
  **Body:**

  ```json
  { "text": "conan tập 100 vietsub" }
  ```

  **Response:**

  ```json
  {
    "label": "KIS",
    "proba": 0.91,
    "votes": {
      "KIS": 2,
      "How-to": 0,
      "Music": 0,
      "News": 0,
      "Sports": 0,
      "Review": 0
    },
    "zero_shot": { "label": "KIS", "score": 0.83 }
  }
  ```

- `GET /labels` → trả danh sách nhãn/alias từ `taxonomy.yaml`.

**Frontend gợi ý:** FE (Next.js/Tailwind) gọi `/predict` và hiển thị badge nhãn + thanh xác suất + bảng phiếu bầu.

---

## Mở rộng & Tùy biến

- **Thêm LFs**: chỉnh `lfs/keyword_lfs.py` (regex/từ khóa VN/EN).
- **Đổi model zero‑shot**: sửa `config.yaml` phần `zero_shot`.
- **Thay End‑Model**: chuyển `end_model.type` sang `logreg` nếu muốn thật nhẹ.
- **Đổi taxonomy**: cập nhật `config/taxonomy.yaml` + rà lại LFs.

**Ví dụ LF rút gọn:**

```python
import re

def lf_music(text: str):
    return "Music" if re.search(r"\blyrics?\b|karaoke|mv|official audio|beat|nhạc|lyric\b", text, re.I) else None
```

---

## Kết quả mẫu & Báo cáo

- **outputs/report.json**: chứa điểm số (Accuracy, Macro‑F1), ma trận nhầm lẫn, v.v.
- **notebooks/**: notebook minh hoạ EDA, ablation và error analysis.
- **Báo cáo**: mục lục gợi ý

  1. Giới thiệu
  2. Liên quan (query intent, datasets, weak supervision)
  3. Phương pháp (taxonomy, LFs, label model, end‑model)
  4. Thực nghiệm (dữ liệu, cấu hình)
  5. Kết quả & Phân tích
  6. Bàn luận (ưu/nhược, đạo đức)
  7. Kết luận & hướng mở

---

## Hạn chế, Quyền riêng tư & Đạo đức

- **Dữ liệu**: repo chỉ cung cấp **text truy vấn mẫu/synthetic** để tránh vấn đề bản quyền.
- **Thiên lệch**: từ khóa và LLM có thể mang bias; cần đánh giá chéo và mô tả rõ hạn chế.
- **Tôn trọng điều khoản**: không crawl nội dung bị hạn chế/vi phạm ToS.
- **Minh bạch**: công bố taxonomy, rules, prompt; cung cấp phân tích lỗi.

---

## Lộ trình (Roadmap)

- [ ] Bộ test vàng song ngữ VN/EN \~300 câu
- [ ] Nâng cấp explainability (highlight từ khóa + lý do LLM)
- [ ] Dockerfile & Compose cho API + UI
- [ ] Thêm pipeline CI (lint, tests)
- [ ] So sánh nhiều sentence‑embeddings cho End‑Model

---

## Đóng góp

Đóng góp chào mừng! Vui lòng:

1. Fork repo → tạo nhánh `feature/<tên>`
2. Viết code kèm docstring, type hints
3. Thêm test tối thiểu (pytest) nếu sửa logic
4. Tạo PR mô tả thay đổi, screenshot/log nếu có

---

## Giấy phép

Phát hành theo **MIT License**. Xem file `LICENSE`.

---

## Trích dẫn

Nếu bạn dùng mã/ý tưởng từ repo này trong báo cáo/đồ án, vui lòng trích dẫn:

```
@software{video_intent_ws_2025,
  title        = {LLM-based Weak Supervision for Query Intent Classification in Video Search},
  author       = {Your Name},
  year         = {2025},
  url          = {https://github.com/<your-username>/video-intent-weak-supervision}
}
```

---

## FAQ & Troubleshooting

**Q:** Không có GPU, chạy được không?
**A:** Được. Pipeline thiết kế cho **CPU**, thời gian hợp lý nhờ model nhẹ và batch nhỏ.

**Q:** Không có khóa LLM?
**A:** Tắt `llm_labeler.enabled` trong `config.yaml` → vẫn có rules + zero‑shot.

**Q:** Thiếu dữ liệu tiếng Việt?
**A:** Bổ sung **synthetic queries** theo từng nhãn trong `data/`, hoặc trích câu truy vấn công khai (chỉ text).

**Q:** Điểm thấp ở lớp gần nhau (vd. KIS vs Music MV)?
**A:** Tinh chỉnh **alias**/regex cho KIS (episode/tập/season/trailer/channel) và Music (lyrics/karaoke/mv/official audio), tăng **min_confidence** khi lấy pseudo‑label, hoặc thêm **ít nhãn tay** cho lớp khó.

**Q:** Làm sao tái lập kết quả?
**A:** Giữ toolchain, phiên bản package và `seed` cố định; log mọi config + kết quả trong `outputs/`.

---

### Phụ lục: Lệnh hữu ích

```bash
# Đóng băng phiên bản lib
pip freeze > requirements.txt

# Kiểm tra style
python -m pip install ruff
ruff check .

# Chạy test (nếu có)
pytest -q
```
