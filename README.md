# Ứng dụng Vision-Language Models (VLMs) Trong Phân Tích Biểu Đồ Tài Chính

Đề tài nghiên cứu khoa học: Khảo sát và tối ưu hóa các mô hình thị giác ngôn ngữ mã nguồn mở (Qwen-VL, Llama-Vision) phục vụ phân tích kỹ thuật và đề xuất chiến lược giao dịch trên hạ tầng Google Colab (GPU T4 16GB).

---

## Cấu Trúc Dự Án

```text
├── configs/          # Quản lý siêu tham số huấn luyện, QLoRA, Prompts, Rubrics
├── data/             # Dữ liệu phân tầng: raw, processed, annotations, splits
├── docs/             # Hồ sơ khoa học: ADRs và Quy chuẩn lưu trữ GitHub/Drive
├── notebooks/        # Jupyter/Colab notebooks cho thực nghiệm và demo
├── src/              # Mã nguồn module hóa (data, models, pipeline, training, evaluation)
├── checkpoints/      # Lưu trữ trọng số LoRA adapters
├── outputs/          # Logs, predictions và đồ thị kết quả
├── scripts/          # Script CLI chạy nhanh: run_train.py, run_evaluate.py
├── ke_hoach_trien_khai.md # Kế hoạch chi tiết 5 giai đoạn
├── requirements.txt  # Thư viện phụ thuộc
└── .gitignore        # Quy tắc loại trừ file nặng khỏi Git
```

> 📖 **Xem chi tiết tài liệu quy chuẩn kiến trúc:**  
> * [Quy chuẩn thiết kế cấu trúc lưu trữ GitHub & Google Drive](file:///e:/NCKH/docs/QUY_CHUAN_LUU_TRU_GITHUB_DRIVE.md)  
> * [ADR-0001: Lựa chọn mô hình VLM nền tảng và lượng hóa INT4](file:///e:/NCKH/docs/adrs/0001-chon-mo-hinh-vlm-va-luong-hoa-int4.md)


---

## Hướng Dẫn Vận Hành Trên Google Colab Qua GitHub

### Bước 1: Đẩy mã nguồn lên GitHub từ máy cục bộ
```bash
git init
git add .
git commit -m "feat: setup project scaffold and VLM pipeline"
git branch -M main
git remote add origin <URL_REPO_GITHUB_CUA_BAN>
git push -u origin main
```

### Bước 2: Khởi chạy trên Google Colab
1. Mở một notebook mới trên [Google Colab](https://colab.research.google.com/), chọn **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
2. Chạy ô đầu tiên để clone mã nguồn và cài đặt:
```bash
!git clone <URL_REPO_GITHUB_CUA_BAN>.git
%cd <TEN_THU_MUC_REPO>
!pip install -r requirements.txt
```
3. Mở file `notebooks/01_colab_workflow_demo.ipynb` hoặc chạy script:
```bash
!python scripts/run_train.py
!python scripts/run_evaluate.py
```
