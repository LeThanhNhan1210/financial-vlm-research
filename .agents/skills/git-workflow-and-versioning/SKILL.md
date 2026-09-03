---
name: git-workflow-and-versioning
description: >-
  Quy chuẩn Git, Conventional Commits và quản lý phiên bản thực nghiệm cho dự án AI/VLM.
  Kích hoạt kỹ năng này khi tạo commit, mở nhánh thử nghiệm mới (experiment branches),
  gắn tag phiên bản benchmark/mô hình, hoặc chuẩn bị đẩy code lên GitHub để chạy trên Colab.
---

# QUY TRÌNH GIT VÀ QUẢN LÝ PHIÊN BẢN (GIT WORKFLOW & VERSIONING)

Quy chuẩn kiểm soát phiên bản mã nguồn sạch sẽ, minh bạch và an toàn, ngăn chặn việc commit các file dữ liệu nặng hoặc secret key lên GitHub.

---

## 1. NGUYÊN TẮC VÀNG TRONG DỰ ÁN MACHINE LEARNING

*   **TUYỆT ĐỐI KHÔNG commit dữ liệu thô và trọng số mô hình:**
    *   Thư mục `data/raw/`, `data/processed/`, `checkpoints/` đã được chặn bởi `.gitignore`.
    *   Trước khi chạy lệnh commit, luôn kiểm tra lại trạng thái: `git status`.
*   **Không commit file chứa API Key:**
    *   File `.env` hoặc file notebook có chứa `OPENAI_API_KEY`, `HUGGINGFACE_TOKEN` phải luôn được bảo mật.

---

## 2. QUY CHUẨN ĐẶT TÊN COMMIT (CONVENTIONAL COMMITS)

Mọi commit phải có tiền tố mô tả rõ bản chất công việc:

| Tiền tố | Mục đích sử dụng | Ví dụ trong dự án VLM |
| :--- | :--- | :--- |
| `feat:` | Thêm tính năng mới | `feat(pipeline): implement ocr cross-validation layer` |
| `fix:` | Sửa lỗi hệ thống | `fix(models): resolve fp16 precision mismatch on colab t4` |
| `data:` | Cập nhật cấu trúc nhãn | `data(splits): generate time-series train-val-test splits` |
| `exp:` | Thử nghiệm mô hình | `exp(prompt): evaluate zero-shot vs few-shot cot baseline` |
| `docs:` | Cập nhật tài liệu | `docs(adr): add adr-0002 for time-series split rationale` |
| `refactor:`| Tái cấu trúc code | `refactor(loader): modularize bitsandbytes int4 config` |
| `perf:` | Tối ưu hóa hiệu năng | `perf(vram): enable gradient checkpointing for qlora` |

---

## 3. CHIẾN LƯỢC PHÂN NHÁNH THỰC NGHIỆM (EXPERIMENT BRANCHING)

Thay vì thử nghiệm hỗn loạn trên nhánh chính `main`, hãy sử dụng nhánh riêng cho từng hướng tiếp cận:
```text
main (mã nguồn ổn định nhất)
 ├── exp/few-shot-cot       # Nhánh nghiên cứu tối ưu hóa Prompting
 ├── exp/qlora-int4         # Nhánh nghiên cứu Fine-tuning QLoRA
 └── exp/ocr-pipeline       # Nhánh phát triển module OCR chống ảo giác
```

Khi thử nghiệm chứng minh có hiệu quả vượt trội (đạt F1/Accuracy cao hơn), mới tiến hành gộp (merge) về `main`.

---

## 4. GẮN TAG PHIÊN BẢN CHO KẾT QUẢ ĐẠT ĐƯỢC (VERSION TAGGING)

Gắn tag Git tương ứng với từng mốc báo cáo khoa học hoặc phiên bản mô hình:
```bash
# Tag khi hoàn thành mốc Baseline Few-shot
git tag -a v0.1.0-baseline -m "Baseline evaluation: Zero-shot vs Few-shot CoT"

# Tag khi huấn luyện xong QLoRA checkpoint đầu tiên
git tag -a v0.2.0-qlora -m "First stable QLoRA 4-bit adapter checkpoint on Colab T4"

# Đẩy tag lên GitHub
git push origin --tags
```
