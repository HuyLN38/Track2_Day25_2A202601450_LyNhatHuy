# Track2_Day25_2A202601450_LyNhatHuy

# Lab: Estimate Training FLOPs — Báo cáo kết quả

Notebook: [Day25_Track_2_1_Gpu_finops_cost_optimization.ipynb](Day25_Track_2_1_Gpu_finops_cost_optimization.ipynb)

## Task 1 — Công thức ước tính FLOPs

```python
def compute_num_flops(param_count: float, num_tokens: float) -> float:
    return 6 * param_count * num_tokens
```

Hệ số 6 = 2 (forward: 1 nhân + 1 cộng cho mỗi tham số / token) + 4 (backward: đạo hàm theo
activations + đạo hàm theo weights, tốn gấp đôi forward).

## Task 2 — FLOPs của các mô hình thực tế (1 epoch)

| Scenario | Parameters (P) | Tokens (N) | Training FLOPs |
|---|---|---|---|
| BERT-Base | 110 M (1.1e8) | 3.3 B (3.3e9) | 2.18e18 |
| T5-Large | 770 M (7.7e8) | 1 T (1e12) | 4.62e21 |
| Gemma-1B (Africa Galore) | 1 B (1e9) | 30 K (3e4) | 1.80e14 |
| PaLM | 540 B (5.4e11) | 780 B (7.8e11) | 2.53e24 |

**Gemma-1B vs BERT-Base:** Gemma nhiều tham số hơn 9.1× nhưng ít token hơn 110,000× →
tổng FLOPs nhỏ hơn 110,000 / 9.1 ≈ 12,100× (đúng bằng 2.18e18 / 1.80e14).

**Yếu tố quyết định:** về mặt toán học P và N có trọng số ngang nhau (FLOPs tuyến tính bậc 1
với cả hai). Yếu tố "mạnh hơn" là yếu tố thay đổi nhiều bậc độ lớn hơn trong bài toán cụ thể —
ở đây là N. Chi phí bùng nổ khi cả P và N cùng tăng, vì FLOPs tăng theo *tích* hai mức tăng.

## Task 3 — Linear Scaling Law trên thang log

Cố định N = 1e10, tăng P từ 1e8 → 1e12:

| log_param_count | Training FLOPs | Hệ số tăng |
|---|---|---|
| 8 | 6.00e18 | — |
| 9 | 6.00e19 | ×10 |
| 10 | 6.00e20 | ×10 |
| 11 | 6.00e21 | ×10 |
| 12 | 6.00e22 | ×10 |

Phần định trị giữ nguyên 6.00, chỉ số mũ tăng đúng 1 đơn vị mỗi bước → tăng P (hoặc N) lên k
lần thì FLOPs tăng đúng k lần. Trên đồ thị log–log đây là đường thẳng độ dốc 1.

## Task 4 — Quy đổi FLOPs → GPU Hours → Chi phí

Tình huống: model 7B tham số, dataset 2T token, 64× NVIDIA A100 (312 TFLOPS FP16/BF16),
MFU = 0.35, giá $2.50 / GPU-giờ.

| Hạng mục | Giá trị |
|---|---|
| Tổng FLOPs = 6 × 7e9 × 2e12 | 8.40e22 FLOPs |
| Effective FLOPS = 64 × 312e12 × 0.35 | 6.9888e15 FLOPS |
| Thời gian huấn luyện | 12,019,231 s ≈ 3,338.7 giờ ≈ 139.1 ngày |
| GPU Hours = 3,338.7 × 64 | 213,675 GPU-giờ |
| **Chi phí = 213,675 × $2.50** | **$534,188** |

Phân tích độ nhạy:

| MFU | 64 GPU | 128 GPU | 256 GPU | GPU Hours | Chi phí |
|---|---|---|---|---|---|
| 30% | 162.3 ngày | 81.1 ngày | 40.6 ngày | 249,288 | $623,219 |
| 35% | 139.1 ngày | 69.6 ngày | 34.8 ngày | 213,675 | $534,188 |
| 45% | 108.2 ngày | 54.1 ngày | 27.0 ngày | 166,192 | $415,480 |

Nhận xét FinOps:

1. Thêm GPU rút ngắn thời gian nhưng **không giảm chi phí** — tổng GPU-hours không đổi vì khối
   lượng công việc (8.4e22 FLOPs) không đổi. Thực tế cụm càng lớn thì MFU càng giảm do overhead
   giao tiếp, nên chi phí có xu hướng nhích lên.
2. **MFU là đòn bẩy chi phí thật sự**: nâng MFU từ 0.30 → 0.45 tiết kiệm ~$208,000 (33%) mà
   không thuê thêm GPU nào.
3. Một lần train 139 ngày gần như chắc chắn gặp lỗi node giữa chừng → dự toán phải cộng thêm
   ngân sách cho checkpointing và các lần restart.
