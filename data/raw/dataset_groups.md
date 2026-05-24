# Dataset Groups - PPE Detection Project

## 1. Phân nhóm Dataset (Set Name / Tags trong Roboflow)

| Tên nhóm (Set Name) | Loại hình ảnh | Tỷ lệ khuyên dùng | Mục đích (Tại sao cần?) |
|---|---|---|---|
| **CORE_CCTV** | Ảnh từ camera an ninh, góc cao, nhìn xuống công trường. | 60% - 70% | **Dữ liệu thực tế:** Đây là môi trường hệ thống sẽ chạy thật. |
| **FEATURE_RICH** | Ảnh selfie rõ nét, ảnh xóa phông (người rõ - cảnh mờ), ảnh cận cảnh. | 15% - 20% | **Bài tập nâng cao:** Giúp mô hình học cực kỳ chi tiết cái khóa mũ, sợi vải phản quang. |
| **HARD_EXAMPLES** | Ảnh mờ do khoảng cách, ảnh thiếu sáng (ban đêm), ảnh công nhân bị che khuất một nửa. | 5% - 10% | **Dữ liệu bổ trợ:** Giúp mô hình không bị "ngợp" khi gặp điều kiện thời tiết xấu hoặc đứng xa. |
| **NULL_BACKGROUND** | Ảnh công trường trống không, không có người, không có đồ bảo hộ. | 5% - 10% | **Chống báo động giả:** Dạy mô hình rằng "đây là cái máy xúc, không phải cái mũ bảo hiểm". |

Core
DIVERSE
Hard Data
Negative Data

PPE không chuẩn


Tóm tắt lại các phần cần chọn làm hard case

Đúng, chốt lại máy dễ sai ở các trường hợp:

Nhóm 1 — Mất thông tin kỹ thuật (mắt thường cũng khó nhìn)

Mờ, motion blur, low resolution
Ánh sáng quá tối / ngược sáng
Bị che khuất nhiều

Nhóm 2 — Màu trộn vào nhau (mắt thường nhìn được nhưng máy dễ nhầm)

Mũ vàng giữa áo phản quang vàng
Áo cam giữa công trình màu cam/đỏ
PPE màu tối trong môi trường tối

Nhóm 3 — Chưa thấy trong training (máy không có reference)

Góc bất thường (từ trên xuống, từ dưới lên)
Overlap nhiều người
Nhiều người dày đặc trong 1 frame


Về PPE không chuẩn — bạn đúng, đây là trường hợp phải tự tạo tình huống vì:

Dataset thường chỉ có PPE đội đúng cách → ngoài thực tế người ta hay đội lệch, để cằm, kính để trên đầu

→ Khó tìm ảnh sẵn → phải chủ động annotate hoặc augment những trường hợp này
Đây cũng là lý do PPE detection ngoài thực tế hay fail hơn trong lab.