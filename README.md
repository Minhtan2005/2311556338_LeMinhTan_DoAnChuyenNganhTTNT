# Trợ lý ảo thông minh hỗ trợ phân tích và truy xuất thông tin trong dữ liệu video giám sát

Đồ án chuyên ngành Trí tuệ nhân tạo.

**Sinh viên:** Lê Minh Tân  
**MSSV:** 2311556338  

## Giới thiệu

Hệ thống hỗ trợ phân tích video giám sát bằng trí tuệ nhân tạo, cho phép phát hiện và theo dõi đối tượng, lưu trữ kết quả phân tích, truy xuất theo thời gian và hỗ trợ truy vấn bằng ngôn ngữ tự nhiên.

## Chức năng chính

- Upload và phân tích video.
- Phát hiện đối tượng bằng YOLOv8.
- Theo dõi đối tượng bằng ByteTrack.
- Hỗ trợ 6 lớp: `person`, `car`, `motorcycle`, `bicycle`, `bus`, `truck`.
- Lưu dữ liệu phân tích bằng SQLite.
- Trích xuất keyframe và timestamp.
- Tìm kiếm đối tượng trong video.
- Truy vấn bằng tiếng Việt.
- Hỗ trợ Gemini NLP khi được cấu hình.
- Hỗ trợ YOLO-World cho truy vấn đối tượng ngoài các lớp chính.

## Công nghệ sử dụng

**Backend**
- Python
- FastAPI
- SQLAlchemy
- SQLite
- Ultralytics YOLO
- ByteTrack

**Frontend**
- React
- TypeScript
- Vite

**AI**
- YOLOv8n fine-tuned
- YOLO-World
- Gemini API
- NLP cục bộ

## Cấu trúc dự án

```text
├── backend/
│   ├── app/
│   ├── models/
│   ├── scripts/
│   └── tests/
├── frontend/
│   └── src/
├── docs/
├── scripts/
├── .env.example
├── run_backend.bat
├── run_frontend.bat
└── README.md
