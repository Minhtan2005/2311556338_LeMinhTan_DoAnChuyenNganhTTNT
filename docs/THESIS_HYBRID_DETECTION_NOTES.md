# Ghi Chú Đồ Án: Hybrid Object Detection

## 1. Vấn Đề Của YOLOv8 Fixed-Class

YOLOv8n fine-tuned trong hệ thống hiện tại hoạt động ổn định cho nhóm đối tượng giao thông đã huấn luyện. Tuy nhiên, mô hình fixed-class chỉ nhận diện được các lớp có trong tập nhãn. Với đồ án này, tập lớp chính gồm:

```text
person
car
motorcycle
bicycle
bus
truck
```

Khi người dùng hỏi các đối tượng ngoài danh sách này, ví dụ `con chó`, `xe cứu thương`, `traffic cone`, hệ thống YOLOv8 fine-tuned không có lớp tương ứng để trả kết quả trực tiếp.

## 2. Lý Do Bổ Sung YOLO-World

YOLO-World hỗ trợ open-vocabulary object detection: mô hình có thể nhận lớp đối tượng dạng text prompt tại thời điểm inference. Vì vậy, YOLO-World phù hợp để làm fallback cho các truy vấn nằm ngoài 6 lớp giao thông chính.

Mục tiêu không phải thay thế YOLOv8n, mà là bổ sung khả năng tìm kiếm đối tượng mở rộng khi người dùng thật sự cần.

## 3. Kiến Trúc Hybrid

```text
Video
  |
  +--> YOLOv8n fine-tuned
  |       |
  |       +--> 6 traffic classes
  |
  +--> YOLO-World (on demand)
          |
          +--> open-vocabulary query
                  |
                  v
             normalized results
                  |
                  v
          Search / Timestamp / Video Seek
```

Pipeline phân tích video thông thường vẫn đi qua YOLOv8n fine-tuned, ByteTrack, SQLite, events, keyframes và search metadata. YOLO-World chỉ chạy khi truy vấn yêu cầu đối tượng ngoài 6 lớp chính.

## 4. Vai Trò Của Từng Model

YOLOv8n fine-tuned:

- Fixed class.
- Đã fine-tune cho bài toán giao thông.
- Là detector chính.
- Dùng cho regular analysis.
- Kết quả đi qua ByteTrack và lưu SQLite.

YOLO-World:

- Open vocabulary.
- Nhận danh sách lớp bằng text prompt.
- Là fallback/on-demand detector.
- Hỗ trợ tìm các lớp ngoài primary vocabulary, tùy năng lực mô hình và ngưỡng confidence.

## 5. Điểm Khác Nhau

| Tiêu chí | YOLOv8n fine-tuned | YOLO-World |
|---|---|---|
| Vocabulary | Cố định | Mở rộng bằng text prompt |
| Vai trò | Primary traffic detector | Open-vocabulary fallback |
| Thời điểm chạy | Khi analysis video | Khi query cần object ngoài 6 lớp |
| Tracking | ByteTrack trong normal analysis | Không ép tracking trong search on-demand |
| Tốc độ | Phù hợp regular analysis | Có thể nặng hơn, dùng sampling |
| Độ tin cậy demo | Cao cho lớp đã fine-tune | Phụ thuộc prompt/model/confidence |

## 6. Phần Đóng Góp Của Đồ Án

Không tuyên bố đồ án tạo ra YOLO hoặc YOLO-World. Phần đóng góp nên diễn đạt theo hướng:

- Thiết kế pipeline hybrid object detection cho video retrieval.
- Fine-tune detector giao thông trên nhóm lớp cần thiết.
- Tích hợp tracking bằng ByteTrack.
- Xây dựng query routing cho tiếng Việt.
- Bổ sung open-vocabulary fallback bằng YOLO-World.
- Chuẩn hóa kết quả detection về timestamp/segment.
- Xây dựng giao diện search/Q&A và click-to-seek theo timestamp.
- Giữ hành vi an toàn cho query màu sắc và relationship chưa đủ bằng chứng.

## 7. Đề Xuất Thí Nghiệm So Sánh

Không tự điền số liệu nếu chưa đo thật.

### Experiment A: YOLOv8n Pretrained vs YOLOv8n Fine-Tuned

Cùng một validation set.

| Model | Precision | Recall | mAP50 | Inference ms/image | FPS | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n pretrained | | | | | | |
| YOLOv8n fine-tuned | | | | | | |

### Experiment B: YOLOv8n vs YOLO-World

Cùng một tập ảnh/video benchmark được chọn.

| Query class | Detector | Detections | Precision | Recall | mAP50 nếu có GT | Inference ms | Query latency |
|---|---|---:|---:|---:|---:|---:|---:|
| car | YOLOv8n | | | | | | |
| car | YOLO-World | | | | | | |
| dog | YOLOv8n | | | | | | |
| dog | YOLO-World | | | | | | |
| ambulance | YOLOv8n | | | | | | |
| ambulance | YOLO-World | | | | | | |
| traffic cone | YOLOv8n | | | | | | |
| traffic cone | YOLO-World | | | | | | |

### Experiment C: Fixed Vocabulary Coverage

| Object | Trong 6 lớp YOLOv8n? | Route kỳ vọng | Kết quả thực tế | Ghi chú |
|---|---|---|---|---|
| car | Có | YOLOv8 metadata | | |
| person | Có | YOLOv8 metadata | | |
| motorcycle | Có | YOLOv8 metadata | | |
| dog | Không | YOLO-World | | |
| ambulance | Không | YOLO-World | | |
| traffic cone | Không | YOLO-World | | |

## 8. Known Limitations

- Open vocabulary không đồng nghĩa với perfect detection.
- Color attribute chưa được đảm bảo trong runtime mặc định.
- Relationship/action query chưa được giải quyết bởi YOLO-World.
- Performance phụ thuộc GPU/CPU, model và độ dài video.
- Sampling có thể bỏ sót object xuất hiện rất ngắn.
- YOLO-World có thể trả zero-result nếu prompt không phù hợp hoặc object không xuất hiện rõ.

## 9. Câu Mô Tả Ngắn Cho Báo Cáo

Hệ thống sử dụng YOLOv8n fine-tuned làm detector chính cho các đối tượng giao thông cố định, kết hợp ByteTrack để theo dõi và lưu metadata vào SQLite. Để mở rộng khả năng tìm kiếm ngoài tập nhãn cố định, hệ thống tích hợp YOLO-World dưới dạng detector fallback theo yêu cầu. Bộ query router phân tích truy vấn tiếng Việt, quyết định dùng metadata YOLOv8 cho 6 lớp chính hoặc gọi YOLO-World cho đối tượng mở rộng, sau đó chuẩn hóa kết quả thành timestamp/segment để người dùng có thể click và seek trực tiếp trong video.

## 10. Bổ Sung Gemini NLP

Gemini được sử dụng ở tầng xử lý ngôn ngữ tự nhiên để phân tích và chuẩn hóa truy vấn của người dùng thành biểu diễn có cấu trúc. Việc xác minh nội dung video vẫn được thực hiện bởi các mô hình thị giác máy tính.

Gemini không phân tích video, không nhận frame hoặc ảnh trong phần tích hợp này, và không tạo ra timestamp hay kết quả detection. Gemini chỉ hỗ trợ hiểu câu hỏi tiếng Việt, xác định intent, trích xuất object, attribute, relationship/action nếu có.

Kiến trúc NLP:

```text
Vietnamese Query
       |
       v
Gemini NLP
       |
       v
Structured Query
       |
       v
Capability Validation
       |
       v
Query Router
    /       \
YOLOv8    YOLO-World
   |          |
   +----+-----+
        |
     Results
```

Fallback khi Gemini lỗi:

```text
Gemini failure
      |
      v
Local NLP
      |
      v
Query Router
```

Vai trò của các thành phần:

- Gemini: hiểu truy vấn tiếng Việt, xác định intent, trích xuất object/attribute/relationship/action.
- YOLOv8n fine-tuned: detector chính cho 6 lớp giao thông cố định.
- YOLO-World: detector open-vocabulary chạy theo yêu cầu cho object ngoài tập lớp chính.
- ByteTrack: tracking và unique object count cho đường YOLOv8 primary.
- Local NLP: fallback offline để hệ thống vẫn hoạt động khi Gemini timeout, hết quota, lỗi JSON hoặc không có Internet.

Các truy vấn màu sắc và relationship vẫn đi qua capability validation. Vì vậy, việc Gemini hiểu `yellow car` hoặc `person riding motorcycle` không đồng nghĩa với việc pipeline thị giác đã có khả năng xác minh màu sắc hoặc quan hệ đó.
