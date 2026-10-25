# RAG Chatbot mẫu có note

Đây là chatbot chạy trong terminal, dùng để minh hoạ luồng:

`tài liệu .txt → chunking → embedding → ChromaDB → semantic search → LLM trả lời → memory + streaming`

`app.py` đã có note tiếng Việt cho từng phần. Dữ liệu trong `knowledge_base/student_faq.txt` là dữ liệu giả lập để thực hành; học sinh có thể thay bằng FAQ của một câu lạc bộ, lớp học hoặc trường.

## 1. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` thành `.env`, sau đó điền `OPENAI_API_KEY`.

## 2. Chạy lần đầu

```bash
python app.py --rebuild
```

Tùy chọn `--rebuild` sẽ đọc lại các file `.txt`, chia chunk và tạo embedding mới. Khi chỉ chat với cùng bộ tài liệu, dùng:

```bash
python app.py
```

## 3. Lệnh khi chat

- `/history`: xem các lượt hội thoại chatbot đang nhớ.
- `/reset`: xóa memory của phiên chat hiện tại.
- `/exit`: thoát chương trình.

## 4. Kịch bản demo nhanh trên lớp

1. Hỏi: `Điều kiện để được dự thi là gì?`
2. Hỏi tiếp: `Nếu em chỉ tham dự 75% thì sao?` — để học sinh thấy chatbot dùng memory.
3. Hỏi: `Một nhóm đồ án có mấy người?`
4. Hỏi ngoài tài liệu: `Học phí kỳ này là bao nhiêu?` — để quan sát chatbot từ chối bịa thông tin.

## 5. Bài mở rộng cho học sinh

1. Thêm file FAQ mới và chạy lại với `--rebuild`.
2. Thử thay đổi `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`; ghi nhận câu trả lời thay đổi ra sao.
3. Thêm metadata `category` rồi chỉ tìm kiếm trong một nhóm tài liệu nhất định.
4. Lưu lịch sử chat ra file JSON.
