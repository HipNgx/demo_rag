"""Chatbot RAG mẫu dùng cho lớp học.

Chạy lần đầu:
    python app.py --rebuild
"""

# ================================================================
# PHẦN 1 — IMPORT VÀ CẤU HÌNH
# Mục đích: nạp thư viện, đọc API key và khai báo các tham số có thể chỉnh.
# ================================================================
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

# Đảm bảo in tiếng Việt không bị lỗi encoding trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Model embedding local (miễn phí, không cần API key)
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


PROJECT_DIR = Path(__file__).resolve().parent

# Load .env sớm để os.getenv() ở module level nhận được giá trị đúng
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(PROJECT_DIR / ".env")
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "student_faq"
CHAT_HISTORY_FILE = PROJECT_DIR / "chat_history.json"

# Có thể đổi model trong file .env mà không phải sửa code.
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

TOP_K = 5                # Lấy 3 đoạn tài liệu gần câu hỏi nhất.
MEMORY_TURNS = 3          # Chỉ giữ 3 lượt hỏi - đáp gần nhất để tránh dài context.
CHUNK_SIZE = 900        # Số ký tự tối đa trong một chunk.
CHUNK_OVERLAP = 200       # Phần gối giữa hai chunk, tránh mất ý ở ranh giới.
MIN_SIMILARITY = 0.3     # Ngưỡng demo; cần tinh chỉnh bằng dữ liệu thực tế.


# ================================================================
# PHẦN 2 — TIỆN ÍCH: ĐỌC FILE VÀ CHIA CHUNK
# Mục đích: biến tài liệu dài thành các đoạn nhỏ để có thể tìm kiếm chính xác.
# Đây là nội dung Document Loading + Chunking (buổi 20).
# ================================================================
def read_text_files(folder: Path) -> list[tuple[str, str]]:
    """Đọc mọi file .txt và trả về [(tên_file, nội_dung), ...]."""
    files = sorted(folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"Không có file .txt trong {folder}. Hãy thêm tài liệu trước khi chạy."
        )

    documents = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((file_path.name, text))
    return documents


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chia text theo ký tự và có overlap; đơn giản, dễ quan sát khi học."""
    # Gộp khoảng trắng thừa để khoảng cách chunk ổn định hơn.
    clean_text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Nếu đã đến cuối thì dừng; nếu không, lùi lại overlap ký tự.
        if end == len(clean_text):
            break
        start = end - overlap

    return chunks


# ================================================================
# PHẦN 3 — EMBEDDING VÀ VECTOR DATABASE
# Mục đích: đổi text thành vector, rồi lưu vector cùng nội dung vào ChromaDB.
# Đây là Embedding/Semantic Search + Vector Database (buổi 19 và 21).
# ================================================================
def create_openai_client() -> OpenAI:
    """Khởi tạo client sau khi kiểm tra API key."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Chưa có OPENAI_API_KEY. Hãy copy .env.example thành .env rồi điền key."
        )
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(base_url=base_url) if base_url else OpenAI()


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Dùng sentence-transformers local để embed text (không cần OpenAI key)."""
    embeddings = _embedding_model.encode(texts).tolist()
    return embeddings


def get_collection(rebuild: bool):
    """Mở database local. rebuild=True sẽ xóa index cũ để lập lại từ đầu."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print("Đã xóa index cũ.")
        except Exception:
            # Lần chạy đầu chưa có collection nên không cần báo lỗi.
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Tài liệu FAQ cho chatbot lớp học"},
    )
    return collection


def index_knowledge_base(collection, client: OpenAI) -> None:
    """Đọc tài liệu -> chunking -> embedding -> lưu vào ChromaDB."""
    if collection.count() > 0:
        print(f"Đang dùng index có sẵn: {collection.count()} chunks.")
        return

    documents = read_text_files(KNOWLEDGE_BASE_DIR)
    all_chunks, all_ids, all_metadatas = [], [], []

    for filename, text in documents:
        category = Path(filename).stem
        chunks = split_into_chunks(text)
        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{Path(filename).stem}_chunk_{index}")
            # Metadata giúp biết câu trả lời lấy từ file, đoạn nào và nhóm (category) nào.
            all_metadatas.append({
                "source": filename,
                "chunk_number": index,
                "category": category,
            })

    print(f"Đang tạo embedding cho {len(all_chunks)} chunks...")
    embeddings = embed_texts(client, all_chunks)
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings,
    )
    print("Đã lập index xong. Lần sau không cần tạo lại trừ khi đổi tài liệu.")


# ================================================================
# PHẦN 4 — RETRIEVAL / SEMANTIC SEARCH
# Mục đích: tìm các chunks có ý nghĩa gần nhất với câu hỏi người dùng.
# Đây là bước R (Retrieval) trong RAG.
# ================================================================
def distance_to_similarity(distance: float) -> float:
    """Đổi distance thành một điểm dễ đọc trong demo; không phải độ chính xác tuyệt đối."""
    return round(1 / (1 + distance), 2)


def retrieve_relevant_chunks(collection, client: OpenAI, question: str,
                             category: str | None = None) -> list[dict]:
    """Embedding câu hỏi rồi truy vấn top-k chunks gần nhất trong vector DB."""
    if collection.count() == 0:
        return []

    question_embedding = embed_texts(client, [question])[0]
    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results": min(TOP_K, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if category:
        query_kwargs["where"] = {"category": category}

    result = collection.query(**query_kwargs)

    retrieved = []
    if not result.get("documents") or not result["documents"][0]:
        return []

    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        retrieved.append(
            {
                "content": document,
                "source": metadata["source"],
                "chunk_number": metadata["chunk_number"],
                "category": metadata.get("category", ""),
                "similarity": distance_to_similarity(distance),
            }
        )
    return retrieved


# ================================================================
# PHẦN 5 — MEMORY VÀ PROMPT
# Mục đích: giữ một ít ngữ cảnh hội thoại, rồi ghép cùng tài liệu đã truy xuất.
# Đây là Memory (buổi 18) và cách chuẩn bị context cho RAG (buổi 22).
# ================================================================
def add_to_memory(history: list[dict], question: str, answer: str) -> None:
    """Lưu lượt hỏi - đáp hiện tại, chỉ giữ số lượt gần nhất theo MEMORY_TURNS."""
    history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])
    del history[:-MEMORY_TURNS * 2]


def save_chat_turn_to_json(question: str, answer: str,
                           chunks: list[dict] | None = None) -> None:
    """Lưu từng lượt hỏi - đáp kèm thời gian và nguồn tham khảo ra file JSON."""
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": item["source"],
                "chunk_number": item["chunk_number"],
                "category": item.get("category", ""),
                "similarity": item["similarity"],
            }
            for item in (chunks or [])
        ],
    }

    history_data = []
    if CHAT_HISTORY_FILE.exists():
        try:
            history_data = json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
            if not isinstance(history_data, list):
                history_data = []
        except Exception:
            history_data = []

    history_data.append(record)
    CHAT_HISTORY_FILE.write_text(
        json.dumps(history_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_history(history: list[dict]) -> str:
    """Biến lịch sử thành text để gửi vào prompt."""
    if not history:
        return "Chưa có lịch sử hội thoại."

    role_name = {"user": "Người dùng", "assistant": "Chatbot"}
    return "\n".join(f"{role_name[item['role']]}: {item['content']}" for item in history)


def build_rag_prompt(question: str, history: list[dict], chunks: list[dict]) -> str:
    """Ghép câu hỏi + memory + các đoạn đã retrieve thành input cho LLM."""
    context = "\n\n".join(
        f"[Nguồn: {item['source']} | đoạn {item['chunk_number']}]\n{item['content']}"
        for item in chunks
    )
    return f"""TÀI LIỆU THAM KHẢO:
{context}

LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{format_history(history)}

CÂU HỎI HIỆN TẠI:
{question}

YÊU CẦU:
- Chỉ trả lời dựa vào TÀI LIỆU THAM KHẢO.
- Nếu tài liệu không có thông tin cần thiết, hãy nói rõ: "Mình chưa tìm thấy thông tin này trong tài liệu hiện có."
- Trả lời ngắn gọn, bằng tiếng Việt, dễ hiểu.
- Không tự bịa quy định, con số hoặc nguồn.
"""


# ================================================================
# PHẦN 6 — GENERATION + STREAMING
# Mục đích: LLM dùng context đã retrieve để soạn câu trả lời và in dần ra màn hình.
# Đây là bước G (Generation) trong RAG và Streaming (buổi 18).
# ================================================================
def stream_answer(client: OpenAI, prompt: str) -> str:
    """Gọi LLM và in từng phần text ngay khi nhận được, thay vì chờ cả câu trả lời."""
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": (
                "Bạn là chatbot hỗ trợ sinh viên. Tuân thủ nghiêm ngặt dữ liệu "
                "được cung cấp trong prompt."
            )},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            answer += delta
    print()
    return answer


# ================================================================
# PHẦN 7 — VÒNG LẶP CHATBOT
# Mục đích: nhận câu hỏi, retrieve, kiểm tra chất lượng, trả lời và cập nhật memory.
# ================================================================
def print_sources(chunks: list[dict]) -> None:
    """Hiện nguồn để người học thấy chatbot đã lấy context ở đâu."""
    print("\nNguồn được tìm thấy:")
    for item in chunks:
        cat_info = f" | nhóm: {item['category']}" if item.get("category") else ""
        print(
            f"- {item['source']}{cat_info} | đoạn {item['chunk_number']} "
            f"| độ gần đúng: {item['similarity']}"
        )


def run_chatbot(collection, client: OpenAI, category: str | None = None) -> None:
    """Chạy chatbot terminal cho tới khi người dùng gõ /exit."""
    history: list[dict] = []
    current_category = category
    print("\nChatbot đã sẵn sàng. Lệnh: /history, /reset, /category <tên|all>, /categories, /exit")
    if current_category:
        print(f"-> Đang lọc theo nhóm: '{current_category}'")

    while True:
        prompt_prefix = f"\nBạn[{current_category}]: " if current_category else "\nBạn: "
        question = input(prompt_prefix).strip()
        if not question:
            continue
        if question.lower() == "/exit":
            print("Tạm biệt!")
            break
        if question.lower() == "/reset":
            history.clear()
            print("Đã xóa memory hội thoại.")
            continue
        if question.lower() == "/history":
            print("\n" + format_history(history))
            continue
        if question.lower() == "/categories":
            try:
                data = collection.get(include=["metadatas"])
                cats = sorted({m.get("category") for m in data.get("metadatas", []) if m and m.get("category")})
                print(f"Các category hiện có trong database: {', '.join(cats) if cats else 'Chưa có thông tin category (hãy chạy --rebuild).'}")
            except Exception as e:
                print(f"Không thể đọc category: {e}")
            continue
        if question.lower().startswith("/category"):
            parts = question.split(maxsplit=1)
            if len(parts) == 1:
                print(f"Nhóm đang lọc: {current_category or 'Tất cả (không lọc)'}")
                print("Dùng '/category <tên>' để lọc hoặc '/category all' để xem toàn bộ.")
            else:
                target_cat = parts[1].strip()
                if target_cat.lower() in ("all", "off", "none", "*"):
                    current_category = None
                    print("Đã tắt bộ lọc category. Chatbot sẽ tìm kiếm trên toàn bộ tài liệu.")
                else:
                    current_category = target_cat
                    print(f"Đã chuyển bộ lọc sang nhóm: '{current_category}'.")
            continue

        chunks = retrieve_relevant_chunks(collection, client, question, category=current_category)

        # Guardrail đơn giản: không đủ liên quan thì không gọi LLM để tránh trả lời bừa.
        best_similarity = max((item["similarity"] for item in chunks), default=0)
        if best_similarity < MIN_SIMILARITY:
            answer = "Mình chưa tìm thấy thông tin này trong tài liệu hiện có."
            print(f"\nChatbot: {answer}")
            add_to_memory(history, question, answer)
            save_chat_turn_to_json(question, answer, [])
            continue

        print_sources(chunks)
        prompt = build_rag_prompt(question, history, chunks)
        print("\nChatbot: ", end="")
        answer = stream_answer(client, prompt)
        add_to_memory(history, question, answer)
        save_chat_turn_to_json(question, answer, chunks)


def main() -> None:
    """Điểm bắt đầu chương trình; --rebuild dùng khi thay đổi file tài liệu."""
    parser = argparse.ArgumentParser(description="Chatbot RAG mẫu cho lớp học")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Xóa index cũ và tạo lại embedding từ knowledge_base",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Chỉ tìm kiếm trong nhóm tài liệu (category) nhất định",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    client = create_openai_client()
    collection = get_collection(rebuild=args.rebuild)
    index_knowledge_base(collection, client)
    run_chatbot(collection, client, category=args.category)


if __name__ == "__main__":
    main()
