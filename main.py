from fastapi import FastAPI, UploadFile, File
import sqlite3
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io

# Khởi tạo model (bạn có thể import từ file khác nếu cần)
class KanjiCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(KanjiCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3832  
model = KanjiCNN(num_classes)

# Load trọng số vào model
state_dict = torch.load("kanji_recognition.pth", map_location=device)
model.load_state_dict(state_dict)  
model.to(device)
model.eval()  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc chỉ định IP của ứng dụng Flutter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_db_connection():
    conn = sqlite3.connect("dictionary.db")
    conn.row_factory = sqlite3.Row
    return conn


# Load label mapping
with open("kanji_labels.txt", "r", encoding="utf-8") as f:
    kanji_labels = f.read().splitlines()

# Hàm tiền xử lý ảnh
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

@app.post("/recognize-kanji")
async def recognize_kanji(image: dict):
    try:
        # Giải mã ảnh từ base64
        image_bytes = base64.b64decode(image["image"])
        input_tensor = preprocess_image(image_bytes)

        # Dự đoán
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top10 = torch.topk(probabilities, 10)

        # Lấy top 10 kanji dự đoán
        top_kanji = [kanji_labels[idx] for idx in top10.indices.tolist()]
        return JSONResponse(content={"predictions": top_kanji})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/words")
def get_words(search: str = ""):
    conn = get_db_connection()
    if search:
        words = conn.execute("""
            SELECT * FROM words_Test 
            WHERE written = ? 
            UNION ALL
            SELECT * FROM words_Test 
            WHERE written LIKE ? AND written != ?
        """, (search, '%' + search + '%', search)).fetchall()
    else:
        words = conn.execute("SELECT * FROM words_Test").fetchall()
    
    conn.close()
    return [dict(word) for word in words]
# WHERE written = ? Lấy những từ có khớp chính xác với từ tìm kiếm.
# UNION ALL
# Ghép thêm kết quả có chứa từ tìm kiếm nhưng không phải khớp chính xác

@app.get("/kanji")
def get_kanji():
    conn = get_db_connection()
    kanji = conn.execute("SELECT * FROM kanji").fetchall()
    conn.close()
    return [dict(k) for k in kanji]

@app.post("/save_kanji")
def save_kanji(data: dict):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO saved_kanji (kanji, pronounced, meaning) VALUES (?, ?, ?)",
        (data["kanji"], data["pronounced"], data["meaning"]),
    )
    conn.commit()
    conn.close()
    return {"message": "Kanji saved successfully"}

@app.delete("/delete_kanji/{kanji}")
def delete_kanji(kanji: str):
    conn = get_db_connection()
    conn.execute("DELETE FROM saved_kanji WHERE kanji = ?", (kanji,))
    conn.commit()
    conn.close()
    return {"message": "Kanji removed"}
    
@app.get("/saved_kanji")
def get_saved_kanji():
    conn = get_db_connection()
    kanji = conn.execute("SELECT * FROM saved_kanji").fetchall()
    conn.close()
    return [dict(k) for k in kanji]