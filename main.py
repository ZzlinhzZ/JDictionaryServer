from fastapi import FastAPI, Header, Depends, HTTPException, status
from fastapi.security import HTTPBearer
import sqlite3
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import numpy as np
import cv2
import os

from pydantic import BaseModel
from passlib.context import CryptContext
import secrets

security = HTTPBearer()

# Khởi tạo model 
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


# Load label mapping và chỉ lấy ký tự kanji
with open("kanji_labels.txt", "r", encoding="utf-8") as f:
    kanji_labels = [line.split(maxsplit=1)[-1] for line in f.read().splitlines()]

# Thư mục lưu ảnh đầu vào để debug
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_bytes):
    # Mở ảnh và chuyển thành grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    #  Lưu ảnh gốc để kiểm tra
    image.save(os.path.join(UPLOAD_FOLDER, "original_received.png"))

    # Chuyển sang numpy để xử lý với OpenCV
    img_np = np.array(image)

    # Binarization (Nhị phân hóa ảnh)
    _, img_bin = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)

    # Kiểm tra nền trắng hay đen, nếu nền trắng thì đảo ngược
    white_pixel_count = np.sum(img_bin == 255)
    black_pixel_count = np.sum(img_bin == 0)
    if white_pixel_count > black_pixel_count:
        img_bin = cv2.bitwise_not(img_bin)

    #  Resize về 64x64
    img_resized = cv2.resize(img_bin, (64, 64), interpolation=cv2.INTER_AREA)

    #  Lưu ảnh sau khi tiền xử lý để kiểm tra
    Image.fromarray(img_resized).save(os.path.join(UPLOAD_FOLDER, "processed_image.png"))

    # Chuyển sang tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa dữ liệu
    ])
    return transform(Image.fromarray(img_resized)).unsqueeze(0)

def get_current_user(token: str = Depends(security)):
    conn = get_db_connection()
    user = conn.execute(
        "SELECT id FROM users WHERE token = ?",
        (token.credentials,)
    ).fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user

@app.post("/recognize-kanji")
async def recognize_kanji(image: dict):
    try:
        # Giải mã ảnh từ base64
        image_bytes = base64.b64decode(image["image"])
        input_tensor = preprocess_image(image_bytes)

        # Dự đoán với model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top10 = torch.topk(probabilities, 10)

        # Lấy top 10 Kanji dự đoán
        top_kanji = [kanji_labels[idx] for idx in top10.indices.tolist()]
        top_probs = top10.values.tolist()

        # 🖨 Ghi log dự đoán
        print("Top 10 Kanji dự đoán:", top_kanji)
        print("Xác suất:", top_probs)

        return JSONResponse(content={"predictions": top_kanji, "probabilities": top_probs})

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
def save_kanji(data: dict, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO saved_kanji (user_id, kanji, pronounced, meaning) VALUES (?, ?, ?, ?)",
            (current_user["id"], data["kanji"], data["pronounced"], data["meaning"]),
        )
        conn.commit()
        return {"message": "Kanji saved successfully"}
    except sqlite3.IntegrityError:
        return JSONResponse(
            content={"error": "Kanji already exists"},
            status_code=400
        )
    finally:
        conn.close()

@app.delete("/delete_kanji/{kanji}")
def delete_kanji(kanji: str, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    conn.execute(
        "DELETE FROM saved_kanji WHERE kanji = ? AND user_id = ?",
        (kanji, current_user["id"])
    )
    conn.commit()
    conn.close()
    return {"message": "Kanji removed"}
    

@app.get("/saved_kanji")
def get_saved_kanji(current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    kanji = conn.execute(
        "SELECT * FROM saved_kanji WHERE user_id = ?",
        (current_user["id"],)
    ).fetchall()
    conn.close()
    return [dict(k) for k in kanji]



#######################################################################

# Thêm class cho dữ liệu đăng nhập/đăng ký
class UserCreate(BaseModel):
    username: str
    password: str
    email: str

class UserLogin(BaseModel):
    username: str
    password: str

# Khởi tạo password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Tạo bảng users trong SQLite (thêm vào đầu file)
# def init_db():
#     conn = sqlite3.connect("dictionary.db")

#     conn.execute("DROP TABLE IF EXISTS saved_kanji")  # Xóa bảng nếu tồn tại
#     conn.commit()
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS saved_kanji (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id INTEGER NOT NULL,
#             kanji TEXT NOT NULL,
#             pronounced TEXT NOT NULL,
#             meaning TEXT NOT NULL,
#             FOREIGN KEY(user_id) REFERENCES users(id)
#         )
#     """)
#     conn.commit()
#     conn.close()

# # Gọi hàm init_db khi khởi động
# init_db()


# Thêm các endpoints mới
@app.post("/register")
async def register(user: UserCreate):
    conn = get_db_connection()
    hashed_password = pwd_context.hash(user.password)
    token = secrets.token_hex(16)
    
    try:
        conn.execute(
            "INSERT INTO users (username, email, password, token) VALUES (?, ?, ?, ?)",
            (user.username, user.email, hashed_password, token)
        )
        conn.commit()
        return {"message": "User created", "token": token}
    except sqlite3.IntegrityError:
        return JSONResponse(
            content={"error": "Username or email already exists"},
            status_code=400
        )
    finally:
        conn.close()

@app.post("/login")
async def login(user: UserLogin):
    conn = get_db_connection()
    db_user = conn.execute(
        "SELECT * FROM users WHERE username = ?", 
        (user.username,)
    ).fetchone()
    
    if not db_user or not pwd_context.verify(user.password, db_user["password"]):
        return JSONResponse(
            content={"error": "Invalid credentials"},
            status_code=401
        )
    
    new_token = secrets.token_hex(16)
    conn.execute(
        "UPDATE users SET token = ? WHERE id = ?",
        (new_token, db_user["id"])
    )
    conn.commit()
    conn.close()
    
    return {"message": "Login successful", "token": new_token}

@app.post("/logout")
async def logout(authorization:  str = Header(None)):
    if not authorization: 
        return JSONResponse(
            content={"error": "Token missing"},
            status_code=401
        )
        # Extract token từ header (Bearer <token>)
    try:
        token = authorization.split(" ")[1]
    except IndexError:
        return JSONResponse(
            content={"error": "Invalid token format"},
            status_code=401
        )
    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET token = NULL WHERE token = ?",
        (token,)
    )
    conn.commit()
    conn.close()
    return {"message": "Logged out successfully"}

@app.get("/me")
async def get_current_user(token: str = Header(None)):
    if not token:
        return JSONResponse(
            content={"error": "Not authenticated"},
            status_code=401
        )
    
    conn = get_db_connection()
    user = conn.execute(
        "SELECT username, email FROM users WHERE token = ?",
        (token,)
    ).fetchone()
    conn.close()
    
    if not user:
        return JSONResponse(
            content={"error": "Invalid token"},
            status_code=401
        )
    
    return {"username": user["username"], "email": user["email"]}


