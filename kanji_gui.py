import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

num_classes = 3832  
model = KanjiCNN(num_classes)
model.load_state_dict(torch.load("kanji_recognition.pth", map_location=device))
model.to(device)
model.eval()

def preprocess_image(image_path):
    """
    Tiền xử lý ảnh: Resize, chuyển ảnh về nền đen chữ trắng
    """
    # Load ảnh bằng OpenCV, chuyển về grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize về 64x64
    img = cv2.resize(img, (64, 64))

    # Binarization (nhị phân hóa ảnh)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Kiểm tra màu nền
    white_pixel_count = np.sum(img == 255)
    black_pixel_count = np.sum(img == 0)

    # Nếu nền ảnh là trắng, thì đảo ngược ảnh
    if white_pixel_count > black_pixel_count:
        img = cv2.bitwise_not(img)

    # Chuyển đổi sang dạng Tensor cho model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    processed_image = transform(Image.fromarray(img)).unsqueeze(0)
    
    return processed_image.to(device), Image.fromarray(img)

def load_kanji_labels():
    label_dict = {}
    with open("kanji_labels.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            label_dict[int(parts[0])] = parts[1]
    return label_dict

kanji_labels = load_kanji_labels()

def predict_top_k(image_path, k=10):
    image_tensor, processed_img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k)

    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    results = []
    for i in range(k):
        kanji = kanji_labels.get(top_indices[i], "Unknown")
        results.append(f"{i+1}. {kanji} - Xác suất: {top_probs[i]*100:.2f}%")

    return results, processed_img

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        results, processed_img = predict_top_k(file_path)
        
        # Hiển thị ảnh gốc
        original_image = Image.open(file_path)
        original_image.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(original_image)
        lbl_original.config(image=img_tk)
        lbl_original.image = img_tk

        # Hiển thị ảnh đã tiền xử lý
        processed_img.thumbnail((200, 200))
        processed_tk = ImageTk.PhotoImage(processed_img)
        lbl_processed.config(image=processed_tk)
        lbl_processed.image = processed_tk

        # Cập nhật kết quả dự đoán
        result_text.set("\n".join(results))

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Kanji Recognition")

# Nút chọn ảnh
btn_select = tk.Button(root, text="Chọn ảnh", command=choose_file)
btn_select.pack()

# Hiển thị ảnh gốc và ảnh tiền xử lý
frame_images = tk.Frame(root)
frame_images.pack()

lbl_original = tk.Label(frame_images, text="Ảnh gốc")
lbl_original.pack(side="left", padx=10)

lbl_processed = tk.Label(frame_images, text="Ảnh đã xử lý")
lbl_processed.pack(side="right", padx=10)

# Hiển thị kết quả dự đoán
result_text = tk.StringVar()
lbl_result = tk.Label(root, textvariable=result_text, justify="left", font=("Arial", 12))
lbl_result.pack()

root.mainloop()

