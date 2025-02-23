# JDictionaryServer

1. Tải xuống cơ sở dữ liệu và bỏ vào folder JDictionaryserver: 
link: https://drive.google.com/file/d/12y9OiMiFfe0MkIjZD_VHjp69sUapFEnw/view?usp=sharing

2. Tạo môi trường ảo cho python
python -m venv venv

3. Kích hoạt môi trường
```
.\venv\Scripts\activate
```

4. Tải xuống các thư viện cần thiết
```
pip install -r requirements.txt
```
hoặc
```
pip install fastapi sqlite3 fastapi[all] torch torchvision Pillow numpy opencv-python
```

5. Chạy server 
```
uvicorn main:app --host 0.0.0.0 --port 8000 
```