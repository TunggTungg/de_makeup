
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from model import load_model
from image_processing import load_image_val
import matplotlib.pyplot as plt 
from PIL import Image
from pyngrok import ngrok
from flask_ngrok import run_with_ngrok

# Load Model
model = load_model("res")


# Khởi tạo Flask
app = Flask(__name__)
run_with_ngrok(app)
app.config['UPLOAD_FOLDER'] = "static"

# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # # Nếu là POST (gửi file)
    if request.method == "POST":
         # try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
                image.save(path_to_save)
                size = cv2.imread(path_to_save).shape
                input_img = np.expand_dims(load_image_val(path_to_save),axis=0)
                output_img = model.predict(input_img)[0]
                plt.imsave("static/output.jpg", output_img)
                # os.remove(path_to_save)
                return render_template("index.html",  msg="Tải file lên thành công", input_img=path_to_save, output_img="static/output.jpg")
            
    #         else:
    #             # Nếu không có file thì yêu cầu tải file
    #             return render_template('index.html', msg='Hãy chọn file để tải lên')

         # except Exception as ex:
         #    # Nếu lỗi thì thông báo
         #    print(ex)
         #    return render_template('index.html', msg='Error')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


def clean_up():
    os.remove("static/output.jpg")
    os.remove("static/input.jpg")
    print("Hello")
    
if __name__ == '__main__':
    try:
        app.run()
    except KeyboardInterrupt:
        clean_up()