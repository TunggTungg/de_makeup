from flask import Flask, render_template, request
from utlis import grpc_infer
import numpy as np
from PIL import Image
import base64, io

# Init Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def Home():
	return render_template("index.html", input_img="static/demo/demo_input.jpg", output_img="static/demo/demo_output.jpg")

@app.route("/generate", methods = ["POST"])
def generate():
	file = request.files['file']
	if file and allowed_file(file.filename):
		# Save the file to a desired location
		input_image_data = file.read()
		input_image_base64 = base64.b64encode(input_image_data).decode('utf-8')
		generated = grpc_infer(input_image_data)
		generated = Image.fromarray((generated[0] * 255).astype(np.uint8))

		img_bytes = io.BytesIO()
		generated.save(img_bytes, format='PNG')
		img_bytes.seek(0)

		output_image_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
	return render_template("index.html", input_img=f'data:image/jpeg;base64,{input_image_base64}', output_img=f'data:image/jpeg;base64,{output_image_base64}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)