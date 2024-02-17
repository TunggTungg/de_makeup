from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils import grpc_infer
import numpy as np
from PIL import Image
import base64, io
import imghdr

# Init FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
def is_valid_image_extension(filename: str) -> bool:
	ext = filename.split(".")[-1]
	return ext.lower() in ALLOWED_IMAGE_EXTENSIONS

def is_valid_image_content(file: bytes) -> bool:
	return imghdr.what(None, h=file) in ALLOWED_IMAGE_EXTENSIONS

@app.get("/")
async def homepage(request: Request):
	return templates.TemplateResponse("index.html", {"request": request, "input_img":"static/demo/demo_input.jpg", \
													 "output_img":"static/demo/demo_output.jpg"})

@app.post("/generate")
async def generate(request: Request, file: UploadFile = File(...)):
	   
	# Check if the filename has a valid image extension
	if not is_valid_image_extension(file.filename):
		raise HTTPException(status_code=400, detail="Only image files are allowed")

	# Read the content of the uploaded file
	input_image_data = await file.read()

	# Check if the content is a valid image
	if not is_valid_image_content(input_image_data):
		raise HTTPException(status_code=400, detail="Invalid image file")
		
	input_image_base64 = base64.b64encode(input_image_data).decode('utf-8')
	generated = grpc_infer(input_image_data)
	generated = Image.fromarray((generated * 255).astype(np.uint8))

	img_bytes = io.BytesIO()
	generated.save(img_bytes, format='PNG')
	img_bytes.seek(0)
	output_image_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

	return templates.TemplateResponse("index.html", {"request": request, "input_img":f'data:image/jpeg;base64,{input_image_base64}', \
												  	 "output_img":f'data:image/jpeg;base64,{output_image_base64}'})
