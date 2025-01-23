import base64
import gc
import io

import numpy as np
from fastapi import FastAPI
from paddleocr import PaddleOCR
from PIL import Image
from pydantic import BaseModel

app = FastAPI()
ocr_module = PaddleOCR(use_angle_cls=True, lang="en")


def text_cvt_orc_format_easyocr(easyocr_result):
    """
    Convert EasyOCR results into the required format.
    """
    texts = []
    for i, line in enumerate(easyocr_result):
        bbox, content, _ = line
        location = {
            "left": int(bbox[0][0]),
            "top": int(bbox[0][1]),
            "right": int(bbox[2][0]),
            "bottom": int(bbox[2][1]),
        }
        texts.append((i, content, location))
    return texts

import easyocr
import numpy as np

easyocr_reader = easyocr.Reader(["en"])

def ocr_results_easyocr(screenshot):
    """
    Perform OCR using EasyOCR and return results.
    """
    screenshot_img = Image.open(io.BytesIO(screenshot))
    if screenshot_img.mode != "RGB":
        screenshot_img = screenshot_img.convert("RGB")
    easyocr_result = easyocr_reader.readtext(np.array(screenshot_img))
    return text_cvt_orc_format_easyocr(easyocr_result)


class ImageData(BaseModel):
    img_bytes: bytes


def text_cvt_orc_format_paddle(paddle_result):
    texts = []
    print("paddle_result: ", paddle_result)
    for i, line in enumerate(paddle_result[0]):
        points = np.array(line[0])
        print("points: ", points)
        location = {
            "left": int(min(points[:, 0])),
            "top": int(min(points[:, 1])),
            "right": int(max(points[:, 0])),
            "bottom": int(max(points[:, 1])),
        }
        print("location: ", location)
        content = line[1][0]
        texts.append((i, content, location))
    return texts


def ocr_results(screenshot):
    screenshot_img = Image.open(io.BytesIO(screenshot))
    result = ocr_module.ocr(np.array(screenshot_img), cls=True)
    return text_cvt_orc_format_paddle(result)

print("ocr_module: ", easyocr_reader)

@app.post("/ocr/")
async def read_image(image_data: ImageData):
    image_bytes = base64.b64decode(image_data.img_bytes)
    results = ocr_results_easyocr(image_bytes)
    print("Received easyOCR response.")
    # Explicitly delete unused variables and run garbage collector
    del image_bytes
    gc.collect()

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
