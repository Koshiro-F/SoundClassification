from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from typing import List
import os
import uvicorn

# # TensorFlow Liteモデルの読み込みと準備
# MODEL_PATH = "./model/a-nn-ikami-mikubo-fujikawa-model-xy-ver2.tflite"

# # TFLiteモデルの読み込み
# interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# FastAPIのインスタンスを作成
app = FastAPI()

# CORSの設定を追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて許可するオリジンを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SpectrumData(BaseModel):
    sampleRate: int
    spectrum: list[float]  # Python3.9+ の場合: list[float] / それ以前は List[float] など



@app.post("/fft")
async def classify_fft(data: SpectrumData):
    """
    Unityから送られてきたスペクトラムデータを受け取り、
    TFLiteモデルで推論した結果を返すエンドポイント。
    """
    # 受け取ったスペクトラムをNumPy配列に変換
    input_data = np.array(data.spectrum, dtype=np.float32)

    # モデル入力の形状に合わせてリシェイプ（例: [1, 8192]など）
    # スペクトラム長に合わせて可変にする場合: reshape((1, -1))
    input_data = input_data.reshape((1, -1))

    # TFLiteモデルに入力データをセット
    # interpreter.set_tensor(input_details[0]['index'], input_data)

    # # 推論実行
    # interpreter.invoke()

    # # モデルの出力を取得
    # output_data = interpreter.get_tensor(output_details[0]['index'])  # shape例: [1, num_classes]

    # # 分類モデルの想定であれば argmax を取る
    # predicted_class = np.argmax(output_data, axis=1)[0]

    # 例: 返却データをJSON形式で返す（クラスインデックスを返却）
    return {"predicted_class": 1}



@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    # ファイル名が空でないかチェック
    if not image.filename:
        raise HTTPException(status_code=400, detail="ファイルが選択されていません")
    
    # ファイルの保存先を指定
    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    
    # ファイルを保存
    content = await image.read()
    with open(save_path, "wb") as f:
        f.write(content)
    
    return JSONResponse(content={"message": "ファイルが正常にアップロードされました"}, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)