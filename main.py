import os
import io
import json
import torch

from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # 解决跨域问题

weights_path = "./MobileNetV2(flower).pth"
class_json_path = "./class_indices.json"

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
# model = eval(num_classes=5)
# # load model weights
# model.load_state_dict(torch.load(weights_path, map_location=device))
# model.to(device)
# model.eval()
# 
# # load class info
# json_file = open(class_json_path, 'rb')
# class_indict = json.load(json_file)


def get_prediction(image_bytes):
    try:
        tensor = ""
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)










import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf
# import nltk
# nltk.download('cmudict')
# 我说明一下tts的逻辑:
# 这个项目首先需要 文字转编码, 用 processor
# 然后 编码啊转mel序列. 用fastspeech2
# 最后mel序列转语音 用model = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-baker-ch")


from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")
from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-baker-ch")








@app.route("/", methods=["GET", "POST"])
def root():
    text = "但是很有可能，实际操作中会报 Connection refused 或 ssl 验证等错误"
    d = bytes.decode(request.data)
    data = json.loads(d)
    input_ids = processor.text_to_sequence(data['text'], inference=True)

    mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )

    audios = model.inference(mel_after)[0, :, 0]
    print(1)

    # save to file
    import time
    a = time.time()
    import time
    now = time.localtime()
    nowt = time.strftime("%Y-%m-%d-%H-%M-%S", now)  # 这一步就是对时间进行格式化
    print(nowt)
    sf.write(f'./{nowt}.wav', audios, 22050, 'PCM_24')
    name=f'/{nowt}.wav'





    return jsonify('/home/troila/zhangbo/123'+name)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)




