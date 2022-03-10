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
model = eval(num_classes=5)
# load model weights
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)


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


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




