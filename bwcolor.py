import numpy as np
import cv2 as cv
from flask import Flask, render_template, request, redirect, url_for, send_file
import os

app = Flask(__name__)
STATIC_FOLDER = 'static/uploads'
os.makedirs(STATIC_FOLDER, exist_ok=True)
W_in, H_in = 224, 224  # Model input dimensions

# Load pre-trained model and kernel
prototxt = 'models/colorization_deploy_v2.prototxt'
caffemodel = 'models/colorization_release_v2.caffemodel'
kernel = 'resources/pts_in_hull.npy'

net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
pts_in_hull = np.load(kernel).transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]


@app.route('/')
def index():
    return render_template("first.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")



@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(STATIC_FOLDER, file.filename)
    file.save(filepath)

    # Process the image
    img = cv.imread(filepath)
    if img is None:
        return "Error reading the image file", 400

    # Coloring logic (unchanged from original script)
    img_rgb = (img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0]
    H_orig, W_orig = img_rgb.shape[:2]

    img_rs = cv.resize(img_rgb, (W_in, H_in))
    img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:, :, 0]
    img_l_rs -= 50

    net.setInput(cv.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    # Save both the original and colorized images
    original_path = os.path.join(STATIC_FOLDER, 'original.png')
    colorized_path = os.path.join(STATIC_FOLDER, 'colorized.png')
    cv.imwrite(original_path, img)
    cv.imwrite(colorized_path, (img_bgr_out * 255).astype(np.uint8))

    return render_template(
        'result.html',
        original_image=url_for('static', filename=f'uploads/original.png'),
        colorized_image=url_for('static', filename=f'uploads/colorized.png')
    )


@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(STATIC_FOLDER, filename)
    return send_file(filepath, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
