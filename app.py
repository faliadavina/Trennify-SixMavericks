import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import cv2
import random

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("index.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r




@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    
    total = 28
    baris = 7
    kolom = 4
    
    image_filters = image_processing.filtering("static/img/img_now.jpg")
    folder_path = "static/img/filter"
    file_paths = []
    for filename in os.listdir(folder_path):
        file_paths.append(os.path.join(folder_path, filename))
        
    random_value = random.sample(range(1, total + 1), total)
        
    return render_template("flip_memory.html", image_filters=image_filters, file_paths=["img/img_normal.jpg"], total=total, random_value=random_value, baris=baris, kolom=kolom)
    
@app.route("/puzzle", methods=["POST"])
@nocache
def puzzle():
    number = int(request.form['nilai_fixed'])
    total = 0
    image_processing.crop("static/img/img_now.jpg", number, number)
    folder_path = "static/img/puzzle"
    file_paths = []
    for filename in os.listdir(folder_path):
        file_paths.append(os.path.join(folder_path, filename))
        
    total = total * total
        
    return render_template("puzzle.html", file_paths=["img/img_normal.jpg"], total=total, baris_kolom=number)

@app.route("/random_puzzle", methods=["POST"])
@nocache
def random_puzzle():
    number = int(request.form['nilai_random'])
    total = 0
    image_processing.crop_and_shuffle("static/img/img_now.jpg", number, number)
    folder_path = "static/img/puzzle"
    file_paths = []
    for filename in os.listdir(folder_path):
        file_paths.append(os.path.join(folder_path, filename))
        
    total = number * number
        
    random_value = random.sample(range(1, total + 1), total)
        
    return render_template("random_puzzle.html", file_paths=["img/img_normal.jpg"], total=total, random_value=random_value, baris_kolom=number)

@app.route("/rgb", methods=["POST"])
@nocache
def rgb():
    img = cv2.imread('image_aini.jpg')
    r, g, b = cv2.split(img)
    print("Red : ", r)
    print("Green : ", g)
    print("Blue : ", b)
        
    return render_template("rgb.html", r=r, g=g, b=b)

@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    #return render_template("uploaded_2.html", file_path="img/img_normal.jpg")
    return histogram_rgb()


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()

@app.route("/identity_kernel", methods=["POST"])
@nocache
def identity_kernel():
    image_processing.identity_kernel()
    return histogram_rgb()

@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    return histogram_rgb()

@app.route("/lowpass_blur", methods=["POST"])
@nocache
def lowpass_blur():
    size = int(request.form['kernel'])
    image_processing.lowpass_blur(size)
    return histogram_rgb()

@app.route("/gaussian_blur", methods=["POST"])
@nocache
def gaussian_blur():
    size = int(request.form['kernel'])
    image_processing.gaussian_blur(size)
    return histogram_rgb()

@app.route("/median_blur", methods=["POST"])
@nocache
def median_blur():
    size = int(request.form['kernel'])
    image_processing.median_blur(size)
    return histogram_rgb()

@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()

@app.route("/bilateral", methods=["POST"])
@nocache
def bilateral():
    size = int(request.form['kernel'])
    image_processing.bilateral(size)
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()

@app.route("/zero_padding", methods=["POST"])
@nocache
def zero_padding():
    size = int(request.form['padding'])
    image_processing.zero_padding(size)
    return histogram_rgb()

@app.route("/highpass", methods=["POST"])
@nocache
def highpass():
    size = int(request.form['kernel'])
    image_processing.highpass(size)
    return histogram_rgb()

@app.route("/bandpass", methods=["POST"])
@nocache
def bandpass():
    size = int(request.form['kernel'])
    image_processing.bandpass(size)
    return histogram_rgb()

@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    img = Image.open("static/img/img_now.jpg")
    lebar, tinggi = (img).size
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("uploaded_2.html", file_paths=["img/img_normal.jpg","img/img_now.jpg","img/grey_histogram.jpg"], lebar=lebar, tinggi=tinggi)
    else:
        return render_template("uploaded_2.html", file_paths=["img/img_normal.jpg","img/img_now.jpg","img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"], lebar=lebar, tinggi=tinggi)


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
