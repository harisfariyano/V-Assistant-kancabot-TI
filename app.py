from flask import Flask, render_template, request, jsonify
import chat
from flask_cors import CORS
from chat import get_response
from flask_mysqldb import MySQL
import datetime
import pickle
import numpy as np
import cv2
import base64
import os
import re
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.image as mpim

from werkzeug.utils import secure_filename
from PIL import Image as im
from tensorflow.keras.utils import load_img
from IPython.display import display, Javascript
from IPython.display import Image
from keras.utils import load_img, img_to_array
from flask import Flask, flash, render_template, url_for, request, jsonify, session, redirect

from recomendation import make_model



app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 2024 * 2024 #ukuran maksimal inputan gambar dalam bentuk byte
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG','.PNG','.png','.jpeg','.JPEG'] #mengatur ekstensi pada gambar yang di upload
app.config['UPLOAD_FOLDER_CLASIFICATION']        = './static/upload/clasification/' #lokasi gambar yang diupload dari user/inputan
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])  #filter ekstensi yang diperbolehkan untuk diinput/upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask'
mysql = MySQL(app)


# Creating a connection cursor
#MODEL IMG menyimpan code modeling dengan menggunakan fungsi make_model yang disimpan di rekomendation.py
model_gender = make_model()

#membuat class get
@app.route("/")
def main():
    return render_template("index.html")
@app.get("/home")
def home_get():
    return render_template("index.html")
@app.get("/blog1")
def blog1():
    return render_template("blog1.html")
@app.get("/blog2")
def blog2():
    return render_template("blog2.html")
@app.get("/blog3")
def blog3():
    return render_template("blog3.html")
@app.get("/profil")
def profil_get():
    return render_template("profil.html")
@app.get("/fasilitas")
def fasilitas_get():
    return render_template("fasilitas.html")
@app.get("/link")
def link_get():
    return render_template("link.html")
@app.get("/camera")
def camera():
    return render_template("camera.html")
@app.get("/classification")
def klasifikasi():
    return render_template("classification.html")


#membuat api post(web service post)
@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    dt = datetime.datetime.now()
    print(dt)
    #TODO: periksa apakah teks valid
    response = get_response(text)
    message = {"answer": response}
    cursor = mysql.connection.cursor()
    cursor.execute('''INSERT INTO test(question,answer,timestamp) VALUES(%s,%s,%s)''',(text,response,dt))
    mysql.connection.commit()
    cursor.close()
    return jsonify(message)

@app.post("/clasification_post")
def clasification_post():
     #simpan nilai default hasil_prediksi dan gambar_prediksi
    hasil_prediksi  = '(none)'
    gambar_prediksi = '(none)'

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['file']
    
    #mendapakan waktu sekarang menggunakan library datetime
    curent_time = datetime.datetime.now()
    #mengubah curenta_time menjadi timestamp
    timestamp = curent_time.timestamp()

    #membuat variable filename isinya timstamp dirubah jadi string terus direplace misal terdapat . pada timestamp
    #filename untuk nama file yang telah diupload
    filename = str(timestamp).replace(".", "") + '.jpg'

    # Simpan Gambar
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER_CLASIFICATION'], filename))
    



    #Training Model
    # predicting images
    path = os.path.join(app.config['UPLOAD_FOLDER_CLASIFICATION'], filename)

    #load gambar yang sudah diupload dan sisesuikan ukurannya
    img = load_img(path, target_size=(224,224))

    #ubah gambar ke dalam array
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
 
    images = np.vstack([x])

    #Prediksi gambar berdasarkan model
    classes = model_gender.predict(images, batch_size=50)
    classes = np.argmax(classes)
  
    #Klasifikasi model
    if classes==0:
        hasil = 'Perempuan'
    elif classes==1:
        hasil = 'Laki-laki'
    else:
        hasil = ('Tidak menemukan jenis kelamin')

    #hasil prediksi disimpan 
    hasil_prediksi = classes

    cursor = mysql.connection.cursor()
    cursor.execute('''INSERT INTO image (foto, nama, hasil) VALUES (%s,%s,%s)''',(filename, request.form['nama'], hasil))
    
    mysql.connection.commit()
    cursor.close()
    

    return redirect(url_for('clasification'))

@app.get('/clasification')
def clasification():
    cursor = mysql.connection.cursor()
    cursor.execute('''SELECT * FROM  image''')
    result = cursor.fetchall()
    cursor.close()
    return render_template("clasification.html", result=result)

# @app.post('/upload')
# def upload():
#     file = request.files['image']

#     file.save("./images/gambarupload.png")
#     return "file berhasil di uplod"

#membuat api history(web service get)
@app.get("/history")
def history():
    #TODO: 
    cursor = mysql.connection.cursor()
    cursor.execute('''SELECT * FROM test  ''')
    test = cursor.fetchall()
    mysql.connection.commit()
    cursor.close()
    return jsonify(test)




# def set_up(speed=150):
#     '''
#     Set up engine configuration
#     '''
#     engine = pyttsx3.init()
#     engine.setProperty('rate', speed)
#     engine.setProperty('volume', 5.0)

#     return engine

# @app.post("/text-to-speech")
# def text_to_speech():
#     '''
#     This endpoint will help to convert text to speech real-time
#     '''
    
#     if not request.json:
#             return jsonify({"message": "Invalid JSON Format", "type": "error"}), 400
#     try:
#             text = request.json['text']
#             text = re.sub(r'[^\w\s]', '', text)
#             text = text.replace(" ", "")
#             speed = request.json['speed']
#             engine = set_up(speed)
#             engine.say(text)
#             engine.runAndWait()

#     except Exception as e:
#             return jsonify({"message": "An error occured when processing the text.", "type": "error"}), 500
#     return jsonify({"type": "success", "message": "You have successfully process the text."}),200



if __name__=="__main__":
    # app.run(debug=True)
    model_gender.load_weights('.\model\gender_model.h5') 
    app.run(host='0.0.0.0', debug=True)
    # from waitress import serve
    # serve(app, host="192.168.100.138", port=5000, debug=True)