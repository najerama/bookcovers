import config
import cv2
from flask import Flask, request
from google.cloud import bigquery
from google.oauth2 import service_account
import keras.backend.tensorflow_backend as tb
from keras.models import load_model
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import scipy
import scipy.cluster
from sklearn.externals import joblib 
from sklearn.neighbors import NearestNeighbors

BQ_KEY_FILE = "bq-service-account-key.json"
CNN_MODEL_FILE = "cover_cnn_model.h5"
INDEXED_BOOKS_CSV = "indexed_books.csv"
INPUT_FORM_HTML_FILE = "static/inputForm.html"
KNN_MODEL_FILE = "knn-model.jlib"
RESULTS_HTML_FILE = "static/results.html"

credentials = service_account.Credentials.from_service_account_file(
    BQ_KEY_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id
)
app = Flask(__name__)

with open(INPUT_FORM_HTML_FILE) as f:
  inputForm = "\n".join(f.readlines())
with open(RESULTS_HTML_FILE) as f:
  results = "\n".join(f.readlines())

knn_model = joblib.load(KNN_MODEL_FILE)  
data = pd.read_csv(INDEXED_BOOKS_CSV)
indexed_books = data['0'].values

cnn_model = load_model(CNN_MODEL_FILE)

def show_color(colors):
    fig,ax = plt.subplots()
    currentAxis = plt.gca()
    x=0
    y=0
    width = 1/len(colors)
    for rgb in colors:
        colour = binascii.hexlify(bytearray(int(c) for c in rgb)).decode('ascii')
        colour = '#'+colour
        currentAxis.add_patch(Rectangle((x, y), width, 1, alpha=1, facecolor=colour))
        x=x+width

def dominant_colors(ar):
    NUM_CLUSTERS = 5
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    return codes , peak

def image_colorfulness(image):
    R = np.array([x[0] for x in image])
    G = np.array([x[1] for x in image])
    B = np.array([x[2] for x in image])
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def image_brightness(im):
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def image_features(im):
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    
    colorfullness = image_colorfulness(ar)
    bright = image_brightness(im)
    codes, peak = dominant_colors(ar)
    
    return [round(peak[0],2),round(peak[2],2),round(peak[2],2),round(bright,2),round(colorfullness,2)]

def read_image_for_cnn(im):
    outputImage = np.zeros((64, 64, 3), dtype="uint8")
    outputImage[0:64, 0:64] = im.resize((64, 64))

    return np.array([outputImage]) / 255.0

def get_similar_ratings(ASINs):
  ASIN = '"' + '","'.join(ASINs) + '"'

  sql = """
SELECT *
FROM
    `eecs-e6893-book-cover.book_metadata.book_ratings`
WHERE
    ASIN in ({})
  """.format(ASIN)
  df = client.query(sql).to_dataframe()
  ratingMap = {}
  for ind in df.index:
    ratingMap[df["ASIN"][ind]] = df["rating"][ind]
  return ratingMap

@app.route('/')
def hello():
  return inputForm

@app.route('/processImage', methods=['POST'])
def processImage():
  uploadedFile = request.files['pic']
  im = Image.open(uploadedFile)

  # HACK to get around https://github.com/keras-team/keras/issues/13353
  tb._SYMBOLIC_SCOPE.value = True

  predictedRating = cnn_model.predict(read_image_for_cnn(im))[0][0] * 5
  print(predictedRating)

  processedIm = image_features(im)
  d,i = knn_model.kneighbors([processedIm])

  ASINs = indexed_books[i[0]].tolist()
  print(ASINs)
  ASINRatingMap = get_similar_ratings(ASINs)

  finalRes = [predictedRating]
  for ASIN in ASINs:
    finalRes.append(ASIN)
    if ASIN in ASINRatingMap:
      finalRes.append("Rating: " + str(ASINRatingMap[ASIN]))
    else:
      finalRes.append("")

  return results % tuple(finalRes)

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
