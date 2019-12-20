import config
import cv2
from flask import Flask, request, send_file
from google.cloud import bigquery
from google.oauth2 import service_account
from io import BytesIO
import keras.backend.tensorflow_backend as tb
from keras.models import load_model
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import requests
import scipy
import scipy.cluster
import shutil
from sklearn.externals import joblib 
from sklearn.neighbors import NearestNeighbors
import time

#server = "http://localhost:8080/"
server = "http://35.222.73.123:80/"

BQ_KEY_FILE = "bq-service-account-key.json"
CNN_MODEL_FILE = "cover_cnn_model_epoch200.h5"
INDEXED_BOOKS_CSV = "indexed_books.csv"
INPUT_FORM_HTML_FILE = "static/inputForm.html"
KNN_MODEL_FILE = "knn-model.jlib"
RESULTS_HTML_FILE = "static/results.html"

credentials = service_account.Credentials.from_service_account_file(
    BQ_KEY_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
bq_client = bigquery.Client(
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
  df = bq_client.query(sql).to_dataframe()
  ratingMap = {}
  for ind in df.index:
    ratingMap[df["ASIN"][ind]] = df["rating"][ind]
  return ratingMap

def downloadIm(url):
  response = requests.get(url, stream=True)
  return Image.open(response.raw)

def get_suggestion(predictedRating, processedIm, ASINs, ASINRatingMap):
  betterASIN = None
  betterASINRating = 0
  for ASIN in ASINs:
    if ASIN in ASINRatingMap and ASINRatingMap[ASIN] > betterASINRating and ASINRatingMap[ASIN] > predictedRating:
      betterASIN = ASIN
      betterASINRating = ASINRatingMap[ASIN]
  if betterASIN == None:
    return "Your book cover is perfect as it is!"
  else:
    imageGCSPath = "https://storage.googleapis.com/book-covers-e6893/covers/224x224/" + betterASIN + ".jpg"
    betterProcessedIm = image_features(downloadIm(imageGCSPath))
    if abs(betterProcessedIm[3] - processedIm[3]) > abs(betterProcessedIm[4] - processedIm[4]):
      if betterProcessedIm[3] > processedIm[3]:
        return "Suggestion: You should make the book cover brighter"
      else:
        return "Suggestion: You should make the book cover less bright"
    else:
      if betterProcessedIm[4] > processedIm[4]:
        return "Suggestion: You should make the book cover more colorful"
      else:
        return "Suggestion: You should make the book cover less colorful"

@app.route('/')
def hello():
  return inputForm

@app.route('/processImage', methods=['POST'])
def processImage():
  uploadedFile = request.files['pic']
  im = Image.open(uploadedFile)
  fileName = "static/im-" + str(time.time()) + ".jpg"
  im.save(fileName)

  # HACK to get around https://github.com/keras-team/keras/issues/13353
  tb._SYMBOLIC_SCOPE.value = True

  predictedRating = cnn_model.predict(read_image_for_cnn(im))[0][0] * 5
  print(predictedRating)

  processedIm = image_features(im)
  d,i = knn_model.kneighbors([processedIm])

  ASINs = indexed_books[i[0]].tolist()
  print(ASINs)
  ASINRatingMap = get_similar_ratings(ASINs)

  suggestion = get_suggestion(predictedRating, processedIm, ASINs, ASINRatingMap)

  finalRes = [server + fileName, "Expected rating = " + str(round(predictedRating, 2)), suggestion]
  for ASIN in ASINs:
    finalRes.append(ASIN)
    if ASIN in ASINRatingMap:
      finalRes.append("rating = " + str(round(ASINRatingMap[ASIN], 2)))
    else:
      finalRes.append("unrated")

  return results % tuple(finalRes)

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
