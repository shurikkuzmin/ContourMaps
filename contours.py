import numpy as np
import cv2
import flask
import os
from pathlib import Path

import time

def processImage(imageName, color):
   
    image = cv2.imread(str(imageName)) 
    borderColorHSV = cv2.cvtColor(np.array([[color]]), cv2.COLOR_BGR2HSV)
    borderColorHSV=borderColorHSV.ravel()
    print(image.shape)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_range = np.array([np.max([0, borderColorHSV[0] - 5]), 0, 50])
    upper_range = np.array([np.max([10, borderColorHSV[0] + 5]), 255, 255])

    # Get the mask
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Now clean the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=6)

    # Finding Contours 
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = np.array(contours)
    areas = np.array([cv2.contourArea(contour) for contour in contours])
    areaIndices = np.argsort(areas)
    contours = contours[areaIndices]

    # Prepare the largest contour
    mainContour = contours[-1]
    mainContour = mainContour.reshape((mainContour.shape[0], mainContour.shape[2]))
    mainContour = mainContour - np.array(np.average(mainContour, axis=0), dtype=np.int)
    xMin, yMin = np.min(mainContour, axis=0)
    xMax, yMax = np.max(mainContour, axis=0)
    mainContourReal = np.asarray(mainContour, dtype=np.float)
    if xMax - xMin != 0 and yMax - yMin != 0:
        ratio = (yMax - yMin) / (xMax - xMin)
        print(ratio)
        mainContourReal[:,0] = 35 + 5 * mainContour[:,0] / (xMax - xMin)
        mainContourReal[:,1] = 55 - 5 * mainContour[:,1] / (yMax - yMin) * ratio
    
    mainContourReal = np.vstack((mainContourReal, mainContourReal[0,:]))

    dictContour = {"type": "FeatureCollection", "features": []}
    dictContour["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [mainContourReal.tolist()]}})
    return dictContour
    
    
UPLOAD_FOLDER = "files"

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET','POST'])
def start():
    urlFileName = None
    if flask.request.method == 'POST':
        f = flask.request.files["mapPicture"]
        fileName = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        urlFileName = app.config['UPLOAD_FOLDER'] + "/" + f.filename
        f.save(fileName)
        #fileString = f.read()
        #numpyImage = np.fromstring(fileString, np.uint8)
        #img = cv2.imdecode(numpyImage, cv2.IMREAD_COLOR)
        #cv2.imshow("TEst",img)
        #cv2.waitKey(0)
        #image = cv2.imread(f)

        #return flask.redirect(flask.url_for('index'))
    
    return flask.render_template("map.html", imageName=urlFileName)

@app.route('/files/<fileName>')
def send_file(fileName):
    return flask.send_from_directory(UPLOAD_FOLDER, fileName)

@app.route('/processImage', methods=['POST'])
def process():
    color = np.array([flask.request.form["blue"], flask.request.form["green"], flask.request.form["red"]], dtype=np.uint8)
    imageName = Path(flask.request.form["fileName"]) 
    contour = processImage(imageName, color)

    return flask.jsonify(contour)

if __name__ == '__main__':
    app.run(debug=True)
