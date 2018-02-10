import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn import model_selection, svm, preprocessing, metrics
from soft import deleteBlanks, deskew



if __name__ == '__main__':
    dataset = datasets.fetch_mldata("MNIST Original")
    features = np.array(dataset.data).astype('float64')
    labels = np.array(dataset.target, 'int')
    deskewedFeatures = []
    for img in features:
        imgR = img.reshape(28, 28)
        #imD = deskew(imgR)
        number = deleteBlanks(imgR).flatten()
        deskewedFeatures.append(number)
        #cv2.imshow("image asdasddasdasdasdaaddasdasdasda asdasddasdasdasdaaddasdasdasda", imD)


    list_hog_fd = []
    for feature in deskewedFeatures:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')

    x_tr, x_te, y_tr, y_te = model_selection.train_test_split(hog_features,labels,test_size=0.2,random_state=42)
    #x_tr /= 255.0
    #y_tr /= 255.0

    clf = svm.SVC(kernel='linear')
    clf.fit(x_tr, y_tr)

    lab_pred = clf.predict(x_te)
    joblib.dump(clf, "digits_cls.pkl", compress=3)

    score = metrics.accuracy_score(y_te, lab_pred)
    print("finished " + str(score))