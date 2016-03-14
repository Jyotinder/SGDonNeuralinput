import numpy as np
import argparse as ap
import os
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,make_scorer
from sklearn.svm import SVC
from sklearn.cross_validation import KFold,cross_val_score

def feature_extraction(folder_path):
    data=[]
    y_true=[]
    count =0
    training_names = os.listdir(folder_path)
    #Parsing npz file and extracting data
    for file in training_names:
        file_path=os.path.join(folder_path,file)
        d=np.load(file_path)
        dic_array=d.f.arr_0 #dictionary of array
        x=dic_array.reshape(-1)[0]
        for key in x:
            data.append(x[key])
            y_true.append(count)
        count=count+1

    # returning data of 21*100 sample each having 4096 feture vector and label for 0-20 classes
    return data,y_true,training_names

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues,Cname=""):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(Cname))
    plt.xticks(tick_marks, Cname, rotation=45)
    plt.yticks(tick_marks, Cname)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--folderpath", help="npz folder",required=True)
    args = vars(parser.parse_args())
    path=args["folderpath"]

    X, y,name =feature_extraction(path)
    clf =linear_model.SGDClassifier()
    #clf=SVC(kernel='linear')
    #print cross_val_score(estimator=clf, X=X, y=y, scoring=None, cv=10,verbose=5)
    print cross_val_score(clf, X, y, cv=5,verbose=1)




if __name__=="__main__":
     main()