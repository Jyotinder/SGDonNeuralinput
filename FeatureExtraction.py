import numpy as np
import argparse as ap
import os
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
#import pydot


def feature_extraction(folder_path):
    data=[]
    y_true=[]
    count =0
    training_names = os.listdir(folder_path)
    #Parsing npz file and extracting data
    for file in training_names:
        file_path=os.path.join(folder_path,file)
        d=np.load(file_path)
        dic_array=d.f.arr_0 #dictionary of arrayun
        x=dic_array.reshape(-1)[0]
        for key in x:
            data.append(x[key])
            y_true.append(count)
        count=count+1

    # returning data of 21*100 sample each having 4096 feture vector and label for 0-20 classes
    return data,y_true,training_names


def feature_extraction1(folder_path,index):
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
            y_true.append(index[count])
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
    parser.add_argument('-t', "--testfolderpath", help="npz folder",required=True)
    args = vars(parser.parse_args())
    path=args["folderpath"]
    test_path=args["testfolderpath"]

    data_X, ture_y,name =feature_extraction(test_path)
    print name
    print ture_y

    X_train, X_test, y_train, y_test = train_test_split(data_X, ture_y, test_size=0.2,random_state=0)
    clf =linear_model.SGDClassifier()
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test);
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm,Cname=name)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    plt.show()
    print cross_val_score(clf, data_X,ture_y, cv=5,verbose=1)

    data_X_e, ture_y_e,name_e =feature_extraction(path)
    y_pred=clf.predict(data_X_e);
    cm = confusion_matrix(ture_y_e, y_pred)
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    plt.show()



if __name__=="__main__":
     main()