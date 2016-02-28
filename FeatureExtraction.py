import numpy as np
import argparse as ap
import os
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import glob
import random as ran
import operator

def feature_extraction(folder_path):
    data=[]
    training_names = os.listdir(folder_path)
    for file in training_names:
        file_path=os.path.join(folder_path,file)
        d=np.load(file_path)
        dic_array=d.f.arr_0 #dictionary of array
        x=dic_array.reshape(-1)[0]
        data.append(x)
    return data

def test_classifer(data,test_no,folder_name):

    probable_list={} # file and its detection
    files= glob.glob("./classifier/*.model")
    for i, name in enumerate(folder_name): #to get the folder name for each classifier
        d=data[i]   # extracting data from dictionary

        for jj in test_no: # looping for the random 5 test files
            test=d[jj] #extracting the content form dictionary like agriculture[78]
            dd= (test).astype(np.float64) # converting the data in float 64 for more precision
            norm1 = dd / np.linalg.norm(dd) # normalizing the data so that it is b/w -1 to 1
            detection={}
            for j in files: # loop for the classifier.
                clf = joblib.load(j)
                pred = clf.predict(norm1)
                if pred==1: # if detection true that print result.
                    print "Predicted {}".format(j)
                    print "Confidence Score {} ".format(clf.decision_function(norm1))
                    print "Accuracy Score is: \n", accuracy_score(pred, [1])
                    print "Actual file type {}\n".format(folder_name[i])
                    print "file number {} \n".format(jj)
                    detection[j]=(clf.decision_function(norm1),((folder_name[i],accuracy_score(pred, [1])),jj))

            sort_dic=sorted(detection.items(),key=operator.itemgetter(1))
            probable_list[name+str(jj)]=(sort_dic[0])[0]
    return probable_list



def onlineSVM(classes,data,random,name):
    clf =linear_model.SGDClassifier()
    data_classi=[]
    label=[]
    count=0

    for i,dic in enumerate(data):
       # print(i+1)
        if count == 0:
            for key in dic:
                data1=dic[key].astype(np.float64)
                norm1 = data1 / np.linalg.norm(data1)
                data_classi.append(norm1)
                label.append(1)
                count=1
                if key > random:
                    break
        else:
            for key in dic:
                data1=dic[key].astype(np.float64)
                norm1 = data1 / np.linalg.norm(data1)
                data_classi.append(norm1)
                label.append(0)
                if key > random:
                    break

    clf.partial_fit(data_classi, label,classes=[0,1])
    path=os.getcwd()
    path=os.path.join(path,"classifier")
    if not os.path.isdir(path):
         os.makedirs(path)

    path=os.path.join(path,name+"sdg.model")
    joblib.dump(clf, path)
    print "Classifier saved to {}".format(path)

def rotate(l,n):
    return l[-n:] + l[:-n]


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--folderpath", help="npz folder",required=True)
    parser.add_argument('-t', "--test", help="1 or zero",required=True,type=int)
    args = vars(parser.parse_args())
    path=args["folderpath"]
    test= args["test"]
    data=feature_extraction(path)
    classes=[0,1]
    random =60

    if test ==0:
        training_names = os.listdir(path)
        for i, name in enumerate(training_names):
            onlineSVM(classes,data,random,name)
            data=rotate(data,-1)
    else:
        training_names = os.listdir(path)
        test=[]
        validation=[]
        for x in range(5):
            test.append( ran.randint(60,79))
            validation.append(ran.randint(80,99))
        dic=test_classifer(data,test,training_names)
        for key, value in dic:
            print key, value




if __name__=="__main__":
     main()