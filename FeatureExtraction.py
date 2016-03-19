import numpy as np
import argparse as ap
import os
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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

    data_X, ture_y,name =feature_extraction(path)
    print name
    print ture_y

    X_train, X_test, y_train, y_test = train_test_split(data_X, ture_y, test_size=0.2,random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    dot_data = StringIO()
    with open('tree.dot', 'w') as dotfile:
        tree.export_graphviz(
            clf,
            dotfile,
            #feature_names=['unique_suits', 'unique_ranks']
        )

    y_pred=clf.predict(X_test);
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print "Probablity::"
    print clf.predict_proba(X_test)

    # tree.export_graphviz(clf, out_file=dot_data,
    #                      #feature_names=iris.feature_names,
    #                      #class_names=iris.target_names,
    #                      filled=True, rounded=True,
    #                      special_characters=True)

    #graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #Image(graph.create_png())


    # tree.export_graphviz(clf,out_file='tree.dot')
    # with open("tree.dot", 'w') as f:
    #     f = tree.export_graphviz(clf, out_file=f)




   #
   #  clf =linear_model.SGDClassifier()
   #
   #  clf.partial_fit(X_train, y_train,classes=[i for i in range(0,21)])
   #
   #
   #
   # #print clf.predict(X_test[20]) #for known classes for UC_MERCED DATA
   # # print y_test[20]
   #
   #
   #
   #
   #
   #  test_name = os.listdir(test_path)
   #  index=[]
   #  for i in test_name:
   #      for id,j in enumerate(name):
   #          if i.split("_")[0] == j.split("_")[0]:
   #              #print i,j
   #              index.append(id)
   #
   #  stest_X, stest_y, stest_name =feature_extraction1(test_path,index)
   #
   #  print stest_name
   #  print stest_y
   #
   #
   #  print clf.predict(stest_X[38])
   #  print stest_y[38]
   #
   #  spredic=clf.predict(stest_X)
   # # print spredic
   #  #print stest_y
   #  cm = confusion_matrix(spredic, stest_y)
   #  print "###########################################"
   #  print "Matrix"
   #  print cm
   #  print "###########################################"



    #print clf.predict(test_X[4]) Faild
    #
    #
    # y_pred=clf.predict(test_X);
    # cm = confusion_matrix(test_y, y_pred)
    #
    # print clf.predict(test_X);
    #
    # plot_confusion_matrix(cm,Cname=name)
    # np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization')
    # print(cm)
    # plt.figure()
    # plot_confusion_matrix(cm)
    # plt.show()



if __name__=="__main__":
     main()