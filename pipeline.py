#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:39:33 2017

@author: aorlenko
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, Binarizer, FunctionTransformer, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
#from tpot.operators.preprocessors import ZeroCount
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import classification_report
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import FeatureAgglomeration
from collections import OrderedDict

tpot_data = np.recfromcsv('/Users/aorlenko/Downloads/anges/anges_data_ind_0_vs_2.csv', delimiter=',', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('indication'), axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(features, tpot_data['indication'], random_state=42)
noclass =  list(tpot_data.dtype.names)
noclass.remove('indication')

big_list1 = []
big_list1_5 = []
big_list2 = []
big_list1_5.append("import pandas as pd\n")
big_list1_5.append("import seaborn as sns\n")
big_list1_5.append("from sklearn.metrics import recall_score\n")
big_list1_5.append("sns.set(color_codes=True)\n")
big_list1_5.append("import matplotlib.pyplot as plt\n")
big_list1_5.append("tpot_data = np.recfromcsv('/Users/aorlenko/Downloads/anges/anges_data_ind_0_vs_2.csv', delimiter=',', dtype=np.float64)\n")
big_list1_5.append("features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('indication'), axis=1)\n")
big_list1_5.append("X_train, X_test, Y_train, Y_test = train_test_split(features, tpot_data['indication'], random_state=42)\n")
big_list1_5.append('big_list_features = []\n')
big_list1_5.append('train = []\n')
big_list1_5.append('test = []\n')
big_list1_5.append("thefile = open('/Users/aorlenko/Downloads/anges/py/0_2/1_out/r2_2.txt', 'w')\n")       
big_list1_5.append("def balanced_accuracy(result):\n")       
big_list1_5.append("\tall_classes = list(set(result['class'].values))\n")       
big_list1_5.append("\tall_class_accuracies = []\n")       
big_list1_5.append("\tfor this_class in all_classes:\n")       
big_list1_5.append("\t\tthis_class_accuracy = len(result[(result['guess'] == this_class) & (result['class'] == this_class)])"+"\\"+"\n")       
big_list1_5.append("\t\t\t/ float(len(result[result['class'] == this_class]))\n")       
big_list1_5.append("\t\tall_class_accuracies.append(this_class_accuracy)\n")       
big_list1_5.append("\tbalanced_accuracy = np.mean(all_class_accuracies)\n")       
big_list1_5.append("\treturn balanced_accuracy\n")       

onlyfiles = [f for f in listdir('/Users/aorlenko/Downloads/anges/py/0_2/1/') if isfile(join('/Users/aorlenko/Downloads/anges/py/0_2/1/', f))]
print(onlyfiles)
length_of_skip = []
for i in range(len(onlyfiles)):
    print(i)
    with open('/Users/aorlenko/Downloads/anges/py/0_2/1/'+ str(onlyfiles[i])) as f:
        print(str(onlyfiles[i]))
        lines = []
        for skippable_line in f:  # First skim over all lines until we find '#NOTE'.
            if '# NOTE' in skippable_line:
                break
            else:
                lines.append(skippable_line)
                #print(skippable_line)
        print(len(lines)) 
        length_of_skip.append(len(lines))
        big_list1.extend(lines)
    #print(i)
    #print(len(length_of_skip))
    with open('/Users/aorlenko/Downloads/anges/py/0_2/1/'+ str(onlyfiles[i])) as f:
        lines2 = []
        for skippable_line in f:  # First skim over all lines until we find '#NOTE'.
            if 'exported_pipeline' in skippable_line:
                if 'make_pipeline' in skippable_line:
                    lines2.append("exported_pipeline" +str(i)+"= make_pipeline(\n")
                else:
                    if 'exported_pipeline.fit' not in skippable_line:
                        lines2.append("exported_pipeline" +str(i)+"= make_pipeline(\n")
                        #print(skippable_line.split(" = ")[1])
                        lines2.append(skippable_line.split(" = ")[1])
                        lines2.append(")\n")

            else:
                lines2.append(skippable_line)
        #print(lines2)
        #print(length_of_skip[i])
        del lines2[:length_of_skip[i]+5]#removing first 14 elements from the file
        del lines2[-3:]#removing last 4 elements from the file
        big_list2.extend(lines2)
        #big_list2.append('\n')
        big_list2.append('clf'+str(i)+' = exported_pipeline'+str(i)+'.fit(X_train, Y_train)'+ '\n')
        big_list2.append('noclass'+str(i)+ ' =  list(tpot_data.dtype.names)\n')
        big_list2.append("noclass"+str(i)+".remove('indication')\n")
        big_list2.append('feature_list_'+str(i)+' = noclass'+str(i)+'\n')
        big_list2.append("for i in range(len(feature_list_"+str(i)+")):\n")
        big_list2.append("\tif feature_list_"+str(i)+"[i] not in big_list_features:\n")
        big_list2.append("\t\tbig_list_features.append(feature_list_"+str(i)+"[i])\n")
        big_list2.append('Y_pred_tr'+str(i)+ '= clf'+str(i)+'.predict(X_train)' + '\n')
        big_list2.append('Y_pred_ts'+str(i)+ '= clf'+str(i)+'.predict(X_test)' + '\n')
        big_list2.append("data_tr"+str(i)+ " = pd.DataFrame({'class': Y_train, 'guess': Y_pred_tr"+str(i)+ "}) \n")        
        big_list2.append("data_ts"+str(i)+ " = pd.DataFrame({'class': Y_test, 'guess': Y_pred_ts"+str(i)+ "})\n")
        big_list2.append("print('balanced accuracy clf"+str(i)+"_train:'+ format(balanced_accuracy(data_tr"+str(i)+ ")))\n")
        big_list2.append("print('balanced accuracy clf"+str(i)+"_test:'+ format(balanced_accuracy(data_ts"+str(i)+ ")))\n")
        big_list2.append('train.append(balanced_accuracy(data_tr'+str(i)+ '))\n')
        big_list2.append('test.append(balanced_accuracy(data_ts'+str(i)+ '))\n')
        big_list2.append("strr"+str(i)+" = str("+str(i)+")+' '+str(clf"+str(i)+".score(X_train, Y_train))+'/' + str(clf"+str(i)+".score(X_test, Y_test))+ '\\n'\n")
        big_list2.append("thefile.write(strr"+str(i)+")\n")
        print(lines2[len(lines2)-3])
        if 'LinearSVC' in lines2[len(lines2)-3]:
            big_list2.append('coef'+str(i)+' = clf'+str(i)+'.steps[-1][1].coef_'+ '\n')
            big_list2.append('print(\'feature size \' + str(coef'+str(i)+'.shape))\n')    
            big_list2.append('new_df'+str(i)+' = pd.DataFrame(data =coef'+str(i)+'[0], index = feature_list_'+str(i)+').sort_values(0,ascending=False)\n')
            big_list2.append("new_df"+str(i)+"_2 = new_df"+str(i)+".sort_values(0)\n")
            big_list2.append("new_df"+str(i)+"_2[0] = new_df"+str(i)+"_2[0].abs()\n")
            big_list2.append("new_df"+str(i)+"_2.sort_values(0,ascending=False, inplace = True)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2[0].rank(ascending=0)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2['Ranked'].round()\n")
            big_list2.append("new_df"+str(i)+".to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/"+str(i)+".csv')\n\n\n")

        if 'LogisticRegression' in lines2[len(lines2)-3]:
            big_list2.append('coef'+str(i)+' = clf'+str(i)+'.steps[-1][1].coef_'+ '\n')
            big_list2.append('print(\'feature size \' + str(coef'+str(i)+'.shape))\n') 
            big_list2.append('new_df'+str(i)+' = pd.DataFrame(data =coef'+str(i)+'[0], index = feature_list_'+str(i)+').sort_values(0,ascending=False)\n')
            big_list2.append("new_df"+str(i)+"_2 = new_df"+str(i)+".sort_values(0)\n")
            big_list2.append("new_df"+str(i)+"_2[0] = new_df"+str(i)+"_2[0].abs()\n")
            big_list2.append("new_df"+str(i)+"_2.sort_values(0,ascending=False, inplace = True)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2[0].rank(ascending=0)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2['Ranked'].round()\n")
            big_list2.append("new_df"+str(i)+".to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/"+str(i)+".csv')\n\n\n")

        if 'GradientBoostingClassifier' in lines2[len(lines2)-3]:
            big_list2.append('coef'+str(i)+' = clf'+str(i)+'.steps[-1][1].feature_importances_ '+ '\n')
            big_list2.append('print(\'feature size \' + str(coef'+str(i)+'.shape))\n')
            big_list2.append('new_df'+str(i)+' = pd.DataFrame(data =coef'+str(i)+', index = feature_list_'+str(i)+').sort_values(0,ascending=False)\n')
            big_list2.append("new_df"+str(i)+"_2 = new_df"+str(i)+".sort_values(0)\n")
            big_list2.append("new_df"+str(i)+"_2[0] = new_df"+str(i)+"_2[0].abs()\n")
            big_list2.append("new_df"+str(i)+"_2.sort_values(0,ascending=False, inplace = True)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2[0].rank(ascending=0)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2['Ranked'].round()\n")
            big_list2.append("new_df"+str(i)+".to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/"+str(i)+".csv')\n\n\n")

        if 'ExtraTreesClassifier' in lines2[len(lines2)-3]:
            big_list2.append('coef'+str(i)+' = clf'+str(i)+'.steps[-1][1].feature_importances_ '+ '\n')
            big_list2.append('print(\'feature size \' + str(coef'+str(i)+'.shape))\n') 
            big_list2.append('new_df'+str(i)+' = pd.DataFrame(data =coef'+str(i)+', index = feature_list_'+str(i)+').sort_values(0,ascending=False)\n')
            big_list2.append("new_df"+str(i)+"_2 = new_df"+str(i)+".sort_values(0)\n")
            big_list2.append("new_df"+str(i)+"_2[0] = new_df"+str(i)+"_2[0].abs()\n")
            big_list2.append("new_df"+str(i)+"_2.sort_values(0,ascending=False, inplace = True)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2[0].rank(ascending=0)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2['Ranked'].round()\n")
            big_list2.append("new_df"+str(i)+".to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/"+str(i)+".csv')\n\n\n")

        if 'DecisionTreeClassifier' in lines2[len(lines2)-3]:
            big_list2.append('coef'+str(i)+' = clf'+str(i)+'.steps[-1][1].feature_importances_ '+ '\n')
            big_list2.append('print(\'feature size \' + str(coef'+str(i)+'.shape))\n') 
            big_list2.append('new_df'+str(i)+' = pd.DataFrame(data =coef'+str(i)+', index = feature_list_'+str(i)+').sort_values(0,ascending=False)\n')
            big_list2.append("new_df"+str(i)+"_2 = new_df"+str(i)+".sort_values(0)\n")
            big_list2.append("new_df"+str(i)+"_2[0] = new_df"+str(i)+"_2[0].abs()\n")
            big_list2.append("new_df"+str(i)+"_2.sort_values(0,ascending=False, inplace = True)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2[0].rank(ascending=0)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2['Ranked'].round()\n")
            big_list2.append("new_df"+str(i)+".to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/"+str(i)+".csv')\n\n\n")

        if 'RandomForestClassifier' in lines2[len(lines2)-3]:
            big_list2.append('coef'+str(i)+' = clf'+str(i)+'.steps[-1][1].feature_importances_ '+ '\n')
            big_list2.append('print(\'feature size \' + str(coef'+str(i)+'.shape))\n') 
            big_list2.append('new_df'+str(i)+' = pd.DataFrame(data =coef'+str(i)+', index = feature_list_'+str(i)+').sort_values(0,ascending=False)\n')
            big_list2.append("new_df"+str(i)+"_2 = new_df"+str(i)+".sort_values(0)\n")
            big_list2.append("new_df"+str(i)+"_2[0] = new_df"+str(i)+"_2[0].abs()\n")
            big_list2.append("new_df"+str(i)+"_2.sort_values(0,ascending=False, inplace = True)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2[0].rank(ascending=0)\n")
            big_list2.append("new_df"+str(i)+"_2['Ranked'] = new_df"+str(i)+"_2['Ranked'].round()\n")
            big_list2.append("new_df"+str(i)+".to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/"+str(i)+".csv')\n\n\n")
        
big_list1 = list(OrderedDict.fromkeys(big_list1))
big_list1.append('\n') 
big_list1.extend(big_list1_5)
big_list1.extend(big_list2)
for i in range(len(onlyfiles)):
    big_list1.append('value'+str(i)+' = []\n')
    big_list1.append('value'+str(i)+'_2 = []\n')
big_list1.append('for i in range(len(big_list_features)): \n')
for i in range(len(onlyfiles)):
    big_list1.append('\tif big_list_features[i] in new_df'+str(i)+'.index.values:\n')
    big_list1.append('\t\tfor j in range(len(new_df'+str(i)+'.index.values)):\n')
    big_list1.append('\t\t\tif big_list_features[i] == new_df'+str(i)+'.index.values[j]:\n')
    big_list1.append('\t\t\t\tvalue'+str(i)+'.append(new_df'+str(i)+'.iloc[j,0])\n')
    big_list1.append('\tif big_list_features[i] not in new_df'+str(i)+'.index.values:\n')
    big_list1.append('\t\tvalue'+str(i)+'.append(0.0)\n')
big_list1.append('for i in range(len(big_list_features)): \n')
for i in range(len(onlyfiles)):
    big_list1.append('\tif big_list_features[i] in new_df'+str(i)+'_2.index.values:\n')
    big_list1.append('\t\tfor j in range(len(new_df'+str(i)+'_2.index.values)):\n')
    big_list1.append('\t\t\tif big_list_features[i] == new_df'+str(i)+'_2.index.values[j]:\n')
    big_list1.append('\t\t\t\tvalue'+str(i)+'_2.append(new_df'+str(i)+'_2.iloc[j,1])\n')
    big_list1.append('\tif big_list_features[i] not in new_df'+str(i)+'_2.index.values:\n')
    big_list1.append('\t\tvalue'+str(i)+'_2.append(len(big_list_features))\n')    
for i in range(len(onlyfiles)):
    big_list1.append('print(len(value'+str(i)+'))\n')
    big_list1.append('print(len(value'+str(i)+'_2))\n')
big_list1.append('new_df_merged = pd.DataFrame(data =None, index = big_list_features, columns = None)\n')
big_list1.append('new_df_merged_2 = pd.DataFrame(data =None, index = big_list_features, columns = None)\n')

for i in range(len(onlyfiles)):
    big_list1.append("new_df_merged['"+str(i)+"'] = pd.Series(value"+str(i)+",index=new_df_merged.index)\n")
    big_list1.append("new_df_merged_2['"+str(i)+"'] = pd.Series(value"+str(i)+"_2,index=new_df_merged_2.index)\n")
big_list1.append("new_df_merged['mean'] = new_df_merged.mean(axis=1)\n")
big_list1.append("new_df_merged.sort_values('mean',ascending=False, inplace =True)\n")
big_list1.append("new_df_merged.to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/full_coefs.csv')\n")

big_list1.append("new_df_merged_2['summ'] = new_df_merged_2.sum(axis=1)\n")
big_list1.append("new_df_merged_2.sort_values('summ', inplace =True)\n")
big_list1.append("new_df_merged_2['summ_r'] = 1/new_df_merged_2['summ']\n")

big_list1.append("new_df_merged_2.to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/rank.csv')\n")
big_list1.append("print('av_train ' + str(sum(train)/len(train)))\n")
big_list1.append("print('av_test ' + str(sum(test)/len(test)))\n")
big_list1.append("strrr = 'av_R2 ' +str(sum(train)/len(train))+'/' +str(sum(test)/len(test))\n")
big_list1.append("thefile.write(strrr)\n")
big_list1.append("metabs = pd.read_csv('/Users/aorlenko/Downloads/anges/anges_data_ind_0_vs_2.csv', delimiter=',')\n")
big_list1.append("metabs.columns = map(str.lower, metabs.columns)\n")
big_list1.append("metabs.drop('indication', axis=1, inplace = True)\n")

big_list1.append("for i in range(metabs.columns.size):\n")
big_list1.append("\tfor j in range(new_df_merged_2.index.values.size):\n")
big_list1.append("\t\tif metabs.columns[i] == new_df_merged_2.index.values[j]:\n")
big_list1.append("\t\t\ta = str(new_df_merged_2.index.values[j])+ '_' + str(new_df_merged_2.summ[j])\n")            
big_list1.append("\t\t\tmetabs.rename(columns={metabs.columns[i]: a}, inplace = True)\n")
big_list1.append("ser = metabs.apply(lambda x: np.all(x==0))\n")
big_list1.append("for i in range(ser.size):\n")
big_list1.append("\tif ser[i] == True:\n")
big_list1.append("\t\tmetabs.drop(ser.index[i],axis= 1, inplace = True)\n")
big_list1.append("\t\tprint(ser.index[i])\n")

big_list1.append("metabs.to_csv('/Users/aorlenko/Downloads/anges/py/0_2/1_out/anges_data_ind_0_vs_2.csv', index = False)\n")

big_list1.append("plt.ylabel('Ranks coefficients')\n")
big_list1.append("plt.xlabel('Features')\n")
#big_list1.append("plt.title('Metphormin metabolites (with daily dosage) from batch_8')\n")  
big_list1.append("plt.bar(range(new_df_merged_2['summ_r'].values.size),new_df_merged_2['summ_r'].values)\n")
big_list1.append("plt.xticks(range(new_df_merged_2['summ_r'].values.size), new_df_merged_2.index,rotation =90)\n")
big_list1.append("plt.tight_layout()\n")
big_list1.append("plt.show()\n")
big_list1.append("plt.savefig('/Users/aorlenko/Downloads/anges/py/0_2/1_out/ranked_features.png')\n")
thefile = open('/Users/aorlenko/Downloads/anges/py/0_2/1_out/0_2.py', 'w')       
for item in big_list1:
    thefile.write(item)