import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = dataset.iloc[:, :12].values
Y = dataset.iloc[:, 12].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

test_data=pd.DataFrame(X_test)
test_data.insert(12,"Death Event",y_test)
test_data.to_csv('heart-failure-TestData.csv') 
   
# Fitting SVM to the Training set    
classifier = SVC(kernel = 'rbf' , random_state=0)
#classifier = SVC(kernel = 'linear' , random_state=0)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_train)

print("Prediction of train data is : ",y_pred,"\n")
print ("Accuracy training data : ", accuracy_score(y_train, y_pred)*100)

TestData=pd.read_csv("heart-failure-TestData.csv")
X_test1= TestData.values[:, 1:13] 
y_test1= TestData.values[:, 13] 
y_testpred= classifier.predict(X_test1)  
print("predction of test data is :",y_testpred)
print("      ")
print ("Accuracy of test data : ", accuracy_score(y_test1, y_testpred)*100)





