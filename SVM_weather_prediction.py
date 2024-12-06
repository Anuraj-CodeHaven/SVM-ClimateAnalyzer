from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as p
from sklearn.metrics import classification_report,accuracy_score


weather=p.read_csv('weather.csv')
print(weather.head())
print(weather.info())
X=weather[['Temperature','Humidity','WindSpeed','Pressure']].values
Y=weather['Weather']

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
Scal=StandardScaler()
X_train=Scal.fit_transform(X_train)
X_test=Scal.transform(X_test)

Classifier=SVC(kernel='rbf')

Classifier.fit(X_train,Y_train)
predict=Classifier.predict(X_test)

print("accuracy of the model is ",accuracy_score(predict,Y_test))
print("classification report is ",classification_report(predict,Y_test))
