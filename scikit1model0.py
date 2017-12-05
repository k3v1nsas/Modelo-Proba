import pandas as pd 
advert = pd.read_csv("./esteeselbueno.csv")

from sklearn.feature_selection import RFE 

from sklearn.svm import SVR

feature_cols = ['Prestamos','Depositos','Activos','Arrendamientos'] 

X = advert[feature_cols] 

Y = advert['Tasa']

estimator = SVR(kernel="linear") 
selector = RFE(estimator,2,step=1) 

selector = selector.fit(X, Y) 

print(selector.support_) 

print(selector.ranking_)