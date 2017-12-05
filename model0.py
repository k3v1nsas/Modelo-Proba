# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:19:13 2017

@author: JAVIER1
"""


import pandas as pd
import statsmodels.formula.api as smf

advert = pd.read_csv("./esteeselbueno.csv")
model3=smf.ols(formula='Tasa~Prestamos+Arrendamientos',data=advert).fit()
print(model3.params)
print(model3.pvalues)

a = model3.params[0]
btv = model3.params[1]
bradio = model3.params[2]

advert["Activos_pred"] = a + btv * advert["Prestamos"] + \
                        bradio * advert["Arrendamientos"]
                        
sales_pred=model3.predict(advert[['Prestamos','Arrendamientos']])
#print(sales_pred.head())

print(model3.summary())
                        
#RSE
import numpy as np

advert['SSD'] = (advert['Tasa']- \
                  advert['Activos_pred'])**2
SSD=advert.sum()['SSD']
n = len(advert["Tasa"])
print("n",n)
p = 2
RSE=np.sqrt(SSD/(n-p-1))
print("RSE", RSE)
salesmean=np.mean(advert['Tasa'])
print("salesmean", salesmean)
error=RSE/salesmean
print("error", error)