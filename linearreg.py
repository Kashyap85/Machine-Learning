import numpy as np 
import pandas as pd
import pickle
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv("student-mat.csv",sep=";")
# print(data.head())
data = data[["G1","G2","G3","studytime","failures","health","absences"]]
predict = "G3"
#g1,g2,studytime,fail,health,absent
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

# best = 0
# for _ in range(30):
#     x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train,y_train)
#     acc = linear.score(x_test,y_test)
#     print(acc)
#     if acc > best:
#         best = acc
#         with open("studentmodel.pickle","wb") as f:
#             pickle.dump(linear,f)

pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)

print("Cofficient: \n",linear.coef_)
print("Intercept: \n",linear.intercept_)

predictions = linear.predict(x_test)
print(predictions)

for x in range(len(predictions)):
    print(predictions[x],x_test[x])
    print("Predicted grade- ",y_test[x],"\n")

p="health"
style.use("ggplot")
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final output grade")
plt.show()