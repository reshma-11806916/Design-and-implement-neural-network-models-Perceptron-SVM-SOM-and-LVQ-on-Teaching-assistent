# Design-and-implement-neural-network-models-Perceptron-SVM-SOM-and-LVQ-on-Teaching-assistent

#NAME : MEDEPALLI SUMANTH DURGA MANIKNATA 
#ROLL NO : 11806916
#PROJECT CODE STARTS BELOW  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import KFold  
from keras.models import Sequential  
from keras.layers import Dense  
from datetime import datetime  
def assessment(f_data, f_y_feature, f_x_feature, f_index=-1):    
 for f_row in f_data:  
 if f_index >= 0:  
 f_color = np.where(f_data[f_row].index == f_index,'r','g')   f_hue = None  
 else:  
 f_color = 'b'  
 f_hue = None  
  
 f_fig, f_a = plt.subplots(1, 2, figsize=(16,4))  
  
 f_chart1 = sns.distplot(f_data[f_x_feature], ax=f_a[0], kde=False,  color='g')  
 f_chart1.set_xlabel(f_x_feature,fontsize=10)  
  
 if f_index >= 0:  
 f_chart2 = plt.scatter(f_data[f_x_feature], f_data[f_y_feature],  c=f_color, edgecolors='w')  
 f_chart2 = plt.xlabel(f_x_feature, fontsize=10)   f_chart2 = plt.ylabel(f_y_feature, fontsize=10)   else:  
 f_chart2 = sns.scatterplot(x=f_x_feature, y=f_y_feature,  data=f_data, hue=f_hue, legend=False)  
 f_chart2.set_xlabel(f_x_feature,fontsize=10)  
 f_chart2.set_ylabel(f_y_feature,fontsize=10)  
 plt.show()  
  
def correlation_map(f_data, f_feature, f_number):  
 f_most_correlated =  
f_data.corr().nlargest(f_number,f_feature)[f_feature].index   f_correlation = f_data[f_most_correlated].corr()  
  
 f_mask = np.zeros_like(f_correlation)  
 f_mask[np.triu_indices_from(f_mask)] = True  
 with sns.axes_style("white"):  
 f_fig, f_ax = plt.subplots(figsize=(20, 10)) 
 sns.heatmap(f_correlation, mask=f_mask, vmin=-1, vmax=1,  square=True,  
 center=0, annot=True, annot_kws={"size": 8},  cmap="PRGn")  
 plt.show()  
sns.set()  
start_time = datetime.now()  
data = pd.read_csv('../input/smart-grid 
stability/smart_grid_stability_augmented.csv')  
map1 = {'unstable': 0, 'stable': 1}  
data['stabf'] = data['stabf'].replace(map1)  
data = data.sample(frac=1)  
data.head()  
for column in data.columns:  
 assessment(data, 'stab', column, -1)  
 data.p1.skew()  
 print(f'Split of "unstable" (0) and "stable" (1) observations in the  original dataset:')  
print(data['stabf'].value_counts(normalize=True))  
correlation_map(data, 'stabf', 14)  
X = data.iloc[:, :12]  
y = data.iloc[:, 13]  
X_training = X.iloc[:54000, :]  
y_training = y.iloc[:54000]  
X_testing = X.iloc[54000:, :]  
y_testing = y.iloc[54000:]  
ratio_training = y_training.value_counts(normalize=True)  ratio_testing = y_testing.value_counts(normalize=True)  ratio_training, ratio_testing  
X_training = X_training.values  
y_training = y_training.values  
X_testing = X_testing.values  
y_testing = y_testing.values  
scaler = StandardScaler()  
X_training = scaler.fit_transform(X_training)  
X_testing = scaler.transform(X_testing)  
classifier = Sequential()  
classifier.add(Dense(units = 24, kernel_initializer = 'uniform',  activation = 'relu', input_dim = 12)) 
classifier.add(Dense(units = 24, kernel_initializer = 'uniform',  activation = 'relu'))  
classifier.add(Dense(units = 12, kernel_initializer = 'uniform',  activation = 'relu'))  
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',  activation = 'sigmoid'))  
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics = ['accuracy'])  
cross_val_round = 1  
print(f'Model evaluation\n')  
for train_index, val_index in KFold(10, shuffle=True,  
random_state=10).split(X_training):  
 x_train, x_val = X_training[train_index], X_training[val_index]   y_train ,y_val = y_training[train_index], y_training[val_index]   classifier.fit(x_train, y_train, epochs=50, verbose=0)   classifier_loss, classifier_accuracy = classifier.evaluate(x_val,  y_val)  
 print(f'Round {cross_val_round} - Loss: {classifier_loss:.4f} |  Accuracy: {classifier_accuracy * 100:.2f} %')  
 cross_val_round += 1  
 y_pred = classifier.predict(X_testing)  
y_pred[y_pred <= 0.5] = 0  
y_pred[y_pred > 0.5] = 1  
cm = pd.DataFrame(data=confusion_matrix(y_testing, y_pred, labels=[0,  1]),  
 index=["Actual Unstable", "Actual Stable"],   columns=["Predicted Unstable", "Predicted Stable"])  cm  
print(f'Accuracy per the confusion matrix: {((cm.iloc[0, 0] + cm.iloc[1,  1]) / len(y_testing) * 100):.2f}%')  
end_time = datetime.now()  
print('\nStart time', start_time)  
print('End time', end_time)  
print('Time elapsed', end_time - start_time)
