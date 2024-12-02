import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, download_plotlyjs, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('Train.csv')
df_test = pd.read_csv('Test.csv')


try:
    df_train.drop(labels=['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)
    df_test.drop(labels=['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)
except Exception as e:
    pass

temp_df = df_train.isnull().sum().reset_index()
temp_df['Percentage'] = (temp_df[0]/len(df_train))*100
temp_df.columns = ['Column Name', 'Number of null values', 'Null values in percentage']
print(f"The length of dataset is \t {len(df_train)}")
# print(temp_df)



def convert(x):
    if x in ['low fat', 'LF']: 
        return 'Low Fat'
    elif x=='reg':
        return 'Regular'
    else:
        return x

df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].apply(convert)
df_test['Item_Fat_Content'] = df_train['Item_Fat_Content'].apply(convert)

print(f"Now Unique values in this column in Train Set are\t  {df_train['Item_Fat_Content'].unique()} ")
print(f"Now Unique values in this column in Test Set are\t  {df_test['Item_Fat_Content'].unique()} ")

# count = df_train['Outlet_Size'].value_counts().reset_index()
# count.iplot(kind='bar', color='deepskyblue', x='index', y='Outlet_Size', title='High VS Mediun VS Small', xTitle='Size', yTitle='Frequency')

df_train['Outlet_Size'].fillna(value='Medium', inplace= True)
df_test['Outlet_Size'].fillna(value='Medium', inplace= True)

x_train = df_train.iloc[:, :-1].values    # Features Matrix
y_train = df_train.iloc[:,-1].values   # Target Vector
x_test = df_test.values    # Features Matrix


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # To deal with Categorical Data in Target Vector.
from sklearn.model_selection import cross_val_score   # To check the accuracy of the model.
from sklearn.preprocessing import StandardScaler   # To appy scaling on the dataset.

imputer = SimpleImputer()

x_train[:,[0]] = imputer.fit_transform(x_train[:,[0]])
x_test[:,[0]] = imputer.fit_transform(x_test[:,[0]])



labelencoder_x = LabelEncoder()
x_train[:, 1 ] = labelencoder_x.fit_transform(x_train[:,1 ])
x_train[:, 3 ] = labelencoder_x.fit_transform(x_train[:,3 ])
x_train[:, 5 ] = labelencoder_x.fit_transform(x_train[:,5 ])
x_train[:, 6 ] = labelencoder_x.fit_transform(x_train[:,6 ])
x_train[:, 7 ] = labelencoder_x.fit_transform(x_train[:,7 ])


# Let's apply same concept on test set.
x_test[:, 1 ] = labelencoder_x.fit_transform(x_test[:,1 ])
x_test[:, 3 ] = labelencoder_x.fit_transform(x_test[:,3 ])
x_test[:, 5 ] = labelencoder_x.fit_transform(x_test[:,5 ])
x_test[:, 6 ] = labelencoder_x.fit_transform(x_test[:,6 ])
x_test[:, 7 ] = labelencoder_x.fit_transform(x_test[:,7 ])


sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=None)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance:", explained_variance)

pca = PCA(n_components=8)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

# Multi Linear Regression

regressor_multi = LinearRegression()
regressor_multi.fit(x_train,y_train)
accuracy_multi = cross_val_score(estimator=regressor_multi, X=x_train, y=y_train,cv=10)
print('\n-------------Multi linear Regression----------')
print(f"The accuracy of the Multi-linear Regressor Model is \t {accuracy_multi.mean()}")
print(f"The deviation in the accuracy is \t {accuracy_multi.std()}")


# Random Forest Model

regressor_random = RandomForestRegressor(n_estimators=100,)
regressor_random.fit(x_train,y_train)
accuracy_rforest = cross_val_score(estimator=regressor_random, X=x_train, y=y_train,cv=10)
print('\n---------------Random Forest Model----------')
print(f"The accuracy of the Random Forest Model is \t {accuracy_rforest.mean()}")
print(f"The deviation in the accuracy is \t {accuracy_rforest.std()}")

# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)
regressor_poly=LinearRegression()
regressor_poly.fit(x_poly,y_train)
accuracy_poly = cross_val_score(estimator=regressor_poly, X=x_train, y=y_train,cv=10)
print('\n----------------Polynomial Regression-----------')
print(f"The accuracy of the Polynomial Regression Model is \t {accuracy_poly.mean()}")
print(f"The deviation in the accuracy is \t {accuracy_poly.std()}")


algorithms = ['Multi-linear Regression', 'Random Forest', 'Polynomial Regression']
results = [accuracy_multi.mean(), accuracy_rforest.mean(), accuracy_poly.mean() ]
plt.bar(algorithms, results, color ='skyblue', width = 0.5)
plt.xlabel('Accuracy')
plt.ylabel('Algorithms')
plt.title('Algorithms vs Accuracy Graph')
plt.show()

y_pred = regressor_multi.predict(x_test)

print('The prediction for the products using Multi linear model: ')
print(*list(y_pred[:15]), sep='\n')



# def students(request):
#     k = request.GET.get('k', None)
#     students = Student.objects.all()
#     if k != None:
#         students = Student.objects.filter(regd_no__contains = k)
#     return render(request, 'students.html', {'students':students})