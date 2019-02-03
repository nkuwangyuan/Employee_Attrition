

### Employees Attrition ###


# Package

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# Data

data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.info()
data = data.drop(columns = ['DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','JobInvolvement','MonthlyRate','Over18','PercentSalaryHike','TrainingTimesLastYear','YearsWithCurrManager'])


# Year at Company Distribution

Attrition_by_Year = pd.DataFrame(pd.crosstab(data['YearsAtCompany'],data['Attrition'])).reset_index()
Attrition_by_Year['Attr%'] = Attrition_by_Year['Yes'] / (Attrition_by_Year['Yes'] + Attrition_by_Year['No']) * 100
Attrition_by_Year = Attrition_by_Year.sort_values(by = ['YearsAtCompany'])


# Year Distribution Plot

tmp = Attrition_by_Year
var = 'YearsAtCompany'

trace1 = go.Bar(
    x=tmp[var],
    y=tmp['Yes'],
    name='Yes',
    marker=dict(color='rgb(49,130,189)')
)

trace2 = go.Bar(
    x=tmp[var],
    y=tmp['No'],
    name='No',
    marker=dict(color='rgb(204,204,204)')
)

trace3 = go.Scatter(
    x=tmp[var],
    y=tmp['Attr%'],
    name='Attr%',
    yaxis = 'y2',
    marker=dict(color='red'),
)

layout = dict(
    title = str(var),
    xaxis=dict(range= [-0, 20]),
    yaxis=dict(range= [-0, 200],
               title= 'Count'), 
    yaxis2=dict(range= [-0, 50],
                overlaying= 'y',
                anchor= 'x',
                side= 'right',
                zeroline=False,
                showgrid= False,
                title= '% Attrition'),
    legend=dict(x=0.8,y=1,))

data_fig = [trace1, trace2, trace3]
fig = go.Figure(data=data_fig, layout=layout)
py.iplot(fig, filename = var)


# Data Preparation

# Categorical columns
cat_cols   = data.nunique()[data.nunique() < 10].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in ["Attrition"]]
num_cols   = [x for x in data.columns if x not in cat_cols + ["Attrition"] + ['EmployeeNumber']]
bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()
multi_cols = [i for i in cat_cols if i not in bin_cols]

# Label encoding
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])
data = pd.get_dummies(data = data,columns = multi_cols )

# Scale columns
std = StandardScaler()
scaled = std.fit_transform(data[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

df_data_og = data.copy()
data = data.drop(columns = num_cols,axis = 1)
data = data.merge(scaled,left_index=True,right_index=True,how = "left")


# AdaBoost Classification

y = np.array(data.Attrition.tolist())
data = data.drop('Attrition', 1)
X = np.array(data.as_matrix())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
dbt_clf = bdt.fit(X_train, y_train)


# Feature Importance Plot

def plot_feature_importance(model):
    tmp = pd.DataFrame({'Feature': list(data), 'Feature importance': model.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False).head(30)
    plt.figure(figsize = (10,15))
    plt.title('Features Importance',fontsize=14)
    s = sns.barplot(y='Feature',x='Feature importance',data=tmp, orient='h', palette="Blues_d")
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.savefig('Feature_Importance.png')
    plt.show()
    
plot_feature_importance(dbt_clf)

