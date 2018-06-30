
# coding: utf-8

# In[45]:


# For data manipulation, data processing
import pandas as pd
from time import time

# Import sklearn models
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
# To calculate the accuracy of the trained classifier
from sklearn.metrics import accuracy_score
# To understand the trained classifier behavior
from sklearn.metrics import confusion_matrix
# For spliting data into training and testing sets
from sklearn.model_selection import train_test_split
# For Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# For Desicion Tree Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib import pyplot


# In[46]:


# print Header
print('---- Breast Cancer Sample Dataset ----')

# Read data in CSV file
dataset = pd.read_csv('E:\\4th Yr\\ML\\Assignmntz\\data.csv')
# INPUT_PATH = "..//Assignmntz//data.csv"
# dataset = pd.read_csv(INPUT_PATH)
# Display first 10 rows 
dataset.head(10)


# In[47]:


# No of cases included in the dataset, size of the dataframe
length = len(dataset)
print 'No of cases in the dataset ->', str(len(dataset))

# No of features in the dataset
print 'No of features in the dataset ->', dataset.shape[1]-1


# In[48]:


# Descriptive statistics for each column
print('---- Descriptive statistics For Breast Cancer Dataset ----')
dataset.describe()


# In[49]:


# Data Cleaning
dataset = dataset.drop(['id', 'Unnamed: 32'], axis = 1)
# dataset.drop('id',axis=1, inplace=True)
# dataset.drop('Unnamed: 32', axis=1, inplace=True)


# In[50]:


# Convert to array
dataset['diagnosis'].unique()


# In[51]:


# Preparing data
dataset['diagnosis'] = dataset['diagnosis'].map({'M':1,'B':0})
dataset.head()


# In[52]:


# target(label) - value we want to predict
target = dataset['diagnosis']

# Remove the labels from the features
# Remove the factorial column 'diagnosis' to find the co-relation between numerical columns
features = dataset.drop('diagnosis', axis = 1)

# Saving features
new_features = list(features.columns)

features.head()


# In[53]:


# Convert to array
target.unique()


# In[54]:


features.describe()


# In[55]:


target.describe()


# In[56]:


# To split the dataset 
def split_data(features, target):
   
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size= 0.3, random_state=101)
    
    print "Training Features Shape -> ", train_x.shape
    print "Training Labels Shape   -> ", train_y.shape
    print "Testing Features Shape  -> ", test_x.shape
    print "Testing Labels Shape    -> ", test_y.shape
    print("\n")
    
    return train_x, test_x, train_y, test_y


# In[57]:


def random_forest_classifier(train_x, train_y):
    
    ran_forest = RandomForestClassifier(n_estimators=100)
#     Train the model on training data
    ran_forest.fit(train_x, train_y)    
#     print("1")
    print "Trained model -> \n", ran_forest
    
    return ran_forest


# In[58]:


def main():
  
    # Train Test Split 
    split_data(features, target)   
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size= 0.3, random_state=101)
    
    # Using Decision Tree Model
    decs_tree = DecisionTreeClassifier()
    decs_tree.fit(train_x, train_y)    
    desc_pred = decs_tree.predict(test_x)
    
    print("---- Decision Tree Model -----\n")
    print " Classification Report -> \n"
    print(classification_report(test_y, desc_pred))
    print " Confusion Matrix -> "
    print(confusion_matrix(test_y, desc_pred))
    print("\n")    
    print("---- Random Forests -----\n")
        
#     Using Random Forests
    trained_model  = random_forest_classifier(train_x, train_y)    
    ran_forest = RandomForestClassifier(n_estimators=100)
#     Train the model on training data
    ran_forest.fit(train_x, train_y)
    
#     Perform predictions
    ran_pred = ran_forest.predict(test_x)
    print("\n")
    print " Classification Report -> \n"
    print(classification_report(test_y, ran_pred))
 
    for i in xrange(0, 5):
        print "Actual outcome -> {} and Predicted outcome -> {}" .format(list(test_y)[i], ran_pred[i]) 

#     Train and Test Accuracy
    print("\n")
    print " Train Accuracy -> ", accuracy_score(train_y, trained_model.predict(train_x))
    print " Test Accuracy  -> ", accuracy_score(test_y, ran_pred)
#     Confusion matrix
    print " Confusion Matrix -> "
    print(confusion_matrix(test_y, ran_pred))
    
if __name__ == "__main__":
    main()


# In[59]:


def predict_model(model, train_x, test_x, train_y, test_y, selected_cols):
    
#     Calculate Training Time
    t0 = time()
    model.fit(train_x[selected_cols], train_y)
    training_time = time() - t0
    
#    Calculate Prediction Time
    t1 = time()
    pred = model.predict(test_x[selected_cols])
    prediction_time = time() - t1
    
#     Calculate f1_score
    score = f1_score(test_y, pred)

    print "In Random Forest \n"
    print "  f1_score is -> " , score
    print '  Accuracy is -> ' , metrics.accuracy_score(test_y, pred)
    print "  cross_val_score is ->", (cross_val_score(model, features[selected_cols], target , cv = 10).mean())   
    print "  Time for training dataset -> ", training_time
    print "  Time for prediction -> ", prediction_time


# In[60]:


Forest = RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1)


# In[61]:


train_x, test_x, train_y, test_y = train_test_split(features, target, test_size= 0.3, random_state=101)
predict_model(Forest, train_x, test_x, train_y, test_y, new_features)


# In[62]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_x, train_y);


# In[63]:


# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(new_features, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
for pair in feature_importances:
     print 'Variable: {:30} Importance: {}'.format(*pair)

# Visualize the output in a bar chart 
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_val = list(range(len(importances)))
# Make a bar chart
plt.bar(x_val, importances, orientation = 'vertical' , color='maroon')
# Tick labels for x axis
barlist = plt.xticks(x_val, new_features, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); 
plt.xlabel('Variable'); 
plt.title('Variable Importances');
       
# barlist = pyplot.bar(range(len(importance)), importance)
 # print (sorted(zip(map(lambda x: round(x, 4), importance), names), reverse=True))              

# Axis labels and title
pyplot.ylabel('Importance'); 
pyplot.xlabel('Variable'); 
pyplot.title('Variable Importances');
pyplot.show()


# In[64]:


df = pd.DataFrame(dataset)

figure, graph = plt.subplots(1)
for i in range(1):
    x=df['perimeter_mean']
    y=df['area_worst']
    graph.scatter(x,y, label=str(i))
    
plt.ylabel('Area Worst'); 
plt.xlabel('Perimeter Mean'); 
plt.title('Co-relation of perimeter_mean and area_worst');
figure.savefig('relation.png')


# In[65]:


yTest = dataset['radius_se'].as_matrix()
# Use the random forest's predict method on the test data
predictions = rf.predict(test_x)

plt.figure(figsize=(21,7))
plt.plot(yTest,label='radius_se',color='blue')
plt.plot(predictions,label='predictions',color='red')
plt.title('Actual vs Predicted')
plt.legend(loc='upper left')
plt.show()

