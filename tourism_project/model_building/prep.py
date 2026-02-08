# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
# for converting text data in to numerical representation
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

repo_id = "vamshf/tourism-package-prediction"
repo_type = "datasets"

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = f"hf://{repo_type}/{repo_id}/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df = df.drop(columns=['CustomerID', 'Designation','Unnamed: 0'],errors='ignore')


#generic code to populate null values
numerical_features = ['Age', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                         'PreferredPropertyStar','NumberOfTrips', 
                         'MonthlyIncome', 'DurationOfPitch', 
                         'NumberOfChildrenVisiting']

categorical_features = ['TypeofContact', 'Occupation', 'Gender',
                         'MaritalStatus', 'ProductPitched']

# listing categorical and numerical features
# numerical_features = df.select_dtypes(include=np.number).columns.tolist()
#categorical_features = df.select_dtypes(include='object').columns.tolist()

for col in numerical_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

for col in categorical_features:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)


'''
We can consolidate data for below features
   1) Gender (Fe Male colums to Female)
   2) MaritalStatus (Un married to Single)
'''
df['Gender'] = df['Gender'].apply(lambda x: 'Female' if x == 'Fe Male' else x)
df['MaritalStatus']=df['MaritalStatus'].apply(lambda x: 'Single' if x == 'Unmarried' else x)


# Define target and features
target = 'ProdTaken'
y = df[target]
X = df[numerical_features + categorical_features]



# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)


files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type="dataset",
    )
