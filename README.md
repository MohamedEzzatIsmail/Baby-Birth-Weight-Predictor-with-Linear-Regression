# 👶 Baby Birth Weight Predictor with Linear Regression
This project builds a Linear Regression model to predict a baby's birth weight (bwt) based on several maternal and pregnancy-related factors. It uses a dataset of prenatal records to train the model and evaluates its accuracy using scikit-learn. The best model is saved using pickle.

# 📁 Dataset
The dataset contains prenatal information for multiple cases. Each row describes one case with the following features:
Column	Description
bwt	Baby birth weight (grams) (target)
gestation	Length of pregnancy (in days)
parity	Number of previous children
age	Mother's age
height	Mother's height (in inches)
weight	Mother's weight (in pounds)
smoke	Smoking status (0 = no, 1 = yes)
URL = https://www.kaggle.com/datasets/jacopoferretti/child-weight-at-birth-and-gestation-details?resource=download

# 🛠️ Features Used
Input Features (X):
gestation
parity
age
height
weight
smoke

Target (Y):
bwt (birth weight)

# 🧠 How It Works
Load and clean the dataset (babies.csv)
Select relevant features
Replace missing values with the column mean
Split the data into training and test sets
Train multiple LinearRegression models on random subsets of the data
Evaluate and save the best model based on accuracy using pickle
Load the best model and predict birth weights for test cases
Print predictions and compare with actual values

# 📦 Dependencies
Install required packages using:
pip install pandas numpy scikit-learn

# 📊 Sample Output
acc =  0.793
acc =  0.845
...
best acc =  0.867
120.3 [284.  0. 27. 62. 100.   0.] 120.0
112.6 [282.   0.  33.  64. 135.   0.] 113.0

# 📁 Files
birth_weight_predictor.py: Main script
babies.csv: Input dataset (should be placed in the same directory)
model.pickle: Serialized model saved with the highest accuracy (> 77% acc)

# 🚀 Future Improvements
Add data visualization (e.g. smoke effect on bwt)
Evaluate using MAE/RMSE metrics
Try other regression models (e.g., Ridge, Lasso)
Add a web or CLI interface for real-time predictions

# 📜 License
This project is licensed under the MIT License.
