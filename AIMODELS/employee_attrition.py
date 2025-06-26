# KNN Classification for Employee Attrition Prediction
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Columns: Age, JobRole, MonthlyIncome, JobSatisfaction, YearsAtCompany, Attrition
data = np.array([
    [29, "Sales Executive", 4800, 3, 4, 1],
    [35, "Research Scientist", 6000, 4, 8, 0],
    [40, "Laboratory Technician", 3400, 2, 6, 0],
    [28, "Sales Executive", 4300, 3, 3, 1],
    [45, "Manager", 11000, 4, 15, 0],
    [25, "Research Scientist", 3500, 1, 2, 1],
    [50, "Manager", 12000, 4, 20, 0],
    [30, "Sales Executive", 5000, 2, 5, 0],
    [37, "Laboratory Technician", 3100, 2, 9, 0],
    [26, "Research Scientist", 4500, 3, 2, 1]
])

# Step 2: Encoding categorical features (JobRole) to numerical values suitable for KNN
job_roles = data[:, 1]  # Extract JobRole column
label_encoder = LabelEncoder()
encoded_job_roles = label_encoder.fit_transform(job_roles)
data[:, 1] = encoded_job_roles  # Replace JobRole with encoded values

# Splitting the data into input features and output labels
inputx = data[:, 0:5].astype(float)  # Features: Age, JobRole, MonthlyIncome, JobSatisfaction, YearsAtCompany
outputy = data[:, 5].astype(int)     # Labels: Attrition

# Step 3: Selecting the KNN model
thismodel = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
print("\nThe model selected is", thismodel)
print("\nThe parameters of the model are\n\n", thismodel.get_params())

# Step 4: Training the model
thismodel.fit(inputx, outputy)

# Step 5: Testing and model prediction
test_data = np.array([
    [32, "Sales Executive", 5200, 3, 6],
    [42, "Manager", 11500, 4, 18],
    [27, "Research Scientist", 4000, 2, 3],
    [38, "Laboratory Technician", 3500, 2, 7]
])

# Encoding JobRole in test data
test_data[:, 1] = label_encoder.transform(test_data[:, 1])  # Encode JobRole
test_data = test_data.astype(float)

print("\n\nThe test inputs are\n\n", test_data)
res = thismodel.predict(test_data)

# Step 6: Visualizing the test results
reslist = []
for val in res:
    if val == 0:
        reslist.append("No Attrition")
    else:
        reslist.append("Attrition")
print("\nThe test results are\n\n", reslist)