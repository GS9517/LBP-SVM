import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import dump, load
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# parameter settings
dataset_dir = 'Aerial_Landscapes'
categories = ['Agriculture', 'Airport', 'Beach', 'City', 'Desert', 'Forest', 'Grassland',
              'Highway', 'Lake', 'Mountain', 'Parking', 'Port', 'Railway', 'Residential', 'River']

lbp_radius = 1
lbp_n_points = 8 * lbp_radius
lbp_method = 'uniform'
resize_size = (128, 128)

# extract LBP features
def extract_lbp_features(img_gray):
    lbp = local_binary_pattern(img_gray, lbp_n_points, lbp_radius, lbp_method)
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, lbp_n_points + 3),
                           range=(0, lbp_n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

# load image data
features = []
labels = []

for category in tqdm(categories):
    folder_path = os.path.join(dataset_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, resize_size)
        lbp_feature = extract_lbp_features(img)
        features.append(lbp_feature)
        labels.append(category)

# convert to a NumPy array
X = np.array(features)
le = LabelEncoder()
y = le.fit_transform(labels)

# split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train, y_train)

# save the model and label encoder
dump(clf, 'lbp_svm_model.joblib')
dump(le, 'label_encoder.joblib')
print("Model saved as lbp_svm_model.joblib and label_encoder.joblib")

# model evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# plot a bar chart of accuracy per class
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 5))
sns.barplot(x=le.classes_, y=class_accuracy)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.xticks(rotation=45)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("per_class_accuracy.png")
plt.show()
