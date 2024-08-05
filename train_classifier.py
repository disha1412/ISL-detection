import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Determine the maximum sequence length
max_length = max(len(sequence) for sequence in data)

# Pad sequences to the same length
data_padded = np.array([np.pad(sequence, (0, max_length - len(sequence)), 'constant') for sequence in data])

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Normalize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model and padding length
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder, 'scaler': scaler, 'max_length': max_length}, f)
