#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("final_medical_data.csv")



# In[2]:


# Define medical advice dictionary
medical_conditions = {
    'flu': 'For flu, rest and hydration are essential. Consider over-the-counter medications for fever and discomfort.',
    'cold': 'For a cold, stay hydrated and use decongestants. Rest is also important.',
    'headache': 'Headaches can be treated with pain relievers and rest. If persistent, consult a healthcare provider.',
    'diabetes': 'Manage diabetes with a balanced diet, regular exercise, and medication as prescribed by your doctor.',
    'hypertension': 'Monitor blood pressure, reduce salt intake, and follow prescribed medications.',
    'asthma': 'Use prescribed inhalers and avoid triggers. Regular check-ups are important.',
    'back pain': 'Rest, proper posture, and physical therapy can help. Consult a specialist if pain persists.',
    'allergy': 'Identify and avoid allergens. Over-the-counter antihistamines can help alleviate symptoms.',
    'infection': 'Follow prescribed antibiotics and complete the full course. Rest and hydration are important.',
    'anxiety': 'Practice relaxation techniques and seek therapy. Medication may be prescribed by a doctor.',
    'depression': 'Consult a mental health professional. Therapy, lifestyle changes, and medication can help.',
    'arthritis': 'Manage arthritis with anti-inflammatory medications, physical therapy, and joint care.',
    'migraine': 'Avoid triggers, use pain relievers, and rest in a dark, quiet room. Consult a specialist for chronic migraines.',
    'heart disease': 'Follow a heart-healthy diet, exercise regularly, and take prescribed medications.',
    'obesity': 'Adopt a healthy diet, increase physical activity, and consider medical interventions if necessary.',
    'stroke': 'Emergency treatment is critical. Rehabilitation and lifestyle changes can aid recovery and prevention.',
    'chronic fatigue syndrome': 'Manage symptoms with a balanced routine, rest, and stress reduction techniques.',
    'gastroesophageal reflux disease (gerd)': 'Avoid trigger foods, eat smaller meals, and consider medications for acid reduction.',
    'irritable bowel syndrome (ibs)': 'Manage IBS with dietary changes, stress management, and prescribed medications.',
    'skin rash': 'Use topical creams and avoid irritants. Consult a dermatologist if the rash persists.',
    'insomnia': 'Establish a regular sleep schedule, avoid caffeine, and seek medical advice for chronic insomnia.',
    'osteoporosis': 'Ensure adequate calcium and vitamin D intake, and engage in weight-bearing exercises. Medications may be required.',
    'kidney stones': 'Stay hydrated and follow a low-sodium diet. Pain management and medical intervention may be necessary.',
    'eczema': 'Moisturize regularly and use prescribed creams. Avoid irritants and allergens.',
    'psoriasis': 'Use prescribed topical treatments and avoid triggers like stress. Consult a dermatologist for severe cases.',
    'pneumonia': 'Take prescribed antibiotics, rest, and stay hydrated. Hospitalization may be needed in severe cases.',
    'anemia': 'Treat anemia with iron supplements and a balanced diet. Seek medical advice for underlying causes.',
    'hypothyroidism': 'Take prescribed thyroid hormone replacement and have regular blood tests to monitor levels.',
    'parkinson’s disease': 'Manage symptoms with medication, physical therapy, and regular doctor visits.'
}

# Clean and preprocess the data
data = data.iloc[:, :7]
data.dropna(inplace=True)
data.columns = data.columns.str.strip()  # Remove any extra spaces from column names
data.rename(columns={'Medical condition': 'Medical_condition'}, inplace=True)
data['Medical_condition'] = data['Medical_condition'].str.lower()

# Encode labels
label_encoder = LabelEncoder()
data['Condition_Label'] = label_encoder.fit_transform(data['Medical_condition'])

# Preprocess conditions for NLP
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])

data['Processed_Condition'] = data['Medical_condition'].apply(preprocess_text)

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['Processed_Condition'])
sequences = tokenizer.texts_to_sequences(data['Processed_Condition'])
padded_sequences = pad_sequences(sequences, maxlen=10)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['Condition_Label'], test_size=0.2, random_state=42)
num_classes = len(data['Condition_Label'].unique())

# Build the NLP model
model = Sequential([
    Embedding(5000, 128, input_length=10),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))




# In[15]:


import time
total_queries = 0
total_response_time = 0
correct_predictions = 0 

data.rename(columns={'Medical condition': 'Medical_condition', 'Date of admission': 'Date_of_Admission', 'Appointment time': 'Appointment_Time'}, inplace=True)
data['Medical_condition'] = data['Medical_condition'].str.lower()

# Convert Date and Time to consistent formats
data['Date_of_Admission'] = pd.to_datetime(data['Date_of_Admission'], errors='coerce').dt.date
data['Appointment_Time'] = pd.to_datetime(data['Appointment_Time'], format='%I:%M %p', errors='coerce').dt.strftime('%I:%M %p')

# Save a DataFrame to track patient records
df = data.copy()
df1=data.copy()

# Function to get medical advice based on condition
def get_advice(condition):
    condition = condition.lower()  # Ensure condition is lowercase for matching
    return medical_conditions.get(condition, "Consult your doctor for further assistance.")
doctor_specializations = {
    'Dr. Silva': ['flu', 'cold', 'infection'],
    'Dr. Chang': ['diabetes', 'hypertension'],
    'Dr. Green': ['headache', 'migraine'],
    'Dr. Richard': ['depression', 'anxiety'],
    'Dr. Bowers': ['arthritis', 'back pain'],
    'Dr. Dixon': ['asthma', 'allergy'],
    'Dr. Woods': ['heart disease', 'obesity', 'stroke'],
    'Dr. Sullivan': ['chronic fatigue syndrome'],
    'Dr. Turner': ['gastroesophageal reflux disease (gerd)', 'irritable bowel syndrome (ibs)'],
    'Dr. Fisher': ['skin rash', 'eczema', 'psoriasis'],
    'Dr. Christopher': ['pneumonia'],
    'Dr. Sarah': ['anemia', 'hypothyroidism'],
    'Dr. James': ['insomnia'],
    'Dr. Robert': ['osteoporosis', 'kidney stones'],
    'Dr. Amanda': ['parkinson’s disease'],
    'Dr. Stephanie': ['allergy', 'infection'],
    'Dr. Emily': ['anxiety', 'depression'],
    'Dr. Melborn': ['flu', 'cold'],
    'Dr. Charles': ['headache', 'migraine'],
    'Dr. Melissa': ['skin rash', 'eczema'],
    'Dr. Ashley': ['infection', 'anemia'],
    'Dr. John': ['heart disease', 'stroke'],
    'Dr. Shiva': ['arthritis', 'back pain']
}
# Main loop for user interaction
while True:
    print("\nPlease choose an option:")
    print("1. Get medical advice")
    print("2. Schedule an appointment")
    print("3. Manage patient records")
    print("4. Exit")
    choice = input("Enter your choice (1/2/3/4): ")

    if choice == '1':
        user_input = input("Enter your medical condition or symptoms: ")

        start_time = time.time()  # Start timing for response time

        # Preprocess user input
        processed_input = preprocess_text(user_input)
        input_seq = tokenizer.texts_to_sequences([processed_input])
        input_padded = pad_sequences(input_seq, maxlen=10)

        # Predict the condition label
        predicted_label = np.argmax(model.predict(input_padded), axis=-1)
        predicted_condition = label_encoder.inverse_transform(predicted_label)[0]
        actual_condition = data.loc[data['Medical_condition'] == processed_input, 'Medical_condition'].values[0]

        if predicted_condition.lower() == actual_condition.lower():
            correct_predictions += 1


        # Display the predicted condition and the advice
        print(f"Predicted Condition: {predicted_condition.capitalize()}")
        print("Advice:", get_advice(predicted_condition))

        end_time = time.time()  # End timing
        total_queries += 1
        total_response_time += (end_time - start_time)

    elif choice == '2':
        # Scheduling an appointment
        name = input("Enter patient name: ")
        age = int(input("Enter the age: "))
        gender = input("Enter the gender of the patient: ")
        print("\nAvailable Doctors and Their Specializations:")
        for doctor, specialties in doctor_specializations.items():
            print(f"{doctor}: {', '.join(specialties)}")
        doctor = input("Enter doctor's name: ").strip()
        appointment_time = input("Enter appointment time (HH:MM AM/PM): ")
        date_of_admission = input("Enter date of admission (YYYY-MM-DD): ")
        medical_condition = input("Enter the medical condition: ")

        # Convert input to match DataFrame formats
        try:
            date_of_admission = pd.to_datetime(date_of_admission).date()
            appointment_time = pd.to_datetime(appointment_time, format='%I:%M %p').strftime('%I:%M %p')
        except ValueError:
            print("Invalid date or time format. Please try again.")
            continue

        # Check if the doctor is available at the requested time and date
        appointment_conflict = df1[(df1['Doctor'].str.lower() == doctor.lower()) &
                                  (df1['Appointment_Time'] == appointment_time) &
                                  (df1['Date_of_Admission'] == date_of_admission)]

        if not appointment_conflict.empty:
            print(f"Doctor {doctor} is unavailable at {appointment_time} on {date_of_admission}. Appointment cannot be scheduled. Please choose another time.")
        else:
            # Add appointment to DataFrame if available
             # Make sure to copy df to df1
            df1=df1.iloc[:,:7]
            new_appointment = pd.DataFrame([[name, age, gender, doctor, appointment_time, date_of_admission, medical_condition]],
                                           columns=['Name', 'Age', 'Gender', 'Doctor', 'Appointment_Time', 'Date_of_Admission', 'Medical_condition'])
            df1 = pd.concat([df1, new_appointment], ignore_index=True)
            
            
            print(f"Appointment successfully scheduled for {name} with Dr. {doctor} on {date_of_admission} at {appointment_time}.")

    elif choice == '3':
        # Simple record management
        name = input("Enter patient name to find records: ")
        records = df1[df1['Name'].str.lower() == name.lower()]
        if not records.empty:
            print(records)
        else:
            print("No records found for this name.")

    elif choice == '4':
        print("Exiting the assistant.")
        if total_queries > 0:
            # Calculate total accuracy before exiting
            total_accuracy = (correct_predictions / total_queries) * 100
            print(f"Total Accuracy of the Model: {total_accuracy:.2f}%")

            # Calculate average response time
            avg_response_time = total_response_time / total_queries
            print(f"Average Response Time: {avg_response_time:.4f} seconds")
        else:
            print("No queries were made, so no accuracy can be calculated.")
        print("Have a great day!")
        break

    else:
        print("Invalid choice, please enter a number between 1 and 4.")


# In[18]:


# Plot accuracy and loss graphs
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[27]:


# Accuracy over epochs (Training vs Validation Accuracy)
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[31]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[33]:


# Creating a dashboard with accuracy and response time
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Accuracy over epochs
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy Over Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend(loc='best')
axes[0].grid(True)

# Right plot: Response time histogram
axes[1].hist(response_times, bins=20, color='skyblue', edgecolor='black')
axes[1].set_title('Distribution of Response Times')
axes[1].set_xlabel('Response Time (seconds)')
axes[1].set_ylabel('Frequency')
axes[1].grid(True)

plt.tight_layout()
plt.show()


# In[36]:


# Plot loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[37]:


plt.figure(figsize=(10, 6))

# Creating the countplot
sns.countplot(data=data, x='Medical_condition', order=data['Medical_condition'].value_counts().index)

# Rotating the x-axis labels for better readability
plt.xticks(rotation=45)

# Adding title and labels
plt.title("Frequency of Medical Conditions")
plt.xlabel("Medical Condition")
plt.ylabel("Count")

# Display the plot
plt.show()


# In[38]:


plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[48]:


plt.figure(figsize=(8, 5))

# Creating the countplot
sns.countplot(x='Gender', data=data)

# Adding title and labels
plt.title("Gender Distribution of Patients")
plt.xlabel("Gender")
plt.ylabel("Count")

# Display the plot
plt.show()


# In[55]:


plt.figure(figsize=(10, 6))

# Create count plot with 'Medical_condition' column
sns.countplot(x='Medical_condition', data=data)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add title and axis labels
plt.title("Frequency of Medical Conditions")
plt.xlabel("Condition")
plt.ylabel("Count")

# Display the plot
plt.show()


# In[58]:


# Convert 'Appointment time' to datetime (make sure the format matches your data)
data['Appointment_hour'] = pd.to_datetime(data['Appointment_Time'], format='%I:%M %p').dt.hour

# Plotting the distribution of appointment times
plt.figure(figsize=(10, 6))
sns.histplot(data['Appointment_hour'], bins=24, kde=True)
plt.title("Distribution of Appointment Times")
plt.xlabel("Hour of Day")
plt.ylabel("Frequency")
plt.show()


# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'Appointment time' is in the correct format
data['Appointment_hour'] = pd.to_datetime(data['Appointment_Time'], errors='coerce').dt.hour

# Clean up Doctor column if necessary
data['Doctor'] = data['Doctor'].str.strip()  # Remove leading/trailing spaces

# Pivot table to get the count of appointments per hour for each doctor
appointment_heatmap_data = data.pivot_table(index='Doctor', columns='Appointment_hour', aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(appointment_heatmap_data, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Heatmap of Appointment Times by Doctor")
plt.xlabel("Hour of Day")
plt.ylabel("Doctor")
plt.show()


# In[62]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for age distribution by medical condition
plt.figure(figsize=(12, 6))
sns.boxplot(x='Medical_condition', y='Age', data=data)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Set the title and labels for the axes
plt.title("Age Distribution for Each Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Age")

# Show the plot
plt.show()


# In[63]:


# Filter data for a specific condition, e.g., "Asthma"
condition_data = data[data['Medical_condition'] == 'asthma']

# Count the gender distribution
gender_counts = condition_data['Gender'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Gender Distribution for Asthma Condition")
plt.show()


# In[64]:


plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='Appointment_hour', hue='Medical_condition', multiple='stack', bins=24)
plt.title("Appointment Times by Medical Condition")
plt.xlabel("Hour of Day")
plt.ylabel("Frequency")
plt.show()


# In[65]:


# Define age groups
bins = [0, 18, 35, 50, 65, 100]
labels = ['0-18', '19-35', '36-50', '51-65', '65+']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)

# Create the countplot
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Age_Group', hue='Medical_condition')
plt.title("Count of Medical Conditions by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Medical Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[ ]:




