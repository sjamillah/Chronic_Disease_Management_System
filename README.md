# Chronic Disease Management System
## Problem Statement
In Africa's rural areas, heart disease, diabetes, lung disease, and cancer are the main causes of death. The lack of access to healthcare in these areas is making the problem worse.Traditional ways of dealing with these diseases often ignore the social and cultural factors and are based on incomplete or unclear sources.Also, the growth of AI in healthcare is facing security issues and inaccurate predictions. This project aims to develop a system that uses machine learning and deep learning technologies to improve early detection, resource usage, and patient-doctor communication in Uganda's healthcare system.
## Dataset
The dataset used in this project consists of both structured and unstructured data.
1. **Structured data**: This includes numbers and categories like age, gender, blood pressure, cholesterol levels, and medical history. It is used to first sort things into groups and make predictions. The Cardiovascular Disease Dataset is used to predict heart disease and is available on [Kaggle's Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
2. **Unstructured data**: This consists of fake patient records that look like real ones. These unstructured notes give more information about the patient's symptoms and treatment.Here are some examples of notes:
   - "Patient shows high blood pressure, high cholesterol, and obesity-related risks. Echocardiogram suggests early-stage rheumatic heart disease.
   - ""Mild high blood pressure noted. Controlled blood sugar levels. Moderate adherence to medication.
   - ""History of tobacco use. Symptoms indicate potential COPD overlap. Needs lifestyle intervention."
   - "Shortness of breath and fatigue reported. Distance to care facility may impact treatment adherence."
## Discussion of Findings
### Training Instance Table
| **Training Instance** | **Optimizer** | **Regularizer** | **Epochs** | **Early Stopping** | **Number of Layers** | **Learning Rate** | **Accuracy** | **F1 Score** | **Precision** | **Recall** | **Loss** |
| **Instance 1** | Default | None | 20 | No | 5 | 0.0 | 0.24 | Class(0): 0.00, Class(1): 0.42, Class(2): 0.08 | Class(0): 1.00, Class(1): 0.28, Class(2): 0.07 | Class(0): 0.00, Class(1): 0.89, Class(2): 0.09 | 1.21 |
| **Instance 2** | Adam | L2 | 150 | Yes | 7 | 0.001 | 0.88 | Class(0): 0.97, Class(1): 0.78, Class(2): 0.57 | Class(0): 0.95, Class(1): 0.75, Class(2): 0.77 | Class(0): 0.99, Class(1): 0.82, Class(2): 0.45 | 0.39 |
| **Instance 3** | SGD | L2 | 500 | Yes | 7 | 0.006 | 0.92 | Class(0): 0.96, Class(1): 0.82, Class(2): 0.85 | Class(0): 0.93, Class(1): 0.93, Class(2): 0.82 | Class(0): 1.00, Class(1): 0.74, Class(2): 0.90 | 0.44 |
| **Instance 4** | RMSprop | L2 | 200 | Yes | 7 | 0.001 | 0.98 | Class(0): 1.00, Class(1): 0.97, Class(2): 0.94 | Class(0): 1.00, Class(1): 0.97, Class(2): 0.94 | Class(0): 1.00, Class(1): 0.96, Class(2): 0.94 | 0.14 |
