# Chronic Disease Management System
## Table of Contents
1. [Project Overview](#project-verview)
2. [Problem Statement](#problem-statement)
3. [Features of the Dataset](#features-of-the-dataset)
4. [Dataset](#dataset)
5. [Discussion of Findings](#discussion-of-findings)
6. [Analysis of the Models' Results](#analysis-of-the-models'-results)
7. [Machine Learning Algorithm (SVM) vs Neural Network](#machine-learning-algorithm-svm-vs-neural-network)
8. [Error Analysis and Hyperparameter Impact](#error-analysis-and-hyperparameter-impact)
9. [Critical Analysis of Optimization Techniques](#critical-analysis-of-optimization-techniques)
10. [Conclusion](#conclusion)
11. [Running the Notebook and Loading the Best Model](#running-the-notebook-and-loading-the-best-model)
12. [References](#references)

## Project Overview
This project looks at how to predict the severity of cardiovascular disease (CVD) in Uganda's healthcare system, especially in rural areas where people often have a hard time getting healthcare.The project puts CVD severity into three levels: low-risk, medium-risk, and high-risk.It looks at medical factors as well as other things like local beliefs, how far away healthcare facilities are, and how well patients follow treatment plans. By using machine learning and deep learning models, the goal is to improve early detection, communication between patients and doctors, and the use of healthcare resources across Uganda.

## Problem Statement
In Africa's rural areas, heart disease, diabetes, lung disease, and cancer are the main causes of death. The lack of access to healthcare in these areas is making the problem worse.Traditional ways of dealing with these diseases often ignore the social and cultural factors and are based on incomplete or unclear sources.Also, the growth of AI in healthcare is facing security issues and inaccurate predictions. This project aims to develop a system that uses machine learning and deep learning technologies to improve early detection, resource usage, and patient-doctor communication in Uganda's healthcare system.

## Features of the Dataset
The dataset features a mix of structured and unstructured data tailored to Uganda's healthcare context:

- Age (continuous)
- Gender (categorical)
- Blood Pressure (continuous)
- Cholesterol Levels (continuous)
- Medical History (categorical)
- Cultural Belief Scores (categorical)
- Distance to Hospitals (categorized into near, moderate, far)
- Treatment Adherence (graded scale: low, medium, high)
- Use of Traditional Healers (categorical)
- Religious Adherence Differences (categorical)

These features allow the model to account for sociocultural influences, treatment behaviors, and healthcare access disparities that are critical in Uganda.

## Dataset
The dataset used in this project consists of both structured and unstructured data.
1. **Structured data**: This includes numbers and categories like age, gender, blood pressure, cholesterol levels, and medical history. It is used to first sort things into groups and make predictions. The Cardiovascular Disease Dataset is used to predict heart disease and is available on [Kaggle's Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
2. **Unstructured data**: This consists of fake patient records that look like real ones. These unstructured notes give more information about the patient's symptoms and treatment.Here are some examples of notes:
   - "Patient shows high blood pressure, high cholesterol, and obesity-related risks. Echocardiogram suggests early-stage rheumatic heart disease."
   - "Mild high blood pressure noted. Controlled blood sugar levels. Moderate adherence to medication."
   - "History of tobacco use. Symptoms indicate potential COPD overlap. Needs lifestyle intervention."
   - "Shortness of breath and fatigue reported. Distance to care facility may impact treatment adherence."

## Discussion of Findings
### Training Instance Table
| **Training Instance** | **Optimizer** | **Regularizer** | **Epochs** | **Early Stopping** | **Number of Layers** | **Learning Rate** | **Accuracy** | **F1 Score** | **Precision** | **Recall** | **Loss** |
|-----------------------|---------------|-----------------|------------|--------------------|----------------------|-------------------|--------------|--------------|---------------|------------|----------|
| **Instance 1** | Default | None | 20 | No | 7 | 0.0 | 0.24 | Class(0): 0.00, Class(1): 0.40, Class(2): 0.06 | Class(0): 0.00, Class(1): 0.26, Class(2): 0.07 | Class(0): 0.00, Class(1): 0.91, Class(2): 0.05 | 1.12 |
| **Instance 2** | Adam | L2 | 150 | Yes | 7 | 0.001 | 0.92 | Class(0): 0.99, Class(1): 0.85, Class(2): 0.77 | Class(0): 0.98, Class(1): 0.85, Class(2): 0.77 | Class(0): 0.99, Class(1): 0.85, Class(2): 0.77 | 0.39 |
| **Instance 3** | SGD | L2 | 500 | Yes | 7 | 0.006 | 0.91 | Class(0): 0.96, Class(1): 0.81, Class(2): 0.84 | Class(0): 0.93, Class(1): 0.93, Class(2): 0.80 | Class(0): 1.00, Class(1): 0.72, Class(2): 0.90 | 0.49 |
| **Instance 4** | RMSprop | L2 | 200 | Yes | 7 | 0.001 | 0.97 | Class(0): 1.00, Class(1): 0.95, Class(2): 0.90 | Class(0): 1.00, Class(1): 0.95, Class(2): 0.90 | Class(0): 1.00, Class(1): 0.95, Class(2): 0.89 | 0.12 |
| **Instance 5** | SVM | C Parameter(1.0) | N/A | No | 1 | N/A | 0.94 | Class(0): 1.00, Class(1): 0.88, Class(2): 0.74 | Class(0): 1.00, Class(1): 0.86, Class(2): 0.77 | Class(0): 1.00, Class(1): 0.89, Class(2): 0.71 | 0.12 |
## Analysis of the Models' Results
Performance of different models and configurations was judged by testing a variety of techniques of optimization, machine learning algorithms, and neural network architectures. The main metrics, such as accuracy, F1 score, precision, recall, and loss of every instance, were analyzed. The report shows the analysis of the results.
### Summary of Results
Instance 4, which was defined by the **RMSprop Optimizer**, included L2 regularization, had 7 layers, a learning rate of 0.001, and early stopping, was the one that stood out in the comparison table. This particular installation gave excellent results in many aspects, so it was declared the best performing neural network. Specifically, its accuracy rate is **0.97**, so it is likely that the model correctly predicted 97% of the 100 test samples - instances of the class determinant are truly positive.

Although the F1 scores were always known to be at the top, the improvement was evident in classes **0**, **1**, and **2**, where the scores were **1.00**, **0.95**, and **0.90**, respectively. Similarly, the objects with a score above 0 also achieved a good score with scores of 1.00, 0.95, and 0.90. This means that the model was very accurate in identifying positive instances in each class. The scores were also impressive, with **class 0** achieving a score of **1.00**, **class 1** scoring **0.95**, and **class 2** receiving **0.89**. These results suggest a strong balance between sensitivity and specificity at the point when the model was finalized.

In addition, the given configuration hardly lost any information (0.12), so the predictions made by the model were close to the actual values.

It can be said that this great result is due to the combination of **RMSprop's adaptive learning rate**, **L2 regularization**, and **early stopping**, which had an impact on the model's ability to generalize well to new data. This configuration prevented overfitting, allowing the model to generalize effectively. This configuration performed well on the training data and significantly outperformed other models when applied to the validation set.
### Machine Learning Algorithm(SVM) vs Neural Network
When comparing the **SVM model (Instance 5)** with the top-performing neural network **(Instance 4)**, the SVM delivered solid results, achieving:

- **Accuracy**: 0.94
- **F1 Score**: Class(0): 1.00, Class(1): 0.88, Class(2): 0.74
- **Precision**: Class(0): 1.00, Class(1): 0.86, Class(2): 0.77
- **Recall**: Class(0): 1.00, Class(1): 0.89, Class(2): 0.71
- **Loss**: 0.12
While the SVM model performed impressively, the neural network (Instance 4) outperformed it, particularly in predicting minority classes, with higher F1-scores and recall values. The SVM's C parameter (1.0) helped balance margin hardness, offering a good trade-off between precision and generalization. However, the neural network’s multi-layer structure allowed it to better capture complex relationships between features, resulting in superior overall performance.
### Error Analysis and Hyperparameter Impact
Instance 1, with the default optimizer, no regularization, only 5 layers, and no early stopping, faced the challenge of significant underfitting, getting the accuracy as low as 0.24 and the loss as high as 1.12. In the absence of the regularization process and due to the relatively small build, the model found it difficult to infer any complex relationships and as a result was worse off.

Moreover, Instance 2 was able to be drug out of its slump by using, in particular, the Adam optimizer, L2 regularization, and early stopping, along with which the model had superior performance achieving an accuracy of 0.92. Adam’s adaptive learning rate improved the quality of the model globally, and L2 regularization prevented it going astray due to overfitting. However, the F1-score for class 2 (0.77) pointed out that there was likely some space for additional maximizing in predicting minority classes.

Another different approach, Instance 3, used the SGD optimizer, L2 regularization, and a learning rate of 0.006, nevertheless it was the model with slightly lower accuracy (0.91) than the Instance 2. The higher learning rate resulted in more chaos during training, as indicated by a higher loss (0.49). On the other hand, the recall for class 2 (0.90) was strikingly better, showing improvement in positive classification rates among minority classes.

Besides the decline of Adam and SGD, the combination of RMSprop’s adaptive learning rate along with the fine-tuning of L2 regularization and early stopping mechanisms outweighed the influence of the two aforementioned approaches on the model and brought about the more optimal performance of Instance 4.

Instance 5, which is SVM with a C parameter of 1.0, had an accuracy of 0.94, still lacking in the recall area for class 2 (0.71) that was at the lower end of the Instances. This means that notwithstanding the SVM model’s generally good performance, it may have had more problems with class distribution than the deep neural networks, which appear to be outstanding in such sensitive issue.
### Critical Analysis of Optimization Techniques
- It was proven that the Early Stopping was a significant contributor to the deep learning models and improved the generalization performance of the network. Through the early halting of training when the model reached the plateau of performance, this method prevented overfitting which was more visible in all of the cases.

- L2 regularization, particularly, was very essential for keeping the trade-off between the bias and variance. For the models that did not have this capability, e.g., in the case of Instance 1, they did either a lot of overfitting or underfitting. L2 regularization helped to avoid the models from being too complicated, which resulted in better generalization and less overfitting risk, thereby. 

- RMSprop was the optimizer that came out to be the best out of all the others, as I used it in Instance 4. Its dynamic setting of the learning rates of each single parameter was the reason that the learning was stable and quick, and the model was able to gain the best results. Furthermore, Adam (Instance 2) was also highly effective, except for its slightly decreased recall with class 2, which made it too difficult to deal with minority class bettering. Meanwhile, SGD (Instance 3) was more influenced by the learning rate, which was responsible for instability in the model's functioning, however, it had also got a better recall for the minority classes.
## Conclusion
Techniques like Early Stopping, L2 regularization, and optimized learning rates were crucial in improving how well the model performed. RMSprop performed better than other optimizers because it handled complex data more efficiently. While the SVM model showed strong results, the deep neural network, especially Instance 4, excelled at predicting minority classes. The combination of these strategies led to the best-performing model, emphasizing the importance of thoughtful tuning and optimization of hyperparameters in machine learning.
## Running the Notebook and Loading the Best Model
### Prerequisites
- Ensure you have a Google account.
- Access the [Google Colab](GoogleColab) environment.
- Upload the dataset and the saved best model (```model4.keras```).
### Running the Notebook
1. Open the notebook in Google Colab.
2. Ensure all necessary dependencies (tensorFlow, scikit-learn, numpy, matplotlib, pandas, joblib) are installed:
   <pre> ```python
   !pip install tensorflow scikit-learn seaborn matplotlib joblib
   ``` </pre>
4. Run cells sequentially for data preprocessing, model training, and evaluation.
### Loading the best model
```python
from tensorflow.keras.models import load_model

# Load the best saved model
best_model = load_model('best_model.keras')

# Evaluate on test data
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Best Model Accuracy: {accuracy:.2f}")
```

## References
- [Academic literature on Uganda's healthcare challenges and sociocultural factors affecting CVD outcomes](https://link.springer.com/article/10.1186/1744-8603-5-10).
- [Kaggle: Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
