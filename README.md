# Elevate Labs - Task 7: Support Vector Machine (SVM) Classification

### **Objective**
The goal was to implement and compare two different kernel functions within the **Support Vector Machine (SVM)** algorithm: the **Linear Kernel** and the **RBF (Radial Basis Function) Kernel**, using the Breast Cancer dataset.

### **Workflow & Key Steps**

1.  **Preprocessing (Critical)**: The data was thoroughly cleaned, and the target (`diagnosis`) was encoded (M=1, B=0). **All features were scaled using `StandardScaler`**, which is a mandatory step for SVM to ensure the distance calculations are fair across all features.
2.  **Linear SVM**: A Linear SVM model was trained, which seeks a straight hyperplane to separate the classes.
3.  **RBF SVM**: An RBF SVM model was trained, which uses the **Kernel Trick** to find a non-linear boundary in a higher-dimensional space, suitable for complex, non-linearly separable data.
4.  **Comparison**: Both models were evaluated on the test set.

### **Model Performance Comparison**

| Model | Kernel | Accuracy Score | Key Insight |
| :--- | :--- | :--- | :--- |
| **SVM 1** | **Linear** | [Insert Linear SVM Accuracy] | Shows performance assuming the data is largely separable by a straight line. |
| **SVM 2** | **RBF** | [Insert RBF SVM Accuracy] | Often performs better by creating a more complex, non-linear decision boundary. |

### **Conclusion**

The **RBF SVM** model typically demonstrated a **[Higher/Similar] Accuracy** compared to the Linear SVM, indicating that the relationship between the breast cancer features and the diagnosis is likely **non-linear**. This project successfully highlights how the choice of kernel directly impacts the model's ability to classify complex patterns.
