# Data Science & Machine Learning Implementation Notebooks

This repository contains foundational Python implementations covering data manipulation with **NumPy**, data processing via **Pandas**, and fundamental machine learning algorithms including **Linear Regression**, **k-Nearest Neighbors (KNN)**, and **Naive Bayes**.

---

## 📌 NumPy Essentials

The included Jupyter notebook provides a hands-on introduction to vectorization, array manipulation, and linear algebra operations using **NumPy** (`v2.4.6`).

### Core Concepts & Code Snippets

#### 1. Array Creation & Type Coercion
NumPy arrays enforce homogeneous data types. Mixed-type inputs automatically upcast elements (e.g., integers and strings cast to Unicode string types `<U21`):

```python
import numpy as np

# Implicit string upcasting
arr_mixed = np.array([1, 2, 3, "hello"])
print(type(arr_mixed))  # <class 'numpy.ndarray'>
print(arr_mixed.shape)  # (4,)

# Initializing zero matrices
zeros_matrix = np.zeros((3, 4))

n1 = np.array([3, 4, 5])
n2 = np.array([6, 7, 8])

# Element-wise multiplication -> [18, 28, 40]
result = n1 * n2

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Basic slicing
print(arr[0:4])  # Output: [1, 2, 3, 4]

# Boolean filtering
gt_three = arr[arr > 3]  # Output: [4, 5, 6, 7, 8, 9]

arr = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
reshaped_arr = arr.reshape(2, 5)

> thank you
