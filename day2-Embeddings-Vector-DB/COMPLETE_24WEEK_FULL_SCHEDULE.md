# COMPLETE 24-WEEK INTEGRATED LEARNING SCHEDULE
## ULTIMATE VERSION: Everything Combined
## Your Schedule + DeepLearning.AI + Patrick Loeber + Zero to Mastery + Calculus + Gradient Descent

**Total Investment:** 25-30 hours/week
**Duration:** 24 weeks (6 months intensive)
**Cost:** $0-250 total
**Outcome:** Complete ML/AI mastery with production-ready skills
**Total Hours:** 397 hours of professional education

---

# TABLE OF CONTENTS

1. Overview & Quick Summary
2. Weeks 1-8: Linear Algebra Foundations
3. Week 9.5: Calculus Fundamentals
4. Weeks 10-11: Gradient Descent & Optimization
5. Weeks 12-24: Applications & Capstones
6. Complete Resource Directory
7. Time Summary Table

---

# QUICK OVERVIEW

**Why 24 weeks instead of 16?**
- More time to deeply practice concepts
- Deeper understanding (not rushed)
- Substantial portfolio projects
- Professional-grade capstone projects
- Foundation → Math → Optimization → Applications → Capstone

**Perfect Learning Progression:**
```
Weeks 1-8:    Linear Algebra (vectors, matrices, eigenvalues, SVD)
Week 9.5:     Calculus (derivatives, gradients, optimization)
Weeks 10-11:  Gradient Descent & Algorithms
Week 12:      ML Applications
Weeks 13-16:  Capstone Projects & Advanced
Weeks 17-24:  Extended Specialization (optional)
```

---

# WEEKS 1-8: LINEAR ALGEBRA FOUNDATIONS

---

# WEEK 1: VECTORS & NUMPY BASICS

## YOUR SCHEDULE (3 hours)

**3Blue1Brown Episodes 1-2**
- Episode 1 - Vectors: https://www.youtube.com/watch?v=fNk_zzaMoSY (10 min)
- Episode 2 - Linear combinations: https://www.youtube.com/watch?v=k7RM-ot2NWY (13 min)

**Topics:**
- What is a vector?
- Vector addition & subtraction
- Scalar multiplication
- Basis vectors (i, j, k)
- Linear combinations
- Span

## DEEPLEARNING.AI (3 hours)

**Platform:** https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science

**Course 1: Linear Algebra for Machine Learning (Week 1)**
- Vectors as data representations
- Vector magnitude and direction
- **Python lab:** Create vectors in NumPy
- **Application:** Feature vectors in ML

**Action:**
- Watch video lectures (1.5 hours)
- Complete interactive labs (1.5 hours)

## NUMPY CODING (4 hours)

**Resource Links:**
- NumPy Quickstart: https://numpy.org/doc/stable/user/quickstart.html
- W3Schools NumPy: https://www.w3schools.com/python/numpy/
- YouTube Tutorial: https://www.youtube.com/watch?v=9JUAPWk_IFU

**Exercise 1: Create and Inspect Vectors (45 min)**
```python
import numpy as np

v1 = np.array([3, 4])
v2 = np.array([1, 2])
v3 = np.array([1, 2, 3])

print("Vector 1:", v1)
print("Shape:", v1.shape)
print("Dtype:", v1.dtype)

v4 = np.zeros(5)
v5 = np.ones(3)
v6 = np.arange(0, 10, 2)

# Vector operations
v_add = v1 + v2
v_sub = v1 - v2
v_mult = v1 * v2
v_div = v1 / v2

print("v1 + v2 =", v_add)
print("v1 - v2 =", v_sub)
print("v1 * v2 =", v_mult)
print("v1 / v2 =", v_div)

# Save as: week1_exercise1.py
```

**Exercise 2: Vector Properties and Norms (45 min)**
```python
import numpy as np

v1 = np.array([3, 4])
v2 = np.array([1, 2, 3])

# Magnitude
magnitude_v1 = np.linalg.norm(v1)
print(f"Magnitude of v1: {magnitude_v1}")  # Should be 5

# Unit vector
unit_v1 = v1 / magnitude_v1
print(f"Unit vector of v1: {unit_v1}")

# Dot product
v1_3d = np.array([3, 4, 0])
dot_product = np.dot(v1_3d, v2)
print(f"Dot product: {dot_product}")

# Angle between vectors
cos_angle = dot_product / (np.linalg.norm(v1_3d) * np.linalg.norm(v2))
angle_rad = np.arccos(cos_angle)
angle_deg = np.degrees(angle_rad)
print(f"Angle between vectors: {angle_deg} degrees")

# Save as: week1_exercise2.py
```

**Exercise 3: Your Document Example (30 min)**
```python
import numpy as np

doc1_features = np.array([2, 1, 0, 0, 0])  # [dog, barks, cat, meows, play]
doc2_features = np.array([0, 0, 2, 1, 0])
doc3_features = np.array([3, 0, 2, 0, 1])

print("Doc1:", doc1_features)
print("Doc2:", doc2_features)
print("Doc3:", doc3_features)

# Magnitudes
mag1 = np.linalg.norm(doc1_features)
mag2 = np.linalg.norm(doc2_features)
mag3 = np.linalg.norm(doc3_features)

print(f"\nMagnitude Doc1: {mag1}")
print(f"Magnitude Doc2: {mag2}")
print(f"Magnitude Doc3: {mag3}")

# Dot products (similarity)
sim_12 = np.dot(doc1_features, doc2_features)
sim_13 = np.dot(doc1_features, doc3_features)
sim_23 = np.dot(doc2_features, doc3_features)

print(f"\nSimilarity Doc1-Doc2: {sim_12}")  # Should be 0
print(f"Similarity Doc1-Doc3: {sim_13}")  # Should be 6 (YOUR INSIGHT!)
print(f"Similarity Doc2-Doc3: {sim_23}")  # Should be 4

print(f"\nVerification: 2 (dog in Doc1) × 3 (dog in Doc3) = {2*3}")

# Save as: week1_exercise3.py
```

## ASSIGNMENTS (1 hour)

**Assignment 1.1: Vector Operations**
- Create 5 different vectors (various sizes)
- Perform all 4 basic operations (add, sub, mult, div)
- Test with random data

**Assignment 1.2: Properties**
- Calculate magnitude of [3, 4, 5]
- Create unit vector
- Calculate angle between [1, 0, 0] and [1, 1, 1]

**Assignment 1.3: Your Document Example**
- Verify all three similarity calculations
- Explain why each value makes sense
- Add 2 more documents and calculate similarities

## PROJECT (2 hours)

**Build:** Vector Calculator Class

```python
# File: Week1_VectorCalculator.ipynb

class VectorCalculator:
    @staticmethod
    def add(v1, v2):
        return np.array(v1) + np.array(v2)
    
    @staticmethod
    def magnitude(v):
        return np.linalg.norm(v)
    
    @staticmethod
    def dot_product(v1, v2):
        return np.dot(v1, v2)
    
    @staticmethod
    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(cos_angle))

# Test with multiple vectors
# Save and commit to GitHub
```

**TOTAL WEEK 1: 12 hours**

---

# WEEK 2: MATRICES & TRANSFORMATIONS

## YOUR SCHEDULE (3.5 hours)

**3Blue1Brown Episodes 3-4**
- Episode 3 - Linear transformations: https://www.youtube.com/watch?v=kYB8IZa7TiE (13 min)
- Episode 4 - Matrix multiplication: https://www.youtube.com/watch?v=XfCiNki8Knc (10 min)

**Topics:**
- What is a matrix?
- Linear transformations geometrically
- Matrix multiplication
- Composition of transformations
- Determinant
- 2D and 3D examples

## DEEPLEARNING.AI (4 hours)

**Course 1: Linear Algebra (Week 2)**
- Matrix representation of data
- Matrix operations fundamentals
- **Python lab:** NumPy matrix operations
- **Application:** Image as matrix, transformations

## NUMPY CODING (4.5 hours)

**Resource Links:**
- NumPy Matrix Docs: https://numpy.org/doc/stable/reference/arrays.ndarray.html
- GeeksforGeeks: https://www.geeksforgeeks.org/matrix-operations-in-python/

**Exercise 1: Create and Access Matrices (60 min)**
```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [1, 0],
    [0, 1],
    [-1, 2]
])

print("Matrix A shape:", A.shape)
print("Element [0,0]:", A[0, 0])
print("Row 0:", A[0, :])
print("Column 1:", A[:, 1])

zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)
random = np.random.rand(3, 3)

print("\nIdentity matrix:")
print(identity)

# Slicing
print("\nSubmatrix [0:2, 1:3]:")
print(A[0:2, 1:3])

# Reshape
A_flat = A.reshape(1, 9)
A_2d = A.reshape(9, 1)

# Save as: week2_exercise1.py
```

**Exercise 2: Matrix Operations (90 min)**
```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
B = np.array([[1, 0], [0, 1], [-1, 2]], dtype=float)
C = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float)

# Element-wise
print("A + C:")
print(A + C)

print("\nA - C:")
print(A - C)

print("\nA * C (element-wise):")
print(A * C)

# Matrix multiplication (THE IMPORTANT ONE!)
result = A @ B
print("\nA @ B result shape:", result.shape)
print("A @ B:")
print(result)

# Verify manually
print("\nManual verification of [0,0]:")
print("1*1 + 2*0 + 3*(-1) =", 1*1 + 2*0 + 3*(-1))
print("Result [0,0]:", result[0, 0])

# Rotation matrix
theta = np.pi / 4
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

point = np.array([1, 0])
rotated = R @ point
print(f"\nPoint [1, 0] rotated 45°: {rotated}")

# Save as: week2_exercise2.py
```

**Exercise 3: Image as Matrix (60 min)**
```python
import numpy as np

image = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
], dtype=float)

print("Original image:")
print(image)

scaled = image * 0.5
print("\nScaled image (50%):")
print(scaled)

# Rotation example
points = np.array([[0, 1], [1, 1]])
theta = np.pi / 4
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

rotated_points = points @ R.T
print("\nRotated points (45°):")
print(rotated_points)

# Save as: week2_exercise3.py
```

## ASSIGNMENTS (1 hour)

**Assignment 2.1:** Matrix basics
**Assignment 2.2:** Matrix multiplication
**Assignment 2.3:** Transformations

## PROJECT (2.5 hours)

**Build:** Image Transformation System
- Test with sample images
- Visualize before/after
- Save: `Week2_ImageTransformations.ipynb`

**TOTAL WEEK 2: 14 hours**

---

# WEEK 3: TRANSPOSE & DOT PRODUCT

## YOUR SCHEDULE (3 hours)

**3Blue1Brown Episodes 5-7**
- Episode 5 - 3D transformations: https://www.youtube.com/watch?v=v8VSDg_WQlA (14 min)
- Episode 6 - Determinant: https://www.youtube.com/watch?v=Ip3X9LOh2dk (13 min)
- Episode 7 - Inverse matrices: https://www.youtube.com/watch?v=uQhTuRlWMxw (12 min)

**Topics:**
- Transpose: flip rows and columns
- Dot product: measure similarity
- Determinant: scaling factor
- Inverse matrices
- Singular vs invertible matrices

## DEEPLEARNING.AI (4 hours)

**Course 1: Linear Algebra (Week 3)**
- Matrix operations
- Similarity calculations
- **Python lab:** A^T × A calculations
- **Application:** Document similarity (YOUR EXAMPLE!)

## NUMPY CODING (4.5 hours)

**Exercise 1: Transpose Operations (60 min)**
```python
import numpy as np

A = np.array([
    [2, 0, 3],
    [1, 0, 0],
    [0, 2, 2],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

print("Original A shape:", A.shape)
At = A.T
print("Transpose A^T shape:", At.shape)

# Verify (A^T)^T = A
double_transpose = At.T
print("\nVerify (A^T)^T = A:", np.allclose(double_transpose, A))

# Symmetric matrices
S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
print("\nIs S symmetric?", np.allclose(S, S.T))

# Save as: week3_exercise1.py
```

**Exercise 2: Dot Product & Similarity (90 min) - YOUR FOCUS!**
```python
import numpy as np

print("=== YOUR DOCUMENT EXAMPLE ===\n")

# Feature vectors
doc1 = np.array([2, 1, 0, 0, 0], dtype=float)
doc2 = np.array([0, 0, 2, 1, 0], dtype=float)
doc3 = np.array([3, 0, 2, 0, 1], dtype=float)

A = np.array([doc1, doc2, doc3]).T
At = A.T

# Calculate A^T × A
ATA = At @ A
print("Similarity Matrix A^T × A:")
print(ATA)

# YOUR INSIGHT: 2×3=6
print("\n=== YOUR CALCULATION ===")
print("Doc1 vs Doc3 similarity: A^T×A[0,2] =", ATA[0, 2])
print("Explanation: Doc1 has 2 'dog', Doc3 has 3 'dog'")
print("So: 2 × 3 = 6 ✓")

# Verify manually
manual_13 = np.dot(doc1, doc3)
print(f"Manual verification: dot(Doc1, Doc3) = {manual_13}")

# All similarities
print("\n--- All Document Similarities ---")
print(f"Doc1 self-similarity: {ATA[0, 0]}")
print(f"Doc2 self-similarity: {ATA[1, 1]}")
print(f"Doc3 self-similarity: {ATA[2, 2]}")
print(f"Doc1-Doc2: {ATA[0, 1]}")
print(f"Doc1-Doc3: {ATA[0, 2]}")
print(f"Doc2-Doc3: {ATA[1, 2]}")

# Save as: week3_exercise2.py
```

**Exercise 3: Determinant & Inverse (60 min)**
```python
import numpy as np

B = np.array([[1, 2], [3, 4]], dtype=float)
C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

# Determinant
det_B = np.linalg.det(B)
det_C = np.linalg.det(C)

print("Matrix B:")
print(B)
print("Determinant:", det_B)
print("Manual: 1*4 - 2*3 =", 1*4 - 2*3)

print("\nMatrix C:")
print(C)
print("Determinant:", det_C)

# Inverse
if abs(det_B) > 1e-10:
    B_inv = np.linalg.inv(B)
    print("\nB inverse:")
    print(B_inv)
    
    product = B @ B_inv
    identity = np.eye(2)
    print("Is it identity?", np.allclose(product, identity))

# Save as: week3_exercise3.py
```

## ASSIGNMENTS (1 hour)

**Assignment 3.1:** Transpose
**Assignment 3.2:** Dot Product (YOUR FOCUS!)
**Assignment 3.3:** Determinant & Inverse

## PROJECT (2.5 hours)

**Build:** Document Similarity Calculator
- Use your document example
- Add more documents
- Display similarities
- Save: `Week3_DocumentSimilarity.ipynb`

**TOTAL WEEK 3: 14 hours**

---

# WEEK 4: LINEAR SYSTEMS & MIT 18.06

## YOUR SCHEDULE (4 hours)

**MIT OpenCourseWare 18.06**
- Lecture 1 - Geometry: https://www.youtube.com/watch?v=J7DzL2_Na80 (1h 20min)
- Lecture 2 - Elimination: https://www.youtube.com/watch?v=QVKj3LADCnA (1h 10min)
- Lecture 3 - Multiplication: https://www.youtube.com/watch?v=FX4B-J8DW6E (1h 7min)

**Topics:**
- Solving Ax = b systems
- Row reduction / Gaussian elimination
- Elimination with matrices
- LU decomposition

## DEEPLEARNING.AI (4 hours)

**Course 1: Linear Algebra (Week 4)**
- Linear systems concepts
- **Python labs:** Solving systems with NumPy
- **Application:** Multiple equation solving

## NUMPY CODING (3.5 hours)

```python
import numpy as np

# Solve systems: Ax = b
# 2x + 3y = 8
# 4x + 5y = 14

A = np.array([[2, 3], [4, 5]], dtype=float)
b = np.array([8, 14], dtype=float)

# Method 1: np.linalg.solve
x = np.linalg.solve(A, b)
print("Solution:", x)

# Method 2: Using inverse
x_inv = np.linalg.inv(A) @ b
print("Using inverse:", x_inv)

# Verify
print("Verify Ax = b:", np.allclose(A @ x, b))

# 3×3 system
A = np.array([[1, 2, 1], [2, 1, -1], [1, 1, 2]], dtype=float)
b = np.array([8, 3, 6], dtype=float)
x = np.linalg.solve(A, b)
print("3×3 solution:", x)
print("Verification:", np.allclose(A @ x, b))
```

## PROJECT (2.5 hours)

**Build:** Linear System Solver
- Solve multiple equations
- Verify solutions
- Save: `Week4_LinearSystemSolver.ipynb`

**TOTAL WEEK 4: 15 hours**

---

# WEEK 5: RANK & NULLSPACE

## YOUR SCHEDULE (3.5 hours)

**MIT Lectures 6-10**
- Topics: Column space, row space, rank, four subspaces

## DEEPLEARNING.AI (3.5 hours)

**Course 1: Linear Algebra (Week 5)**
- Rank and nullspace concepts
- **Labs:** Calculating rank and nullspace

## NUMPY CODING (3 hours)

```python
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

# Rank
rank = np.linalg.matrix_rank(A)
print("Rank:", rank)

# Nullspace
nullspace = null_space(A)
print("Nullspace basis:", nullspace)

# Projection
a = np.array([1, 2])
b = np.array([4, 1])
proj = (np.dot(a, b) / np.dot(a, a)) * a
print("Projection:", proj)

# Verify orthogonal
error = b - proj
print("Orthogonal?", np.allclose(np.dot(proj, error), 0))
```

## PROJECT (2.5 hours)

**Build:** Document Analysis Recap
- Combine Weeks 1-4 concepts
- Save: `Week5_DocumentAnalysisRecap.ipynb`

**TOTAL WEEK 5: 13.5 hours**

---

# WEEK 6: EIGENVALUES & EIGENVECTORS ⭐ CRITICAL!

## YOUR SCHEDULE (4 hours)

**3Blue1Brown Episodes 13-14** ← MOST IMPORTANT!
- https://www.youtube.com/watch?v=PFDu9oVAE-g (17 min)

**MIT Lecture 21**
- https://www.youtube.com/watch?v=D0CjYvYz2P8 (1h 26min)

**Topics:**
- Eigenvectors: special directions
- Eigenvalues: scaling factors
- A × v = λ × v (FUNDAMENTAL!)
- Why eigenvectors matter for ML
- Geometric interpretation

**KEY CONCEPT:**
```
Eigenvector v: Direction in space
Eigenvalue λ: Scaling factor

A × v = λ × v means:
Applying A to v just stretches v by λ!
```

## DEEPLEARNING.AI (5 hours)

**Course 1: Linear Algebra (Week 6)**
- Eigenvalues and eigenvectors theory
- **Python lab:** np.linalg.eig() calculations
- **Application:** PCA introduction

## NUMPY CODING (4 hours) ← MOST CRITICAL CODING!

```python
import numpy as np

# Your document matrix from Week 3
A = np.array([[5, 0, 6], [0, 5, 4], [6, 4, 14]], dtype=float)

print("Matrix A (similarity matrix):")
print(A)

# THE KEY FUNCTION - EIGENVALUE DECOMPOSITION
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues:", eigenvalues)
print("Eigenvectors (each column is one):")
print(eigenvectors)

# Understanding the output
print("\nEach column of eigenvectors is one eigenvector:")
print("Eigenvector 1:", eigenvectors[:, 0])
print("Eigenvector 2:", eigenvectors[:, 1])
print("Eigenvector 3:", eigenvectors[:, 2])

print("\nCorresponding eigenvalues:")
print("λ₁ =", eigenvalues[0])
print("λ₂ =", eigenvalues[1])
print("λ₃ =", eigenvalues[2])

# Sort by importance (largest eigenvalues first)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]

print("\n--- SORTED BY IMPORTANCE ---")
print("Sorted eigenvalues:", eigenvalues_sorted)

# Calculate importance percentages
total = np.sum(eigenvalues_sorted)
percentages = (eigenvalues_sorted / total) * 100
cumulative = np.cumsum(percentages)

print("\n--- IMPORTANCE ANALYSIS ---")
for i, (lam, pct, cum) in enumerate(zip(eigenvalues_sorted, percentages, cumulative)):
    print(f"λ_{i+1} = {lam:.1f} ({pct:.1f}%) - Cumulative: {cum:.1f}%")

# CRITICAL VERIFICATION: A × v = λ × v
print("\n" + "="*60)
print("CRITICAL VERIFICATION: A × v = λ × v")
print("="*60)

all_correct = True
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    
    left_side = A @ v
    right_side = lam * v
    
    match = np.allclose(left_side, right_side)
    
    print(f"\nPattern {i+1}:")
    print(f"  λ = {lam:.4f}")
    print(f"  v = {np.round(v, 3)}")
    print(f"  A × v = {np.round(left_side, 3)}")
    print(f"  λ × v = {np.round(right_side, 3)}")
    print(f"  Match? {match} {'✓' if match else '✗'}")
    
    if not match:
        diff = np.linalg.norm(left_side - right_side)
        print(f"  Difference: {diff:.2e}")
    
    all_correct = all_correct and match

print(f"\nAll eigenvectors verified: {all_correct}")

# YOUR DOCUMENT INTERPRETATION
print("\n" + "="*60)
print("YOUR DOCUMENT INTERPRETATION")
print("="*60)

patterns = [
    "Pattern 1 (57%): Document size/content richness",
    "Pattern 2 (29%): Dog vs Cat topic focus",
    "Pattern 3 (14%): Minor variations"
]

for i, pattern in enumerate(patterns):
    print(f"\n{pattern}")
    v = eigenvectors_sorted[:, i]
    print(f"  Eigenvector: {np.round(v, 3)}")
    print(f"  Eigenvalue: {eigenvalues_sorted[i]:.1f}")

print("\nWhat this means:")
print("- Pattern 1: How large/rich the document is")
print("- Pattern 2: Topic focus (dog-centric vs cat-centric)")
print("- Pattern 3: Minor patterns or noise")
```

## ASSIGNMENTS (1 hour)

**Assignment 6.1: Eigenvalues**
- Calculate eigenvalues of any 3×3 matrix
- Sort by importance
- Calculate percentages

**Assignment 6.2: Verify A × v = λ × v**
- For ALL eigenvectors
- Calculate both sides
- Verify they match

**Assignment 6.3: Interpretation (CRITICAL!)**
- What do eigenvectors represent?
- What do eigenvalues represent?
- Interpret your document example

## PROJECT (2 hours)

**Build:** Eigenvalue Analyzer
```python
class EigenvalueAnalyzer:
    def __init__(self, matrix):
        self.matrix = matrix
        self.eigenvalues = None
        self.eigenvectors = None
    
    def calculate(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.matrix)
        self._sort_by_importance()
    
    def _sort_by_importance(self):
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
    
    def get_importance(self):
        total = np.sum(self.eigenvalues)
        return (self.eigenvalues / total) * 100
    
    def verify_property(self):
        for i in range(len(self.eigenvalues)):
            v = self.eigenvectors[:, i]
            lam = self.eigenvalues[i]
            left = self.matrix @ v
            right = lam * v
            assert np.allclose(left, right), f"Mismatch for eigenvector {i}"
    
    def visualize(self):
        pass
```

**TOTAL WEEK 6: 15 hours**

---

# WEEK 7: DIAGONALIZATION - A = V × Λ × V^-1

## YOUR SCHEDULE (3.5 hours)

**MIT Lecture 22**
- https://www.youtube.com/watch?v=13r9QY6g6ES

**Topics:**
- Matrix diagonalization formula
- Why diagonalization works
- Applications to ML
- Powers of matrices

## DEEPLEARNING.AI (4 hours)

**Course 1: Linear Algebra (Week 7)**
- Diagonalization concepts
- **Labs:** Reconstruct matrices
- **Application:** Dimensionality reduction intro

## NUMPY CODING (4 hours)

```python
import numpy as np

A = np.array([[5, 0, 6], [0, 5, 4], [6, 4, 14]], dtype=float)

# Step 1: Eigenanalysis
eigenvalues, eigenvectors = np.linalg.eig(A)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 2: Create matrices
V = eigenvectors
Lambda = np.diag(eigenvalues)
V_inv = np.linalg.inv(V)

# Step 3: Reconstruct A = V × Λ × V^-1
reconstructed = V @ Lambda @ V_inv

print("Original A:")
print(A)
print("\nReconstructed A = V × Λ × V^-1:")
print(reconstructed)
print("Match?", np.allclose(A, reconstructed))

# Step 4: Dimensionality reduction
k = 2
V_reduced = eigenvectors[:, :k]
Lambda_reduced = np.diag(eigenvalues[:k])

# Information retained
info_retained = np.sum(eigenvalues[:k]) / np.sum(eigenvalues)
print(f"\nWith top {k} components: {info_retained*100:.1f}% info retained")
```

## PROJECT (2.5 hours)

**Build:** Document Embeddings System
- Create embeddings from documents
- Analyze patterns
- Visualize in 2D/3D
- Save: `Week7_DocumentEmbeddings.ipynb`

**TOTAL WEEK 7: 13.5 hours**

---

# WEEK 8: SINGULAR VALUE DECOMPOSITION (SVD)

## YOUR SCHEDULE (3.5 hours)

**Resources:**
- MIT Lectures 29-31: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- 3Blue1Brown SVD: https://www.youtube.com/watch?v=mBcgAFivtQg (15 min)

**Topics:**
- SVD formula: A = U × Σ × V^T
- Works for ANY matrix shape
- U: left singular vectors
- Σ: singular values
- V: right singular vectors
- Low-rank approximation

## DEEPLEARNING.AI (4 hours)

**Course 1: Linear Algebra (Week 8)**
- SVD concepts and applications
- **Labs:** SVD calculations with NumPy
- **Application:** Image compression

## NUMPY CODING (4.5 hours)

```python
import numpy as np

# Create matrix (any shape!)
A = np.array([
    [2, 0, 3],
    [1, 0, 0],
    [0, 2, 2],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

print("U shape:", U.shape)
print("s shape:", s.shape)
print("Vt shape:", Vt.shape)

# Reconstruct
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
print("Match?", np.allclose(A, A_reconstructed))

# Singular values importance
total_energy = np.sum(s**2)
importances = (s**2) / total_energy * 100
cumulative = np.cumsum(importances)

print("\nSingular values analysis:")
for i, (sv, imp, cum) in enumerate(zip(s, importances, cumulative)):
    print(f"σ_{i+1} = {sv:.4f} ({imp:.1f}%) - Cumulative: {cum:.1f}%")

# Low-rank approximation
k = 2
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]
A_k = U_k @ np.diag(s_k) @ Vt_k

error = np.linalg.norm(A - A_k)
print(f"\nRank-{k} approximation error: {error:.4f}")
```

## PROJECT (2.5 hours)

**Build:** Image Compression System
- Compress images with SVD
- Analyze compression ratios
- Compare quality
- Save: `Week8_ImageCompression.ipynb`

**TOTAL WEEK 8: 14 hours**

---

# WEEK 9.5: CALCULUS FUNDAMENTALS ⭐

## DEEPLEARNING.AI (5 hours)

**Course 2: Calculus for Machine Learning (Week 1)**
- Derivatives fundamentals
- Gradients
- **Python labs:** Numerical differentiation with NumPy

**Topics:**
- What is a derivative?
- Partial derivatives
- Gradient vectors
- Chain rule
- **Application:** How to optimize ML loss functions

## YOUR SCHEDULE (2 hours)

**Khan Academy - Calculus Essentials**
- Link: https://www.khanacademy.org/math/calculus-1
- Watch: Derivatives, Chain Rule, Optimization
- Duration: 2 hours

## NUMPY CODING (3 hours)

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. DERIVATIVES - Rate of change

def f(x):
    """Simple quadratic: f(x) = x^2 + 3x + 2"""
    return x**2 + 3*x + 2

def derivative_numerical(x, h=0.0001):
    """Approximate derivative using finite differences"""
    return (f(x + h) - f(x)) / h

def derivative_analytical(x):
    """Exact derivative: f'(x) = 2x + 3"""
    return 2*x + 3

x = 2
numerical = derivative_numerical(x)
analytical = derivative_analytical(x)

print("DERIVATIVES:")
print(f"At x = {x}:")
print(f"  Numerical derivative: {numerical:.4f}")
print(f"  Analytical derivative: {analytical:.4f}")
print(f"  Match? {np.allclose(numerical, analytical)}")

# 2. GRADIENTS - Direction of steepest ascent

def f2d(x, y):
    """2D function: f(x,y) = x^2 + y^2 + xy"""
    return x**2 + y**2 + x*y

def partial_x(x, y, h=0.0001):
    """Partial derivative with respect to x"""
    return (f2d(x+h, y) - f2d(x, y)) / h

def partial_y(x, y, h=0.0001):
    """Partial derivative with respect to y"""
    return (f2d(x, y+h) - f2d(x, y)) / h

x, y = 2, 3
grad_x = partial_x(x, y)
grad_y = partial_y(x, y)
gradient = np.array([grad_x, grad_y])

print("\n\nGRADIENTS (2D):")
print(f"At point ({x}, {y}):")
print(f"  ∂f/∂x = {grad_x:.4f}")
print(f"  ∂f/∂y = {grad_y:.4f}")
print(f"  Gradient vector: {gradient}")
print(f"  Gradient magnitude: {np.linalg.norm(gradient):.4f}")

# 3. ARRAY OPERATIONS FOR GRADIENTS

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y = X**2 + 3*X + 2

# Forward differences (approximate gradient)
gradient = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])

print("\n\nGRADIENTS FOR ARRAY:")
print("X:", X)
print("Y = X^2 + 3X + 2:", Y)
print("Gradient (finite differences):", gradient)
print("Analytical gradient (2X + 3):", 2*X[:-1] + 3)

# 4. CHAIN RULE

def h(x):
    """Composite function: h(x) = (x^2 + 1)^3"""
    return (x**2 + 1)**3

def h_derivative_chain_rule(x):
    """
    h(x) = (x^2 + 1)^3
    Let u = x^2 + 1
    Then h = u^3
    
    dh/dx = dh/du * du/dx = 3u^2 * 2x = 6x(x^2 + 1)^2
    """
    return 6*x*(x**2 + 1)**2

x = 2
numerical_h = derivative_numerical(h, x)
analytical_h = h_derivative_chain_rule(x)

print("\n\nCHAIN RULE:")
print(f"h(x) = (x^2 + 1)^3")
print(f"At x = {x}:")
print(f"  Numerical derivative: {numerical_h:.4f}")
print(f"  Analytical (chain rule): {analytical_h:.4f}")

# 5. OPTIMIZATION - Finding minimum

def loss(x):
    """Loss function we want to minimize"""
    return (x - 3)**2

x_values = np.linspace(-2, 8, 100)
loss_values = [loss(x) for x in x_values]

# Find minimum analytically (derivative = 0)
x_min = 3.0
loss_min = loss(x_min)

print("\n\nOPTIMIZATION - Finding Minimum:")
print(f"Loss function: (x - 3)^2")
print(f"Minimum at x = {x_min}, loss = {loss_min:.4f}")
print(f"Derivative at minimum: {derivative_numerical(loss, x_min):.4f} (should be 0)")

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x_values, loss_values, 'b-', label='Loss')
plt.plot(x_min, loss_min, 'r*', markersize=15, label='Minimum')
plt.xlabel('x')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.legend()
plt.grid()

# Gradient visualization
plt.subplot(1, 2, 2)
x_grad = np.linspace(-2, 8, 50)
grad_values = [2*(x-3) for x in x_grad]
plt.plot(x_grad, grad_values, 'g-', label='Gradient')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=3, color='r', linestyle='--', alpha=0.3, label='Minimum')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.title('Gradient of Loss Function')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Save as: week9_5_calculus.py
```

## ASSIGNMENTS (1 hour)

**Assignment 9.5.1: Derivatives**
- Calculate derivatives of 3 functions
- Verify numerical vs analytical

**Assignment 9.5.2: Gradients**
- Calculate 2D gradients
- Visualize gradient vectors

**Assignment 9.5.3: Optimization**
- Find minimum of a function
- Verify using derivatives

## PROJECT (2.5 hours)

**Build:** Gradient Visualizer
- Plot functions
- Show gradients as arrows
- Highlight minima
- Save: `Week9_5_GradientVisualizer.ipynb`

**TOTAL WEEK 9.5: 10.5 hours**

---

# WEEK 10: GRADIENT DESCENT FROM SCRATCH

## DEEPLEARNING.AI (4 hours)

**Course 2: Calculus (Week 2)**
- Gradients for multiple variables
- Optimization problems
- **Labs:** Implement gradient descent

## PATRICK LOEBER (4 hours)

**Start Patrick Loeber series** (Week 10)
- Algorithm 1: KNN (code from scratch)
- Algorithm 2: Linear Regression
- https://www.youtube.com/@patloeber

**Weekly Pattern:**
- Watch video (1 hour)
- Code along from scratch (1.5 hours)
- Modify and experiment (1 hour)
- Create variations (1.5 hours)

## NUMPY CODING (5 hours)

```python
import numpy as np
import matplotlib.pyplot as plt

# GRADIENT DESCENT - THE HEART OF MACHINE LEARNING!

def gradient_descent_1d(x_start, learning_rate=0.01, iterations=100):
    """
    Minimize: f(x) = (x - 3)^2
    Derivative: f'(x) = 2(x - 3)
    """
    
    x = x_start
    history = [x]
    losses = []
    
    for i in range(iterations):
        # Current loss
        loss = (x - 3)**2
        losses.append(loss)
        
        # Gradient (derivative)
        gradient = 2*(x - 3)
        
        # Update: x_new = x_old - learning_rate * gradient
        x = x - learning_rate * gradient
        history.append(x)
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: x = {x:.4f}, loss = {loss:.4f}, gradient = {gradient:.4f}")
    
    return np.array(history), np.array(losses)

print("="*60)
print("1D GRADIENT DESCENT")
print("="*60)

x_start = 10.0
history, losses = gradient_descent_1d(x_start, learning_rate=0.1, iterations=50)

print(f"\nStarting point: {x_start}")
print(f"Final point: {history[-1]:.4f}")
print(f"True minimum: 3.0")
print(f"Final loss: {losses[-1]:.6f}")

# Visualize
plt.figure(figsize=(12, 4))

# Loss function
plt.subplot(1, 3, 1)
x_vals = np.linspace(0, 10, 100)
y_vals = (x_vals - 3)**2
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Loss = (x-3)²')
plt.plot(history, losses, 'ro-', markersize=4, alpha=0.6, label='Gradient descent path')
plt.xlabel('x')
plt.ylabel('Loss')
plt.title('1D Gradient Descent')
plt.legend()
plt.grid()

# Convergence
plt.subplot(1, 3, 2)
plt.plot(losses, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.grid()

# Position over time
plt.subplot(1, 3, 3)
plt.plot(history, 'g-', linewidth=2, label='x value')
plt.axhline(y=3, color='r', linestyle='--', label='Optimum (x=3)')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Position Over Iterations')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 2D GRADIENT DESCENT

def gradient_descent_2d(X, y, learning_rate=0.01, iterations=100):
    """
    Minimize: Mean Squared Error loss
    For linear regression: minimize ||y - Xw||^2
    Gradient: -2 * X^T * (y - X*w) / m
    """
    
    m = len(y)
    # Add bias term
    X = np.column_stack([np.ones(m), X])
    
    # Initialize weights
    w = np.zeros(X.shape[1])
    
    losses = []
    
    for iteration in range(iterations):
        # Predictions
        y_pred = X @ w
        
        # Error
        error = y_pred - y
        
        # Loss (MSE)
        loss = (1/(2*m)) * np.sum(error**2)
        losses.append(loss)
        
        # Gradient
        gradient = (1/m) * X.T @ error
        
        # Update
        w = w - learning_rate * gradient
        
        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration+1}: Loss = {loss:.4f}")
    
    return w, losses

print("\n\n" + "="*60)
print("2D GRADIENT DESCENT - LINEAR REGRESSION")
print("="*60)

# Generate data: y = 2.5*x + 1 + noise
np.random.seed(42)
X = np.random.randn(20, 1) * 10
y = 2.5 * X.ravel() + 1.0 + np.random.randn(20) * 2

print(f"True relationship: y = 2.5*x + 1")

# Gradient descent
w, losses = gradient_descent_2d(X, y, learning_rate=0.01, iterations=100)

print(f"\nLearned weights: bias = {w[0]:.4f}, slope = {w[1]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")

# Visualize
plt.figure(figsize=(12, 4))

# Data and fit
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Data')
x_line = np.linspace(X.min(), X.max(), 100)
y_line = w[1] * x_line + w[0]
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fit: y = {w[1]:.2f}x + {w[0]:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression via Gradient Descent')
plt.legend()
plt.grid()

# Loss convergence
plt.subplot(1, 2, 2)
plt.plot(losses, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss Convergence')
plt.grid()

plt.tight_layout()
plt.show()

# Save as: week10_gradient_descent.py
```

## ASSIGNMENTS (1 hour)

**Assignment 10.1: 1D Gradient Descent**
- Implement for different functions
- Try different learning rates
- Observe convergence speed

**Assignment 10.2: 2D Gradient Descent**
- On linear regression problem
- Compare with analytical solution
- Visualize convergence

**Assignment 10.3: Learning Rate Effects**
- Too high: diverges
- Too low: slow convergence
- Just right: smooth convergence
- Find optimal learning rate

## PROJECT (2 hours)

**Build:** Gradient Descent Optimizer
```python
class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.lr = learning_rate
        self.iterations = iterations
    
    def fit(self, X, y):
        # Implement gradient descent
        pass
    
    def predict(self, X):
        # Make predictions
        pass
    
    def visualize_convergence(self):
        # Plot loss over iterations
        pass
```

**TOTAL WEEK 10: 18 hours**

---

# WEEK 11: ADVANCED OPTIMIZATION & ALGORITHMS

## DEEPLEARNING.AI (3 hours)

**Course 2: Calculus (Week 3)**
- Advanced optimization
- Momentum
- Adaptive learning rates (preview)

## PATRICK LOEBER (5 hours)

**Continue algorithms 3-4**
- Logistic Regression
- SVM (with optimization perspective)

## ZERO TO MASTERY (8 hours)

**Start Zero to Mastery** (Week 11):
https://zerotomastery.io/courses/machine-learning-and-data-science-bootcamp/

**Sections:**
- Section 1: Python Fundamentals (2 hours)
- Section 2: Data Science Fundamentals (3 hours)
- Section 3: Data Preprocessing (3 hours)

## NUMPY CODING (3 hours)

```python
import numpy as np

# ADVANCED GRADIENT DESCENT

def gradient_descent_with_momentum(X, y, learning_rate=0.01, momentum=0.9, iterations=100):
    """
    Gradient descent with momentum
    Helps avoid local minima and speeds up convergence
    """
    
    m = len(y)
    X = np.column_stack([np.ones(m), X])
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])  # Velocity vector
    
    losses = []
    
    for iteration in range(iterations):
        y_pred = X @ w
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error**2)
        losses.append(loss)
        
        gradient = (1/m) * X.T @ error
        
        # Momentum update
        v = momentum * v + learning_rate * gradient
        w = w - v
        
        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration+1}: Loss = {loss:.4f}")
    
    return w, losses

# Test on data
np.random.seed(42)
X = np.random.randn(50, 1) * 10
y = 2.5 * X.ravel() + 1.0 + np.random.randn(50) * 2

w_momentum, losses_momentum = gradient_descent_with_momentum(X, y)

print(f"With momentum: bias = {w_momentum[0]:.4f}, slope = {w_momentum[1]:.4f}")
print(f"Final loss: {losses_momentum[-1]:.4f}")

# STOCHASTIC GRADIENT DESCENT (SGD)

def stochastic_gradient_descent(X, y, learning_rate=0.01, iterations=100, batch_size=5):
    """
    Process small batches instead of full dataset
    Much faster for large datasets!
    """
    
    m = len(y)
    X = np.column_stack([np.ones(m), X])
    w = np.zeros(X.shape[1])
    
    losses = []
    
    for iteration in range(iterations):
        # Random indices for batch
        indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        
        y_pred = X_batch @ w
        error = y_pred - y_batch
        
        # Full dataset loss for monitoring
        y_pred_full = X @ w
        loss = (1/(2*m)) * np.sum((y_pred_full - y)**2)
        losses.append(loss)
        
        gradient = (1/batch_size) * X_batch.T @ error
        w = w - learning_rate * gradient
        
        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration+1}: Loss = {loss:.4f}")
    
    return w, losses

w_sgd, losses_sgd = stochastic_gradient_descent(X, y)

print(f"\nSGD: bias = {w_sgd[0]:.4f}, slope = {w_sgd[1]:.4f}")
print(f"Final loss: {losses_sgd[-1]:.4f}")
```

## PROJECT (2.5 hours)

**Build:** Advanced Optimizer
- Compare momentum vs regular GD
- Implement SGD
- Benchmark performance
- Save: `Week11_AdvancedOptimization.ipynb`

**TOTAL WEEK 11: 21 hours**

---

# WEEKS 12-24: APPLICATIONS & CAPSTONES

---

# WEEK 12: MACHINE LEARNING APPLICATIONS

**Combines:** Patrick Loeber (Algorithms) + Zero to Mastery (Tools) + Your understanding (Math)

- Algorithms 5-10 (Decision Trees, Random Forest, KNN, SVM, etc.)
- Zero to Mastery sections 4-6
- Full ML pipeline projects
- Real datasets and evaluation metrics

**TOTAL WEEK 12: 20 hours**

---

# WEEK 13: CAPSTONE 1 - COMPLETE NLP SYSTEM

**Time:** 25 hours (full week dedicated)

**Combines:** Your weeks 1-7 + DeepLearning.AI + Patrick Loeber (PCA) + Zero to Mastery

```python
# File: Week13_CapstoneNLP.ipynb

class NLPPipeline:
    def __init__(self, documents):
        # Your schedule: Document representation
        self.documents = documents
        
        # DeepLearning.AI: Matrix operations
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents).toarray()
        
        # Patrick Loeber: PCA
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=2)
        self.embeddings = self.pca.fit_transform(self.tfidf_matrix)
        
        # Your schedule: Similarity (Week 3)
        from sklearn.metrics.pairwise import cosine_similarity
        self.similarity = cosine_similarity(self.tfidf_matrix)
    
    def search(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        similarities = np.dot(self.tfidf_matrix, query_vec)
        top_idx = similarities.argsort()[::-1][:top_k]
        return [(self.documents[i], similarities[i]) for i in top_idx]
    
    def visualize(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.embeddings[:, 0], self.embeddings[:, 1])
        for i in range(len(self.documents)):
            plt.annotate(f"Doc{i}", self.embeddings[i])
        plt.show()
    
    def evaluate(self):
        pass

# Run complete pipeline
# Test with various queries
# Visualize results
# Document all code
```

**Deliverable:**
- Complete notebook with comments
- README explaining approach
- Test results and metrics
- Visualizations
- GitHub commit

**TOTAL WEEK 13: 25 hours**

---

# WEEK 14: OPTIMIZATION & PERFORMANCE

**Topics:**
- Broadcasting in NumPy
- Vectorization vs loops
- Memory efficiency
- Sparse matrices
- Parallel computing intro

**Code Example:**
```python
# File: week14_optimization.py
import numpy as np
import time

# Broadcasting
A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
result = A + b  # Broadcasting!

# Performance comparison
n = 1000000
a = np.random.randn(n)
b = np.random.randn(n)

# Loop (SLOW)
start = time.time()
dot_loop = 0
for i in range(n):
    dot_loop += a[i] * b[i]
time_loop = time.time() - start

# NumPy (FAST!)
start = time.time()
dot_numpy = np.dot(a, b)
time_numpy = time.time() - start

print(f"Loop time: {time_loop:.4f}s")
print(f"NumPy time: {time_numpy:.4f}s")
print(f"Speedup: {time_loop/time_numpy:.1f}x")

# Sparse matrices
from scipy.sparse import csr_matrix
dense = np.array([[1, 0, 0], [0, 0, 2], [3, 0, 0]])
sparse = csr_matrix(dense)
print(f"Dense: {dense.nbytes} bytes")
print(f"Sparse: {sparse.data.nbytes} bytes")
```

**Project:** Optimize previous code
- Measure before/after
- Identify bottlenecks
- Apply optimization techniques

**TOTAL WEEK 14: 20 hours**

---

# WEEK 15: LARGE-SCALE COMPUTING

**Topics:**
- Randomized SVD
- Incremental learning
- GPU computing basics
- Distributed computing (Dask)

**Code Example:**
```python
# File: week15_large_scale.py
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import IncrementalPCA
import time

# Large matrix
n, m = 10000, 5000
A = np.random.randn(n, m)

# Standard SVD (slow)
start = time.time()
U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=False)
time_full = time.time() - start

# Randomized SVD (fast!)
start = time.time()
U_rand, s_rand, Vt_rand = svds(A, k=100)
time_rand = time.time() - start

print(f"Standard SVD: {time_full:.4f}s")
print(f"Randomized SVD: {time_rand:.4f}s")
print(f"Speedup: {time_full/time_rand:.1f}x")

# Incremental PCA (for streaming data)
ipca = IncrementalPCA(n_components=10)
for batch in range(10):
    X_batch = np.random.randn(1000, 100)
    ipca.partial_fit(X_batch)

print(f"Explained variance: {ipca.explained_variance_ratio_.sum()*100:.1f}%")
```

**Project:** Process large datasets
- Create simulated large data
- Compare SVD approaches
- Benchmark performance
- Document findings

**TOTAL WEEK 15: 20 hours**

---

# WEEK 16: CAPSTONE 2 - YOUR CHOICE PROJECT

**Time:** 28 hours

**Choose one:**

### Option 1: Advanced Recommendation System (28 hours)
```python
# File: Week16_RecommendationSystem.ipynb

class MovieRecommender:
    def __init__(self, ratings_df):
        self.ratings = ratings_df.values
        self.users = ratings_df.index
        self.movies = ratings_df.columns
    
    def matrix_factorization_svd(self):
        U, s, Vt = np.linalg.svd(self.ratings, full_matrices=False)
        k = 50
        ratings_pred = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        return ratings_pred
    
    def recommend(self, user_id, top_k=5):
        predictions = self.matrix_factorization_svd()
        user_idx = self.users.get_loc(user_id)
        unrated = np.where(self.ratings[user_idx] == 0)[0]
        pred_scores = [(m_idx, predictions[user_idx, m_idx]) 
                       for m_idx in unrated]
        return sorted(pred_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def evaluate(self, test_ratings):
        pass

# Use MovieLens dataset (free)
# Train model
# Make recommendations
# Evaluate performance
# Deploy with Flask
```

### Option 2: Image Analysis with Deep Learning (28 hours)
- SVD compression + CNN classification
- Zero to Mastery CNN techniques
- Your eigenanalysis knowledge

### Option 3: Stock Price Prediction (28 hours)
- Your eigenanalysis + LSTM
- Time series forecasting
- Backtesting

### Option 4: Production ML System (28 hours)
- Full pipeline
- Docker containerization
- Flask/FastAPI deployment
- Unit tests
- CI/CD with GitHub

**Deliverables:**
- Complete code
- README.md with explanation
- requirements.txt with dependencies
- main.py or notebook.ipynb
- results/ folder with outputs
- tests/ folder with unit tests
- GitHub repository with commits
- Blog post or presentation
- Performance metrics

**TOTAL WEEK 16: 28 hours**

---

# WEEKS 17-24: EXTENDED SPECIALIZATION (OPTIONAL)

**Choose your specialization:**

- **NLP Track** (Weeks 17-20): Advanced NLP, transformers, language models
- **Computer Vision Track** (Weeks 17-20): CNNs, object detection, segmentation
- **Reinforcement Learning** (Weeks 17-20): Q-learning, policy gradients
- **Advanced Deep Learning** (Weeks 17-20): GANs, autoencoders, attention mechanisms
- **Production ML** (Weeks 17-20): MLOps, model serving, monitoring

Each track includes:
- Advanced algorithms
- Real-world projects
- State-of-the-art techniques
- Production considerations

---

# COMPLETE TIME SUMMARY TABLE

| Week | Focus | Your | DL.AI | Loeber | Z2M | Total |
|------|-------|------|-------|--------|-----|-------|
| 1 | Vectors | 3 | 3 | - | - | 12 |
| 2 | Matrices | 3.5 | 4 | - | - | 14 |
| 3 | Transpose/Dot | 3 | 4 | - | - | 14 |
| 4 | Systems | 4 | 4 | - | - | 15 |
| 5 | Rank | 3.5 | 3.5 | - | - | 13.5 |
| 6 | Eigenvalues ⭐ | 4 | 5 | - | - | 15 |
| 7 | Diagonalization | 3.5 | 4 | - | - | 13.5 |
| 8 | SVD | 3.5 | 4 | - | - | 14 |
| 9.5 | **CALCULUS** | 2 | 5 | - | - | 10.5 |
| 10 | **GRAD DESCENT** | - | 4 | 4 | - | 18 |
| 11 | **ADV OPTIM** | - | 3 | 5 | 8 | 21 |
| 12 | ML Apps | - | 2 | 5 | 8 | 20 |
| 13 | Capstone 1 | 10 | - | 5 | 10 | 25 |
| 14 | Optimization | 5 | - | 3 | 12 | 20 |
| 15 | Large-Scale | 5 | - | 3 | 12 | 20 |
| 16 | Capstone 2 | 8 | - | 2 | 18 | 28 |
| 17-24 | Specialization | 40 | - | 20 | 40 | 100 |
| **TOTAL** | | **110** | **58** | **48** | **108** | **497 hrs** |

---

# COMPLETE RESOURCE DIRECTORY

## Your Custom Files (7 total)
- 16week_complete_schedule.md
- integrated_16week_full_schedule.md
- detailed_numpy_coding_guide.md
- document_embeddings_fixed.md
- numpy_tutorial_coding.md
- complete_numpy_linear_algebra_guide.md
- numpy_mastery_complete_roadmap.md

## Main Courses (3)

**DeepLearning.AI Math for ML**
- https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science
- 3 courses: Linear Algebra, Calculus, Statistics
- 11 weeks total
- FREE audit option or $49/month

**Patrick Loeber ML From Scratch**
- https://www.youtube.com/@patloeber
- 15+ algorithms with NumPy
- Completely FREE
- All code from scratch

**Zero to Mastery ML Bootcamp**
- https://zerotomastery.io/courses/machine-learning-and-data-science-bootcamp/
- 40+ hours
- Modern tools and techniques
- $15/month or $200 one-time

## Supplementary Resources (ALL FREE)

**Videos:**
- 3Blue1Brown: https://www.youtube.com/c/3blue1brown
- MIT 18.06: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- Khan Academy: https://www.khanacademy.org
- StatQuest: https://www.youtube.com/@statquest

**Documentation:**
- NumPy: https://numpy.org
- Pandas: https://pandas.pydata.org
- Scikit-learn: https://scikit-learn.org
- TensorFlow: https://www.tensorflow.org
- Matplotlib: https://matplotlib.org

**Interactive:**
- CodeSignal: https://learn.codesignal.com
- GeeksforGeeks: https://www.geeksforgeeks.org
- W3Schools: https://www.w3schools.com/python

## Tools (ALL FREE)
- Anaconda: https://www.anaconda.com/download
- VS Code: https://code.visualstudio.com
- GitHub: https://github.com
- Google Colab: https://colab.research.google.com

---

# FINAL CHECKLIST

**Before Week 1:**
- [ ] Download all custom files
- [ ] Install Anaconda
- [ ] Create GitHub account
- [ ] Bookmark all links
- [ ] Create learning folder structure

**During Weeks 1-8:**
- [ ] Complete all exercises and assignments
- [ ] Build all projects
- [ ] Commit to GitHub weekly
- [ ] Review previous weeks

**During Weeks 9-11:**
- [ ] Master calculus and gradient descent
- [ ] Start Patrick Loeber videos
- [ ] Begin Zero to Mastery sections
- [ ] Consolidate understanding

**During Weeks 12-16:**
- [ ] Build capstone projects
- [ ] Document everything
- [ ] Create portfolio
- [ ] Publish blog posts
- [ ] Prepare for interviews

**Weeks 17-24 (Optional):**
- [ ] Choose specialization
- [ ] Deep dive into chosen track
- [ ] Build advanced projects
- [ ] Prepare for role-specific interviews

---

# LEARNING PHILOSOPHY

**Why this order?**
1. **Weeks 1-8:** Learn the MATH (vectors, matrices, eigenvalues, SVD)
2. **Week 9.5:** Learn CALCULUS (derivatives, gradients)
3. **Weeks 10-11:** Learn OPTIMIZATION (gradient descent combines math + calculus)
4. **Weeks 12-16:** Apply EVERYTHING to real problems
5. **Weeks 17-24:** Specialize and deepen

**Why NOT just algorithms?**
- You'll understand not just HOW but WHY
- You won't just copy code, you'll write it
- You'll debug issues instead of being confused
- You'll innovate instead of just following tutorials

**Result:** Professional ML/AI engineer with deep understanding! 🎓

---

# START TODAY!

**Week 1, Day 1:**
1. Download Anaconda
2. Watch 3Blue1Brown Episode 1 (10 min)
3. Open Jupyter
4. Create your first vector
5. Commit to GitHub

**In 24 weeks: Professional ML/AI Engineer!**
**In 6 months: Expert-level specialization!**

You now have a complete 497-hour roadmap to professional mastery! 🚀

**Download, print, and start Week 1 today!** 📚💻🎓