# 16-WEEK COMPLETE SCHEDULE
## NumPy & Linear Algebra Mastery

---

# WEEK 1: VECTORS & BASICS

## THEORY (3 hours)
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
## DeepLearning.AI Linear Algebra (3 hours)
**Coursera Link:** https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science

**Course 1: Linear Algebra for ML (Week 1 content)**
- Vectors as data representations
- Vector magnitude and direction
- **Python lab:** Create vectors in NumPy
- **Application:** Feature vectors in ML
**Action:**
- Watch video lectures (1.5 hours)
- Complete interactive labs (1.5 hours)
## NUMPY CODING (4 hours)
- W3Schools NumPy: https://www.w3schools.com/python/numpy/
- YouTube Tutorial: https://www.youtube.com/watch?v=9JUAPWk_IFU
**Resources:** https://numpy.org/doc/stable/user/quickstart.html

```python
import numpy as np

# Create vectors
v1 = np.array([3, 4])
v2 = np.array([1, 2])

# Vector operations
v_sum = v1 + v2
v_diff = v1 - v2
v_scaled = 2 * v1

# Magnitude
magnitude = np.linalg.norm(v1)

# Dot product
dot = np.dot(v1, v2)

# Angle between vectors
cos_angle = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.arccos(cos_angle)
```
**Exercises (from detailed_numpy_coding_guide.md):**

```python
# Exercise 1: Create and Inspect Vectors (45 min)
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

```python
# Exercise 2: Vector Properties and Norms (45 min)
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

```python
# Exercise 3: Your Document Example (30 min)
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
print(f"Similarity Doc1-Doc3: {sim_13}")  # Should be 6 (your insight!)
print(f"Similarity Doc2-Doc3: {sim_23}")  # Should be 4

print(f"\nVerification: 2 (dog in Doc1) × 3 (dog in Doc3) = {2*3}")

# Save as: week1_exercise3.py
```

## ASSIGNMENTS (1 hour)

**Assignment 1.1: Vector Operations**
```python
# File: assignment1_1.py
# Create 5 different vectors (various sizes)
# Perform all 4 basic operations (add, sub, mult, div)
# Test with random data
```

**Assignment 1.2: Properties**
```python
# File: assignment1_2.py
# Calculate magnitude of [3, 4, 5]
# Create unit vector
# Calculate angle between [1, 0, 0] and [1, 1, 1]
```

**Assignment 1.3: Your Document Example**
```python
# File: assignment1_3.py
# Verify all three similarity calculations
# Explain why each value makes sense
# Add 2 more documents and calculate similarities
```
## Patrick Loeber (0 hours this week - starts Week 2)

## Zero to Mastery (0 hours this week - starts Week 9)
## PROJECT (2 hours)
- Build vector calculator
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

- Test: addition, subtraction, scaling, dot product
- Save: `Week1_VectorCalculator.ipynb`

**TOTAL WEEK 1: 12 hours**
---

# WEEK 2: MATRICES & TRANSFORMATIONS

## THEORY (3.5 hours)
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

## DeepLearning.AI Linear Algebra (4 hours)
**Course 1: Week 2 - Matrices**
- Matrix representation of data
- Matrix operations fundamentals
- **Python lab:** NumPy matrix operations
- **Application:** Image as matrix, transformations
**Action:**
- Watch lectures (2 hours)
- Complete labs (2 hours)
```python
# DeepLearning.AI lab code
import numpy as np

# Represent image as matrix
image_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Matrix operations
print("Matrix shape:", image_matrix.shape)
print("Matrix transpose:", image_matrix.T)

# Real application: Filter (matrix) applied to image
kernel = np.array([[1, 0], [0, 1]])  # Identity-like kernel
```

## NUMPY CODING (4.5 hours)

```python
import numpy as np

# Create matrices
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[1, 0], [0, 1], [-1, 2]])

# Access elements
print(A[0, 0])  # Element
print(A[0, :])  # Row
print(A[:, 0])  # Column

# Special matrices
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)

# Matrix multiplication
result = A @ B
print("A @ B shape:", result.shape)

# Rotation matrix
theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# Apply transformation
v = np.array([1, 0])
transformed = R @ v
```

**Resource Links:**
- NumPy Matrix Docs: https://numpy.org/doc/stable/reference/arrays.ndarray.html
- GeeksforGeeks: https://www.geeksforgeeks.org/matrix-operations-in-python/

**Exercises:**

```python
# Exercise 1: Create and Access Matrices (60 min)
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

```python
# Exercise 2: Matrix Operations (90 min)
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

# Special matrices
I = np.eye(3)
print("\nA @ I (should equal A):")
print(np.allclose(A @ I, A))

# Rotation matrix (2D)
theta = np.pi / 4  # 45 degrees
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

point = np.array([1, 0])
rotated = R @ point
print(f"\nPoint [1, 0] rotated 45°: {rotated}")

# Save as: week2_exercise2.py
```

```python
# Exercise 3: Image as Matrix (60 min)
import numpy as np

# Create a small "image"
image = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
], dtype=float)

print("Original image:")
print(image)

# Scale image
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
print("\nOriginal points:")
print(points)
print("Rotated points (45°):")
print(rotated_points)

# Save as: week2_exercise3.py
```

## ASSIGNMENTS (1 hour)

**Assignment 2.1: Matrix Basics**
```python
# File: assignment2_1.py
# Create 3 different matrices
# Access specific elements
# Extract rows/columns/submatrices
```

**Assignment 2.2: Matrix Multiplication**
```python
# File: assignment2_2.py
# Multiply [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
# Verify manually
# Test commutativity (A@B vs B@A)
```

**Assignment 2.3: Transformations**
```python
# File: assignment2_3.py
# Create rotation matrix for 60 degrees
# Rotate points [1, 0], [0, 1], [1, 1]
# Verify distances preserved
```

## PROJECT (2.5 hours)
- Image rotation using matrices
- Transform shapes geometrically
**Build:** Image Transformation System
```python
# File: Week2_ImageTransformations.ipynb

class ImageTransformer:
    @staticmethod
    def rotate(image, degrees):
        theta = np.radians(degrees)
        # Implementation
        pass
    
    @staticmethod
    def scale(image, factor):
        # Implementation
        pass
    
    @staticmethod
    def combine_transforms(image, transforms):
        # Apply multiple transformations
        pass

# Test with sample images
# Visualize before/after
# Save and commit
```

- Visualize before/after
- Save: `Week2_ImageRotation.ipynb`
- Document similarity matrix (Week 3 preview)
## Patrick Loeber (0 hours)
**TOTAL WEEK 2: 14 hours**
---

# WEEK 3: TRANSPOSE & DOT PRODUCT

## THEORY (3 hours)
**3Blue1Brown Episodes 5-7**
- Episode 5 - 3D transformations: https://www.youtube.com/watch?v=v8VSDg_WQlA (14 min)
- Episode 6 - Determinant: https://www.youtube.com/watch?v=Ip3X9LOh2dk (13 min)
- Episode 7 - Inverse matrices: https://www.youtube.com/watch?v=uQhTuRlWMxw (12 min)

**Topics:**
- Transpose: flip rows and columns
- Dot product: measure similarity
- Determinant: scaling factor
- Inverse matrices
- Singular vs invertible
## DeepLearning.AI Linear Algebra (4 hours)
**Course 1: Week 3 - Matrix Operations**
- Transpose operations
- Dot product for similarity
- **Python lab:** A^T × A calculations
- **Application:** Document similarity (YOUR EXAMPLE!)

```python
# DeepLearning.AI integration with YOUR document example
import numpy as np

# Word frequencies (from your example)
A = np.array([
    [2, 0, 3],
    [1, 0, 0],
    [0, 2, 2],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

# Transpose
At = A.T

# Similarity matrix
ATA = At @ A

# YOUR CALCULATION: 2×3=6
print("A^T × A[0,2] (your insight):", ATA[0, 2])

# Lab assignment:
# - Calculate document similarity
# - Explain using matrix operations
# - Visualize results
```

## NUMPY CODING (4.5 hours)

```python
import numpy as np

# Transpose
A = np.array([[2, 0, 3], [1, 0, 0], [0, 2, 2], [0, 1, 0], [0, 0, 1]], dtype=float)
At = A.T
print("Verify (A^T)^T = A:", np.allclose(At.T, A))

# Similarity matrix - YOUR CALCULATION!
ATA = At @ A
print("A^T × A:")
print(ATA)
print("Element [0,2] (your 2×3=6):", ATA[0, 2])

# Determinant
B = np.array([[1, 2], [3, 4]], dtype=float)
det_B = np.linalg.det(B)
print("Determinant:", det_B)

# Inverse
B_inv = np.linalg.inv(B)
print("Verify B @ B^-1 = I:", np.allclose(B @ B_inv, np.eye(2)))
```
**Exercises:**

```python
# Exercise 1: Transpose Operations (60 min)
import numpy as np

A = np.array([
    [2, 0, 3],
    [1, 0, 0],
    [0, 2, 2],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

print("Original A shape:", A.shape)
print("A:")
print(A)

At = A.T
print("\nTranspose A^T shape:", At.shape)
print("A^T:")
print(At)

# Verify (A^T)^T = A
double_transpose = At.T
print("\nVerify (A^T)^T = A:", np.allclose(double_transpose, A))

# Transpose properties
B = np.array([[1, 2], [3, 4]], dtype=float)
C = np.array([[5, 6], [7, 8]], dtype=float)

BC = B @ C
BC_T = BC.T
Ct_Bt = C.T @ B.T

print("\nProperty: (B@C)^T = C^T @ B^T")
print("Are equal?", np.allclose(BC_T, Ct_Bt))

# Symmetric matrices
S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
print("\nIs S symmetric?", np.allclose(S, S.T))

# Save as: week3_exercise1.py
```

```python
# Exercise 2: Dot Product & Similarity (90 min) - YOUR FOCUS!
import numpy as np

print("=== YOUR DOCUMENT EXAMPLE ===\n")

# Feature vectors
doc1 = np.array([2, 1, 0, 0, 0], dtype=float)  # [dog, barks, cat, meows, play]
doc2 = np.array([0, 0, 2, 1, 0], dtype=float)
doc3 = np.array([3, 0, 2, 0, 1], dtype=float)

A = np.array([doc1, doc2, doc3]).T  # (5, 3)
At = A.T

print("Matrix A (word frequencies):")
print(A)
print("Shape:", A.shape)

print("\nTranspose A^T:")
print(At)
print("Shape:", At.shape)

# Calculate A^T × A
ATA = At @ A
print("\nSimilarity Matrix A^T × A:")
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

```python
# Exercise 3: Determinant & Inverse (60 min)
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
print("(This is 0 - columns are linearly dependent)")

# Inverse
print("\n--- Inverse ---")
if abs(det_B) > 1e-10:
    B_inv = np.linalg.inv(B)
    print("B inverse:")
    print(B_inv)
    
    product = B @ B_inv
    identity = np.eye(2)
    print("\nB @ B^-1 =")
    print(product)
    print("Is it identity?", np.allclose(product, identity))

# Save as: week3_exercise3.py
```

## ASSIGNMENTS (1 hour)

**Assignment 3.1: Transpose**
```python
# File: assignment3_1.py
# Transpose 3 matrices
# Verify (A^T)^T = A
# Find a symmetric matrix
```

**Assignment 3.2: Dot Product (YOUR FOCUS!)**
```python
# File: assignment3_2.py
# Calculate your document similarity matrix
# Verify 2×3=6 calculation
# Explain each number
# Create new documents and calculate
```

**Assignment 3.3: Determinant & Inverse**
```python
# File: assignment3_3.py
# Calculate determinants of 2 matrices
# Find inverses (if possible)
# Verify A @ A^-1 = I
```

## PROJECT (2.5 hours)
- Calculate document similarity matrix
**Build:** Document Similarity Calculator
```python
# File: Week3_DocumentSimilarity.ipynb

class DocumentSimilarity:
    def __init__(self, documents):
        self.documents = documents
        self.word_freq_matrix = self._create_matrix()
    
    def _create_matrix(self):
        # Create word frequency matrix
        pass
    
    def calculate_similarity(self):
        # Calculate A^T × A
        At = self.word_freq_matrix.T
        return At @ self.word_freq_matrix
    
    def find_similar(self, doc_idx, top_k=3):
        # Find most similar documents
        pass
    
    def visualize(self):
        # Visualize similarity matrix
        pass

# Use your document example
# Add more documents
# Calculate and display similarities
```
- Implement 2×3=6 correctly!
- Display with Pandas
- Save: `Week3_DocumentSimilarity.ipynb`
**TOTAL WEEK 3: 14 hours**
---

# WEEK 4: MIT 18.06 LECTURES 1-5 - SYSTEMS

## THEORY (4 hours)
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
- Build linear system solver
- Solve multiple equations
- Verify solutions
- Save: `Week4_LinearSystemSolver.ipynb`

---

# WEEK 5: MIT LECTURES 6-10 - RANK & NULLSPACE

## THEORY (3.5 hours)
**MIT Lectures 6-10**
- Topics: Column space, row space, rank, four subspaces, orthogonal, projections

## DEEPLEARNING.AI (3.5 hours)

**Course 1: Linear Algebra (Week 5)**
- Rank and nullspace concepts
- **Labs:** Calculating rank and nullspace

## NUMPY CODING (3 hours)

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

# Rank
rank = np.linalg.matrix_rank(A)
print("Rank:", rank)

# Nullspace
from scipy.linalg import null_space
nullspace = null_space(A)
print("Nullspace basis:", nullspace)

# Projection onto vector a onto vector b
a = np.array([1, 2])
b = np.array([4, 1])
proj = (np.dot(a, b) / np.dot(a, a)) * a
print("Projection:", proj)

# Verify orthogonal
error = b - proj
print("Error:", error)
print("Orthogonal?", np.allclose(np.dot(proj, error), 0))
```

## PROJECT (2.5 hours)
- Complete document analysis recap
- Combine Weeks 1-4 concepts
- Save: `Week5_DocumentAnalysisRecap.ipynb`

---

# WEEK 6: CRITICAL - EIGENVALUES & EIGENVECTORS ⭐

## THEORY (4 hours) ← MOST IMPORTANT WEEK!
**3Blue1Brown + MIT Lecture 21**
- 3Blue1Brown Episode 13-14: https://www.youtube.com/watch?v=PFDu9oVAE-g (17 min)
- MIT Lecture 21: https://www.youtube.com/watch?v=D0CjYvYz2P8 (1h 26min)

**Topics:**
- What is an eigenvector? (special direction)
- What is an eigenvalue? (scaling factor)
- The fundamental equation: A × v = λ × v
- Why eigenvectors matter
- Geometric interpretation
- Characteristic polynomial

**KEY CONCEPT:**
```
Eigenvector v: A direction in space
Eigenvalue λ: How much that direction is scaled by A

A × v = λ × v means:
Applying transformation A to direction v 
just stretches/shrinks v by λ,
doesn't change its direction!
```
## DEEPLEARNING.AI (5 hours)

**Course 1: Linear Algebra (Week 4)**
- Eigenvalues and eigenvectors theory
- **Python lab:** np.linalg.eig() calculations
- **Application:** PCA introduction

**Action:**
- Watch lectures (2.5 hours)
- Complete labs (2.5 hours)

## NUMPY CODING (4 hours) ← MOST IMPORTANT CODING!

```python
import numpy as np

# Create matrix
A = np.array([[5, 0, 6], [0, 5, 4], [6, 4, 14]], dtype=float)

# THE KEY FUNCTION!
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Sort by importance
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate percentages
total = np.sum(eigenvalues)
percentages = (eigenvalues / total) * 100

print("Sorted eigenvalues:", eigenvalues)
print("Percentages:", percentages)

# VERIFY: A × v = λ × v
print("\n" + "="*50)
print("VERIFICATION: A × v = λ × v")
print("="*50)

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    
    left_side = A @ v
    right_side = lam * v
    
    match = np.allclose(left_side, right_side)
    
    print(f"Pattern {i+1}: λ = {lam:.4f}")
    print(f"  A × v = {np.round(left_side, 4)}")
    print(f"  λ × v = {np.round(right_side, 4)}")
    print(f"  Match? {match} ✓")
```

```python
# Exercise 2: Verify A × v = λ × v (90 min)
import numpy as np

A = np.array([[5, 0, 6], [0, 5, 4], [6, 4, 14]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)

# Sort
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("=== VERIFY: A × v = λ × v ===\n")

all_match = True
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    
    print(f"Pattern {i+1}:")
    print(f"  λ = {lam:.4f}")
    print(f"  v = {np.round(v, 4)}")
    
    # Left side: A × v
    left_side = A @ v
    print(f"  A × v = {np.round(left_side, 4)}")
    
    # Right side: λ × v
    right_side = lam * v
    print(f"  λ × v = {np.round(right_side, 4)}")
    
    # Check if equal
    match = np.allclose(left_side, right_side)
    print(f"  Match? {match} {'✓' if match else '✗'}")
    
    diff = np.linalg.norm(left_side - right_side)
    print(f"  Difference: {diff:.2e}")
    print()
    
    all_match = all_match and match

print(f"All eigenvectors verified: {all_match}")

# Save as: week4_exercise2.py
```

```python
# Exercise 3: Interpretation (90 min)
import numpy as np

A = np.array([[5, 0, 6], [0, 5, 4], [6, 4, 14]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# From your document guide
patterns = [
    "Pattern 1 (57%): Document size/content richness",
    "Pattern 2 (29%): Dog vs Cat topic focus",
    "Pattern 3 (14%): Minor variations"
]

print("=== INTERPRETATION ===\n")

for i, pattern in enumerate(patterns):
    print(f"{pattern}")
    v = eigenvectors[:, i]
    print(f"  Eigenvector: {np.round(v, 3)}")
    print(f"  Eigenvalue: {eigenvalues[i]:.1f}")
    print()

# Document interpretation
print("=== DOCUMENT INTERPRETATION ===\n")
print("Doc1 score on Pattern 1:", eigenvectors[0, 0], "→ Smaller document")
print("Doc1 score on Pattern 2:", eigenvectors[0, 1], "→ Dog-focused")
print("Doc1 score on Pattern 3:", eigenvectors[0, 2], "→ Minor pattern\n")

print("Doc2 score on Pattern 1:", eigenvectors[1, 0], "→ Smaller document")
print("Doc2 score on Pattern 2:", eigenvectors[1, 1], "→ Cat-focused")
print("Doc2 score on Pattern 3:", eigenvectors[1, 2], "→ Minor pattern\n")

print("Doc3 score on Pattern 1:", eigenvectors[2, 0], "→ LARGER document")
print("Doc3 score on Pattern 2:", eigenvectors[2, 1], "→ Balanced (0)")
print("Doc3 score on Pattern 3:", eigenvectors[2, 2], "→ Minor pattern")

# Save as: week4_exercise3.py
```

## ASSIGNMENTS (1 hour)

**Assignment 4.1: Eigenvalues**
```python
# File: assignment4_1.py
# Calculate eigenvalues of your matrix
# Sort by importance
# Calculate percentages
```

**Assignment 4.2: Verify Property**
```python
# File: assignment4_2.py
# For 3 different eigenvectors
# Verify A × v = λ × v
# Show calculations step by step
```

**Assignment 4.3: Interpretation (KEY!)**
```python
# File: assignment4_3.py
# Explain what eigenvectors represent
# Explain what eigenvalues represent
# For YOUR document matrix
# Create interpretation document
```

## PROJECT (2 hours)
**Build:** Eigenvalue Analyzer
```python
# File: Week4_EigenvalueAnalyzer.ipynb

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
        # Verify A × v = λ × v for all eigenvectors
        pass
    
    def visualize(self):
        # Visualize eigenvalues and eigenvectors
        pass

# Use your document example
# Analyze patterns
# Create visualizations
```
- Eigenvalue visualization
- Verify fundamental property
- Save: `Week6_EigenvalueVisualization.ipynb`

---

# WEEK 7: DIAGONALIZATION - A = V × Λ × V^-1

## THEORY (3.5 hours)
**MIT Lecture 22**
- Link: https://www.youtube.com/watch?v=13r9QY6g6ES

**Topics:**
- Matrix diagonalization
- Why diagonalization works
- Applications of diagonalization
- Powers of matrices
## DEEPLEARNING.AI (4 hours)

**Course 1: Linear Algebra (Week 5)**
- Diagonalization concepts
- **Python labs:** Reconstruct matrices
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
Vt = V.T

# Step 3: Reconstruct A = V × Λ × V^T
reconstructed = V @ Lambda @ np.linalg.inv(V)

print("Original A:")
print(A)
print("\nReconstructed (V × Λ × V^-1):")
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
## ASSIGNMENTS

**Assignment 5.1-5.3:** (Same pattern as previous weeks)

## PROJECT (2.5 hours)
- Create document embeddings
**Build:** Document Embeddings System
```python
# File: Week5_DocumentEmbeddings.ipynb

class DocumentEmbedding:
    def __init__(self, documents):
        self.documents = documents
        self.matrix = None
        self.embeddings = None
    
    def create_matrix(self):
        # Create word-document matrix
        pass
    
    def diagonalize(self):
        # Apply A = V × Λ × V^-1
        pass
    
    def get_embeddings(self, n_components=3):
        # Get reduced embeddings
        pass
    
    def similarity(self, doc_i, doc_j):
        # Calculate similarity using embeddings
        pass

# Use your document example
# Create embeddings
# Analyze patterns
# Visualize in 2D/3D
```

- Full 7-week application
- Save: `Week7_DocumentEmbeddings.ipynb`

---

# WEEK 8: SINGULAR VALUE DECOMPOSITION (SVD)

## THEORY (3.5 hours)
**Resources:**
- Coursera: https://www.coursera.org/learn/linear-algebra-machine-learning
- MIT Lecture 29-31: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- 3Blue1Brown SVD: https://www.youtube.com/watch?v=mBcgAFivtQg (15 min)

**Topics:**
- SVD formula: A = U × Σ × V^T
- Works for ANY matrix (even non-square!)
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

# Create matrix (can be any shape!)
A = np.array([[2, 0, 3], [1, 0, 0], [0, 2, 2], [0, 1, 0], [0, 0, 1]], dtype=float)

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
- Image compression with SVD
- Compression ratio analysis
- Save: `Week8_ImageCompression.ipynb`

---

# WEEK 9: PRINCIPAL COMPONENT ANALYSIS (PCA)

## THEORY (3.5 hours)
**Resources:**
- StatQuest Video: https://www.youtube.com/watch?v=FD4DeN81ODY
- Coursera ML: https://www.coursera.org/learn/machine-learning
- 3Blue1Brown Basis: https://www.youtube.com/watch?v=P2LTAQQVPA0

**Topics:**
- What is PCA?
- Finding directions of maximum variance
- Covariance matrix
- Eigenvectors of covariance matrix
- Dimensionality reduction
- Standardization importance
- Variance explained ratio
**KEY CONCEPT:**
```
PCA Process:
1. Standardize data (mean=0, std=1)
2. Calculate covariance matrix
3. Find eigenvalues & eigenvectors (Week 6!)
4. Sort by eigenvalues (importance)
5. Project data onto top eigenvectors
6. Reduced dimensions!

Connection to YOUR knowledge:
- Covariance matrix = like similarity matrix (Week 3)
- Eigenanalysis = exactly Week 6
- Projection = Week 7 diagonalization
```

## DEEPLEARNING.AI (2 hours)

**Course 3: Statistics for ML (Week 1)**
- Covariance and correlation
- **Labs:** PCA implementation
- **Application:** Dimensionality reduction
## PATRICK LOEBER (5 hours)

**Video 7: PCA from Scratch**
- Watch video (1 hour)
- Code along from scratch (1.5 hours)
- Understand each step (1 hour)
- Apply to iris dataset (1.5 hours)

## NUMPY CODING (4 hours)

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = iris.data

# PCA from scratch
X_centered = X - np.mean(X, axis=0)
X_std = X_centered / np.std(X_centered, axis=0)

# Covariance matrix
cov_matrix = np.cov(X_std.T)

# Eigenanalysis
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort
idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]

# Variance explained
variance_explained = eigenvalues_sorted / np.sum(eigenvalues_sorted)
print("Variance explained:", variance_explained)

# Project onto top 2 components
n_components = 2
W = eigenvectors_sorted[:, :n_components]
X_reduced = X_std @ W

print(f"Original shape: {X_std.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Information retained: {np.sum(variance_explained[:n_components])*100:.1f}%")

# Using scikit-learn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("\nExplained variance ratio:", pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())
```
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

print("="*60)
print("PCA FROM SCRATCH - YOUR UNDERSTANDING!")
print("="*60)

# Load iris dataset
iris = load_iris()
X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width
y = iris.target

print(f"\nOriginal data shape: {X.shape}")
print("Features: sepal length, sepal width, petal length, petal width")

# STEP 1: Standardize (CRITICAL!)
print("\n" + "="*60)
print("STEP 1: STANDARDIZE DATA")
print("="*60)

X_centered = X - np.mean(X, axis=0)
X_std = X_centered / np.std(X_centered, axis=0)

print("Why standardize?")
print("- Features have different scales")
print("- PCA is sensitive to scale")
print("- Standardization: mean=0, std=1")

# STEP 2: Covariance Matrix
print("\n" + "="*60)
print("STEP 2: COVARIANCE MATRIX")
print("="*60)

cov_matrix = np.cov(X_std.T)

print("Covariance matrix shape:", cov_matrix.shape)
print("Covariance matrix:")
print(np.round(cov_matrix, 2))

print("\nConnection to Week 3:")
print("- Covariance = like similarity")
print("- Measures how features vary together")
print("- Symmetric matrix (like Week 3!)")

# STEP 3: Eigenanalysis (YOUR WEEK 6 KNOWLEDGE!)
print("\n" + "="*60)
print("STEP 3: EIGENANALYSIS (WEEK 6!)")
print("="*60)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors shape:", eigenvectors.shape)

# STEP 4: Sort by importance
print("\n" + "="*60)
print("STEP 4: SORT BY IMPORTANCE")
print("="*60)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]

print("Sorted eigenvalues:", eigenvalues_sorted)

# Calculate variance explained
variance_explained = eigenvalues_sorted / np.sum(eigenvalues_sorted)

print("\n--- VARIANCE EXPLAINED ---")
for i, (lam, var_exp) in enumerate(zip(eigenvalues_sorted, variance_explained)):
    print(f"PC{i+1}: λ = {lam:.4f}, Variance = {var_exp*100:.1f}%")

cumulative = np.cumsum(variance_explained)
print("\n--- CUMULATIVE VARIANCE ---")
for i, cum in enumerate(cumulative):
    print(f"Top {i+1} PCs: {cum*100:.1f}%")

# STEP 5: Project onto top 2 components
print("\n" + "="*60)
print("STEP 5: PROJECT DATA (DIMENSIONALITY REDUCTION)")
print("="*60)

n_components = 2
W = eigenvectors_sorted[:, :n_components]  # Top 2 eigenvectors

print(f"Projection matrix W shape: {W.shape}")
print("W (top 2 eigenvectors):")
print(np.round(W, 3))

X_reduced = X_std @ W

print(f"\nOriginal shape: {X_std.shape} (4D)")
print(f"Reduced shape: {X_reduced.shape} (2D)")
print(f"Information retained: {cumulative[n_components-1]*100:.1f}%")

# COMPARISON WITH SCIKIT-LEARN
print("\n" + "="*60)
print("VERIFY WITH SCIKIT-LEARN")
print("="*60)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())
print("Match with manual?", np.allclose(
    pca.explained_variance_ratio_, 
    variance_explained[:2], 
    atol=0.01
))

# VISUALIZATION
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

plt.figure(figsize=(14, 5))

# Original data (2 features only)
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data (2 features)')
plt.colorbar(label='Species')

# PCA projection
plt.subplot(1, 3, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('PC1 (73.0%)')
plt.ylabel('PC2 (22.9%)')
plt.title('PCA Projection (from 4D to 2D)')
plt.colorbar(label='Species')

# Variance explained
plt.subplot(1, 3, 3)
plt.bar(range(1, 5), variance_explained * 100)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.title('Variance Explained by Each PC')
plt.xticks([1, 2, 3, 4])
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# INTERPRETATION
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

print("PC1 (First Principal Component):")
print("  Explains 73% of variance")
print("  Represents overall flower size")
print("  All features contribute positively")

print("\nPC2 (Second Principal Component):")
print("  Explains 23% of variance")
print("  Contrasts sepal vs petal dimensions")
print("  Helps separate species")

print("\nWith just 2 PCs:")
print(f"  We retain {cumulative[1]*100:.1f}% of information")
print("  But reduce from 4D to 2D!")
print("  Perfect for visualization!")

# CONNECTION TO YOUR DOCUMENT EXAMPLE
print("\n" + "="*60)
print("CONNECTION TO YOUR DOCUMENT EXAMPLE (WEEK 6)")
print("="*60)

print("Your document matrix → similarity matrix → eigenanalysis")
print("Iris features → covariance matrix → eigenanalysis")
print("\nSame process! Just different matrices!")

# Save as: week9_pca_from_scratch.py
```
## PROJECT (2.5 hours)
- Iris dataset visualization

```python
class IrisVisualizer:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = None
        self.X_reduced = None
    
    def fit_transform(self, X):
        # Standardize
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Covariance
        cov = np.cov(X_std.T)
        
        # Eigenanalysis
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Project
        W = self.eigenvectors[:, :self.n_components]
        self.X_reduced = X_std @ W
        
        return self.X_reduced
    
    def explained_variance_ratio(self):
        return self.eigenvalues / np.sum(self.eigenvalues)
    
    def visualize_2d(self, y):
        plt.scatter(self.X_reduced[:, 0], self.X_reduced[:, 1], c=y)
        plt.xlabel(f'PC1 ({self.explained_variance_ratio()[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio()[1]*100:.1f}%)')
        plt.show()

# Use and test
visualizer = IrisVisualizer(n_components=2)
X_reduced = visualizer.fit_transform(iris.data)
visualizer.visualize_2d(iris.target)
```
- Reduce 4D to 2D
- Calculate variance explained
- Save: `Week9_IrisVisualization.ipynb`
**TOTAL WEEK 9: 22 hours**

---

# WEEK 9.5: CALCULUS FUNDAMENTALS

**WHY NOW:** Bridge between linear algebra and optimization

## DEEPLEARNING.AI (5 hours)

**Course 2: Calculus for Machine Learning (Week 1)**
- Derivatives fundamentals
- Gradients
- **Python labs:** Numerical differentiation with NumPy

## YOUR SCHEDULE (2 hours)

**Khan Academy - Calculus Essentials**
- Link: https://www.khanacademy.org/math/calculus-1
- Derivatives, Chain Rule, Optimization

## NUMPY CODING (3 hours)

```python
import numpy as np
import matplotlib.pyplot as plt

# (Full calculus code from previous file - derivatives, gradients, chain rule, optimization)
# See COMPLETE_24WEEK_FULL_SCHEDULE.md for complete code
```

**TOTAL WEEK 9.5: 10.5 hours**

---

# WEEK 10: DOCUMENT EMBEDDINGS & NLP
**WHY NOW:** Combines YOUR document example (Week 3), PCA (Week 9), and prepares for real NLP!


## THEORY (3 hours)
**Resources:**
- Coursera ML: https://www.coursera.org/learn/machine-learning
- TF-IDF Tutorial: https://www.youtube.com/watch?v=4vJJIUhMwkI

**Topics:**
- Bag of words representation
- TF-IDF weighting
- Cosine similarity
- Document similarity
- Vector space model
**KEY CONCEPT:**
```
YOUR Document Example Evolution:

Week 3: Simple word counts (dog, cat, barks, meows, play)
        → Dot product similarity
        → 2×3=6 insight!

Week 10: TF-IDF weights (more sophisticated)
         → Still uses dot product/cosine similarity
         → Real NLP applications!

Connection: Same math, better representation!
```

## DEEPLEARNING.AI (2 hours)

**Course 3: Statistics (Week 2)**
- Text representation
- Feature engineering

## PATRICK LOEBER (4 hours)

**Algorithms 3-4:**
- Logistic Regression (2 hours)
- Code from scratch (2 hours)

## NUMPY CODING (5 hours)

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Documents
documents = [
    "dog dog barks",
    "cat cat meows",
    "dog dog dog cat cat play"
]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents).toarray()

print("TF-IDF matrix:")
print(tfidf_matrix)

# Cosine similarity
similarity = cosine_similarity(tfidf_matrix)
print("\nSimilarity matrix:")
print(similarity.round(3))

# Find similar documents
query_idx = 0
similar_docs = similarity[query_idx].argsort()[::-1][1:]

print(f"\nMost similar to Doc {query_idx}:")
for rank, doc_idx in enumerate(similar_docs[:2], 1):
    score = similarity[query_idx, doc_idx]
    print(f"  {rank}. Doc {doc_idx} (similarity: {score:.3f})")
```
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("DOCUMENT EMBEDDINGS - EVOLUTION OF YOUR EXAMPLE!")
print("="*60)

# YOUR ORIGINAL DOCUMENTS (WEEK 3)
documents = [
    "dog dog barks",
    "cat cat meows",
    "dog dog dog cat cat play"
]

print("Documents:")
for i, doc in enumerate(documents):
    print(f"  Doc{i}: {doc}")

# METHOD 1: Simple Word Counts (YOUR WEEK 3 APPROACH!)
print("\n" + "="*60)
print("METHOD 1: SIMPLE WORD COUNTS (WEEK 3)")
print("="*60)

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents).toarray()

print("Vocabulary:", count_vectorizer.get_feature_names_out())
print("\nWord count matrix:")
print(count_matrix)

# YOUR ORIGINAL CALCULATION
print("\n--- YOUR WEEK 3 INSIGHT ---")
print("Doc0 has 2 'dog', Doc2 has 3 'dog'")
print("Similarity = 2 × 3 = 6 ✓")

# Verify
doc0 = count_matrix[0]
doc2 = count_matrix[2]
similarity_0_2 = np.dot(doc0, doc2)
print(f"Calculated similarity: {similarity_0_2}")

# METHOD 2: TF-IDF (SOPHISTICATED!)
print("\n" + "="*60)
print("METHOD 2: TF-IDF WEIGHTS (WEEK 10)")
print("="*60)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents).toarray()

print("TF-IDF matrix (weighted!):")
print(np.round(tfidf_matrix, 3))

print("\nWhat is TF-IDF?")
print("TF (Term Frequency): How often word appears in document")
print("IDF (Inverse Document Frequency): How rare word is across all documents")
print("TF-IDF = TF × IDF")
print("→ Common words (like 'the') get lower weight")
print("→ Rare, informative words get higher weight")

# Similarity with TF-IDF
similarity = cosine_similarity(tfidf_matrix)

print("\n" + "="*60)
print("SIMILARITY COMPARISON")
print("="*60)

# Simple count similarity
count_sim = cosine_similarity(count_matrix)

print("Simple count similarity:")
print(np.round(count_sim, 3))

print("\nTF-IDF similarity:")
print(np.round(similarity, 3))

print("\nDifference:")
print("- Simple counts: treats all words equally")
print("- TF-IDF: weights by importance")

# SEARCH FUNCTION
print("\n" + "="*60)
print("DOCUMENT SEARCH ENGINE")
print("="*60)

def search_documents(query, documents, vectorizer, tfidf_matrix, top_k=2):
    """Search for most similar documents to query"""
    
    # Transform query to same space
    query_vec = vectorizer.transform([query]).toarray()[0]
    
    # Calculate similarity
    similarities = cosine_similarity([query_vec], tfidf_matrix)[0]
    
    # Get top k
    top_idx = similarities.argsort()[::-1][:top_k]
    
    print(f"\nQuery: '{query}'")
    print("Top matches:")
    for rank, doc_idx in enumerate(top_idx, 1):
        score = similarities[doc_idx]
        print(f"  {rank}. Doc{doc_idx} (similarity: {score:.3f})")
        print(f"      '{documents[doc_idx]}'")
    
    return top_idx

# Test searches
search_documents("dog barks", documents, tfidf_vectorizer, tfidf_matrix)
search_documents("cat", documents, tfidf_vectorizer, tfidf_matrix)
search_documents("play animals", documents, tfidf_vectorizer, tfidf_matrix)

# LARGER EXAMPLE
print("\n" + "="*60)
print("LARGER DOCUMENT COLLECTION")
print("="*60)

larger_docs = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A fast brown fox runs beside a lazy dog",
    "The cat meows loudly at night",
    "Cats and dogs are popular pets",
    "Machine learning uses mathematical algorithms",
    "Neural networks learn from data",
    "Deep learning is a subset of machine learning"
]

print(f"Number of documents: {len(larger_docs)}")

# TF-IDF
large_vectorizer = TfidfVectorizer()
large_tfidf = large_vectorizer.fit_transform(larger_docs).toarray()

print(f"TF-IDF matrix shape: {large_tfidf.shape}")
print(f"Vocabulary size: {len(large_vectorizer.get_feature_names_out())}")

# Search
print("\n--- SEARCH RESULTS ---")
search_documents("dog", larger_docs, large_vectorizer, large_tfidf, top_k=3)
search_documents("machine learning", larger_docs, large_vectorizer, large_tfidf, top_k=3)

# COMBINE WITH PCA (WEEK 9!)
print("\n" + "="*60)
print("COMBINE WITH PCA (DIMENSIONALITY REDUCTION)")
print("="*60)

from sklearn.decomposition import PCA

# Reduce TF-IDF to 2D
pca = PCA(n_components=2)
docs_2d = pca.fit_transform(large_tfidf)

print(f"Original shape: {large_tfidf.shape}")
print(f"Reduced shape: {docs_2d.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(docs_2d[:, 0], docs_2d[:, 1], s=100)
for i, doc in enumerate(larger_docs):
    plt.annotate(f'Doc{i}', docs_2d[i], fontsize=8)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Document Embeddings (TF-IDF + PCA)')
plt.grid(True, alpha=0.3)
plt.show()

print("\nObservations:")
print("- Dog/fox documents cluster together (left)")
print("- Machine learning documents cluster together (right)")
print("- Your Week 3 insight + PCA = Modern NLP!")

# Save as: week10_document_embeddings.py
```
## ZERO TO MASTERY (8 hours)

**Sections 4-5:**
- ML Fundamentals (4 hours)
- Scikit-Learn Basics (4 hours)

## ASSIGNMENTS (1 hour)

**Assignment 9.1: PCA from Scratch**
- Implement on iris dataset
- Calculate variance explained
- Verify with scikit-learn

**Assignment 9.2: 3D Projection**
- Keep top 3 components
- Visualize in 3D
- Compare with 2D

**Assignment 9.3: Your Document PCA**
- Apply PCA to your document example
- Compare with eigenanalysis from Week 6
- What patterns do you find?
**Assignment 10.1:** Your documents with TF-IDF
**Assignment 10.2:** Search engine
**Assignment 10.3:** Combine with PCA
## PROJECT  (2.5 hours)
- Build document search engine
**Build:** Document Search Engine
```python
class DocumentSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents).toarray()
        
        # PCA for visualization
        self.pca = PCA(n_components=2)
        self.embeddings = self.pca.fit_transform(self.tfidf_matrix)
    
    def search(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        similarities = cosine_similarity([query_vec], self.tfidf_matrix)[0]
        top_idx = similarities.argsort()[::-1][:top_k]
        return [(i, self.documents[i], similarities[i]) for i in top_idx]
    
    def visualize(self):
        plt.scatter(self.embeddings[:, 0], self.embeddings[:, 1])
        for i in range(len(self.documents)):
            plt.annotate(f'Doc{i}', self.embeddings[i])
        plt.show()

# Use and test
engine = DocumentSearchEngine(larger_docs)
results = engine.search("machine learning")
engine.visualize()
```

**Save:** `Week10_DocumentSearchEngine.ipynb`
**TOTAL WEEK 10: 20 hours**
---

# WEEK 11: RECOMMENDATION SYSTEMS

**WHY NOW:** Uses SVD (Week 8) + similarity (Week 3) + real applications!

## THEORY (3.5 hours)
**Resources:**
- Coursera: https://www.coursera.org/learn/recommender-systems
- YouTube: https://www.youtube.com/watch?v=n3RKsY2H-js

**Topics:**
- Collaborative filtering
- Matrix factorization with SVD
- User-item matrix
- Latent factors
- Predicting missing ratings
**KEY CONCEPT:**
```
Your SVD Knowledge (Week 8):

A = U × Σ × V^T

For recommendations:
- A = user-item ratings matrix
- U = user features (latent)
- Σ = importance weights
- V^T = item features (latent)

Reconstruct A to predict missing ratings!
```

## PATRICK LOEBER (5 hours)

**Algorithms 5-6:**
- Decision Trees (2.5 hours)
- Random Forest (2.5 hours)

## ZERO TO MASTERY (8 hours)

**Sections 6-7:**
- Supervised Learning (4 hours)
- Model Evaluation (4 hours)

## NUMPY CODING (4 hours)

```python
import numpy as np
import pandas as pd

# User-item ratings (1-5 stars, 0 = not rated)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
], dtype=float)

# SVD for recommendations
U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

# Reconstruct (predicts missing values)
ratings_pred = U @ np.diag(s) @ Vt

print("Original ratings:")
print(ratings)
print("\nPredicted ratings:")
print(ratings_pred.round(1))

# Get recommendations for user 0
user_id = 0
unrated_idx = np.where(ratings[user_id] == 0)[0]

predictions = [(idx, ratings_pred[user_id, idx]) for idx in unrated_idx]
predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

print(f"\nTop 2 recommendations for User {user_id}:")
for movie_idx, score in predictions[:2]:
    print(f"  Movie {movie_idx}: Predicted rating {score:.1f}/5")
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("RECOMMENDATION SYSTEMS WITH SVD (YOUR WEEK 8!)")
print("="*60)

# User-item ratings (1-5 stars, 0 = not rated)
ratings = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 0],
    [0, 1, 5, 4, 0],
    [5, 4, 0, 0, 1],
], dtype=float)

users = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank']
movies = ['Inception', 'Titanic', 'Matrix', 'Avatar', 'Interstellar']

print("User-Item Ratings Matrix:")
df = pd.DataFrame(ratings, index=users, columns=movies)
print(df)
print("\n0 = not rated yet")

# SVD FOR MATRIX FACTORIZATION
print("\n" + "="*60)
print("SVD DECOMPOSITION (WEEK 8 KNOWLEDGE!)")
print("="*60)

U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

print(f"U shape: {U.shape} (users × latent factors)")
print(f"s shape: {s.shape} (latent factor importance)")
print(f"Vt shape: {Vt.shape} (latent factors × movies)")

print("\nLatent factors:")
print("- Hidden patterns in data")
print("- e.g., 'action vs romance', 'old vs new', etc.")
print("- SVD discovers these automatically!")

# Singular values importance
print("\n--- SINGULAR VALUES ---")
for i, sv in enumerate(s):
    print(f"σ_{i+1} = {sv:.2f}")

# RECONSTRUCT TO PREDICT
print("\n" + "="*60)
print("RECONSTRUCT MATRIX (PREDICT MISSING RATINGS)")
print("="*60)

# Full reconstruction
ratings_pred = U @ np.diag(s) @ Vt

print("Original ratings:")
print(df)

print("\nPredicted ratings:")
df_pred = pd.DataFrame(np.round(ratings_pred, 1), index=users, columns=movies)
print(df_pred)

print("\nHow it works:")
print("- SVD fills in 0s (missing ratings)")
print("- Based on similar users & items")
print("- Uses latent factor patterns")

# GET RECOMMENDATIONS
print("\n" + "="*60)
print("GET RECOMMENDATIONS")
print("="*60)

def get_recommendations(user_id, ratings, ratings_pred, users, movies, top_k=3):
    """Get top movie recommendations for a user"""
    
    user_idx = users.index(user_id)
    
    # Find unrated movies
    unrated_idx = np.where(ratings[user_idx] == 0)[0]
    
    # Get predicted ratings for unrated movies
    predictions = [(idx, movies[idx], ratings_pred[user_idx, idx]) 
                   for idx in unrated_idx]
    
    # Sort by predicted rating
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
    
    print(f"\nTop {top_k} recommendations for {user_id}:")
    for rank, (idx, movie, score) in enumerate(predictions[:top_k], 1):
        print(f"  {rank}. {movie}: Predicted rating {score:.1f}/5")
    
    return predictions[:top_k]

# Test recommendations
get_recommendations('Alice', ratings, ratings_pred, users, movies)
get_recommendations('Bob', ratings, ratings_pred, users, movies)
get_recommendations('Charlie', ratings, ratings_pred, users, movies)

# LOW-RANK APPROXIMATION
print("\n" + "="*60)
print("LOW-RANK APPROXIMATION (DIMENSIONALITY REDUCTION)")
print("="*60)

# Use only top k factors
k = 2
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

ratings_k = U_k @ np.diag(s_k) @ Vt_k

print(f"Using only top {k} factors:")
df_k = pd.DataFrame(np.round(ratings_k, 1), index=users, columns=movies)
print(df_k)

# Compare
print(f"\nWith {k} factors:")
print(f"- Simpler model")
print(f"- Still captures main patterns")
print(f"- Prevents overfitting")

# EVALUATE
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

# Calculate error on known ratings
mask = ratings > 0
known_ratings = ratings[mask]
pred_known = ratings_pred[mask]

mae = np.mean(np.abs(known_ratings - pred_known))
rmse = np.sqrt(np.mean((known_ratings - pred_known)**2))

print(f"Mean Absolute Error: {mae:.3f}")
print(f"Root Mean Squared Error: {rmse:.3f}")

# VISUALIZE LATENT FACTORS
print("\n" + "="*60)
print("VISUALIZE LATENT FACTORS")
print("="*60)

plt.figure(figsize=(12, 5))

# User latent factors
plt.subplot(1, 2, 1)
plt.scatter(U[:, 0], U[:, 1], s=100)
for i, user in enumerate(users):
    plt.annotate(user, (U[i, 0], U[i, 1]))
plt.xlabel('Latent Factor 1')
plt.ylabel('Latent Factor 2')
plt.title('User Preferences in Latent Space')
plt.grid(True, alpha=0.3)

# Movie latent factors
plt.subplot(1, 2, 2)
plt.scatter(Vt[0, :], Vt[1, :], s=100)
for i, movie in enumerate(movies):
    plt.annotate(movie, (Vt[0, i], Vt[1, i]), fontsize=8)
plt.xlabel('Latent Factor 1')
plt.ylabel('Latent Factor 2')
plt.title('Movie Features in Latent Space')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("- Users close together have similar tastes")
print("- Movies close together are similar types")
print("- Distance predicts rating!")

# CONNECTION TO YOUR KNOWLEDGE
print("\n" + "="*60)
print("CONNECTION TO YOUR KNOWLEDGE")
print("="*60)

print("Week 8: Learned SVD decomposition")
print("Week 11: Applied SVD to recommendations")
print("\nSame math, real application!")
print("\nOther SVD applications:")
print("- Image compression (Week 8)")
print("- Document analysis (Week 10)")
print("- Recommendation systems (Week 11)")
print("- Latent Semantic Analysis")
print("- Collaborative filtering")

# Save as: week11_recommendation_systems.py
```
## ASSIGNMENTS (1 hour)

**Assignment 11.1:** Basic recommender
**Assignment 11.2:** Low-rank approximation
**Assignment 11.3:** Evaluation metrics
## PROJECT (2.5 hours)
- Movie recommendation system
**Build:** Movie Recommendation System
```python
class MovieRecommender:
    def __init__(self, ratings_df):
        self.ratings = ratings_df.values
        self.users = ratings_df.index.tolist()
        self.movies = ratings_df.columns.tolist()
        
        # SVD
        self.U, self.s, self.Vt = np.linalg.svd(self.ratings, full_matrices=False)
        self.ratings_pred = self.U @ np.diag(self.s) @ self.Vt
    
    def recommend(self, user_id, top_k=5):
        user_idx = self.users.index(user_id)
        unrated = np.where(self.ratings[user_idx] == 0)[0]
        
        pred_scores = [(self.movies[i], self.ratings_pred[user_idx, i]) 
                       for i in unrated]
        return sorted(pred_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def similar_users(self, user_id, top_k=3):
        user_idx = self.users.index(user_id)
        user_vec = self.U[user_idx]
        
        similarities = [np.dot(user_vec, self.U[i]) for i in range(len(self.users))]
        similar_idx = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [(self.users[i], similarities[i]) for i in similar_idx]
    
    def evaluate(self):
        mask = self.ratings > 0
        mae = np.mean(np.abs(self.ratings[mask] - self.ratings_pred[mask]))
        rmse = np.sqrt(np.mean((self.ratings[mask] - self.ratings_pred[mask])**2))
        return mae, rmse

# Use and test
recommender = MovieRecommender(df)
recommendations = recommender.recommend('Alice')
similar = recommender.similar_users('Alice')
mae, rmse = recommender.evaluate()
```

**Save:** `Week11_MovieRecommender.ipynb`

**TOTAL WEEK 11: 20 hours**

---

# WEEK 12: APPLIED LINEAR ALGEBRA IN ML

**WHY NOW:** Brings everything together before gradient descent!

## THEORY (3 hours)

**Topics:**
- Linear regression from scratch (uses matrix operations)
- Normal equation vs gradient descent
- Neural networks (matrix operations)
- Word embeddings (matrix factorization)

## DEEPLEARNING.AI (2 hours)

**Course 2: Calculus (Week 2)**
- Optimization preview

## PATRICK LOEBER (5 hours)

**Algorithms 7-8:**
- K-Means Clustering (2.5 hours)
- Naive Bayes (2.5 hours)

## ZERO TO MASTERY (8 hours)

**Sections 8-9:**
- Ensemble Methods (4 hours)
- Model Selection (4 hours)

## NUMPY CODING (5 hours)

```python
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("APPLIED LINEAR ALGEBRA IN MACHINE LEARNING")
print("="*60)

# LINEAR REGRESSION FROM SCRATCH
print("\n" + "="*60)
print("LINEAR REGRESSION - TWO METHODS")
print("="*60)

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 2.5 * X.ravel() + 1.0 + np.random.randn(100) * 2

print(f"Data: {len(X)} samples")
print(f"True relationship: y = 2.5x + 1")

# METHOD 1: Normal Equation (uses Week 4!)
print("\n--- METHOD 1: NORMAL EQUATION ---")
print("Uses: (X^T X)^-1 X^T y")
print("Connections: Week 4 (linear systems)")

X_with_bias = np.column_stack([np.ones(len(X)), X])

# β = (X^T X)^-1 X^T y
XtX = X_with_bias.T @ X_with_bias
Xty = X_with_bias.T @ y
beta = np.linalg.solve(XtX, Xty)

print(f"\nCoefficients: bias = {beta[0]:.2f}, slope = {beta[1]:.2f}")
print(f"Close to true (1.0, 2.5)? {np.allclose([beta[0], beta[1]], [1.0, 2.5], atol=0.5)}")

y_pred_normal = X_with_bias @ beta

# METHOD 2: Gradient Descent (preview of Week 13!)
print("\n--- METHOD 2: GRADIENT DESCENT ---")
print("Preview of next weeks!")

def gradient_descent(X, y, lr=0.01, iterations=1000):
    X = np.column_stack([np.ones(len(y)), X])
    w = np.zeros(X.shape[1])
    m = len(y)
    
    losses = []
    
    for i in range(iterations):
        y_pred = X @ w
        error = y_pred - y
        loss = np.mean(error**2)
        losses.append(loss)
        
        gradient = (1/m) * X.T @ error
        w = w - lr * gradient
        
        if (i + 1) % 200 == 0:
            print(f"Iteration {i+1}: Loss = {loss:.2f}")
    
    return w, losses

w_gd, losses = gradient_descent(X, y, lr=0.01, iterations=1000)
print(f"\nCoefficients: bias = {w_gd[0]:.2f}, slope = {w_gd[1]:.2f}")

y_pred_gd = X_with_bias @ w_gd

# COMPARE
print("\n--- COMPARISON ---")
print(f"Normal Equation: {beta}")
print(f"Gradient Descent: {w_gd}")
print(f"Match? {np.allclose(beta, w_gd, atol=0.1)}")

# Visualize
plt.figure(figsize=(12, 4))

# Data and fits
plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_normal, 'r-', linewidth=2, label='Normal Eq')
plt.plot(X, y_pred_gd, 'g--', linewidth=2, label='Grad Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression: Two Methods')
plt.legend()
plt.grid(True, alpha=0.3)

# Gradient descent convergence
plt.subplot(1, 3, 2)
plt.plot(losses, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Gradient Descent Convergence')
plt.grid(True, alpha=0.3)

# Error comparison
plt.subplot(1, 3, 3)
errors_normal = y - y_pred_normal
errors_gd = y - y_pred_gd
plt.hist(errors_normal, alpha=0.5, label='Normal Eq', bins=20)
plt.hist(errors_gd, alpha=0.5, label='Grad Descent', bins=20)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# NEURAL NETWORK BASICS (Matrix operations!)
print("\n" + "="*60)
print("NEURAL NETWORK BASICS (MATRIX OPERATIONS)")
print("="*60)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Simple 2-layer network
X_nn = np.random.randn(5, 3)  # 5 samples, 3 features
W1 = np.random.randn(3, 4)    # 3 input, 4 hidden
b1 = np.zeros(4)
W2 = np.random.randn(4, 1)    # 4 hidden, 1 output
b2 = 0

# Forward pass (all matrix operations!)
Z1 = X_nn @ W1 + b1  # Hidden layer
A1 = sigmoid(Z1)
Z2 = A1 @ W2 + b2    # Output layer
A2 = sigmoid(Z2)

print("Input shape:", X_nn.shape)
print("Hidden layer shape:", A1.shape)
print("Output shape:", A2.shape)
print("\nAll operations are matrix multiplications!")
print("This is why linear algebra is crucial for ML!")

# SUMMARY
print("\n" + "="*60)
print("SUMMARY: YOUR LINEAR ALGEBRA JOURNEY")
print("="*60)

summary = """
Week 1: Vectors → Feature representations
Week 2: Matrices → Data transformations
Week 3: Dot Product → Similarity (2×3=6!)
Week 4: Linear Systems → Normal equation
Week 5: Rank & Nullspace → Data analysis
Week 6: Eigenvalues → PCA (Week 9!)
Week 7: Diagonalization → Dimensionality reduction
Week 8: SVD → Recommendations (Week 11!)
Week 9: PCA → Real application!
Week 10: NLP → Your document example evolved!
Week 11: Recommendations → SVD applied!
Week 12: ML Pipeline → Everything combined!
"""

print(summary)

print("Next: Calculus & Gradient Descent!")
print("Then: Algorithms, Deep Learning, Production ML!")

# Save as: week12_applied_linear_algebra.py
```

## ASSIGNMENTS (1 hour)

**Assignment 12.1:** Linear regression both methods
**Assignment 12.2:** Neural network forward pass
**Assignment 12.3:** Combine everything from Weeks 1-12

## PROJECT (2 hours)

**Build:** ML Pipeline
```python
class MLPipeline:
    def __init__(self):
        self.models = {}
    
    def linear_regression_normal(self, X, y):
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.solve(X.T @ X, X.T @ y)
        self.models['linear_normal'] = beta
        return beta
    
    def linear_regression_gd(self, X, y, lr=0.01, iterations=1000):
        w, losses = gradient_descent(X, y, lr, iterations)
        self.models['linear_gd'] = w
        return w, losses
    
    def predict(self, X, model_name='linear_normal'):
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.models[model_name]
    
    def evaluate(self, y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Use
pipeline = MLPipeline()
beta = pipeline.linear_regression_normal(X, y)
w, losses = pipeline.linear_regression_gd(X, y)
y_pred = pipeline.predict(X)
metrics = pipeline.evaluate(y, y_pred)
```

**Save:** `Week12_MLPipeline.ipynb`

**TOTAL WEEK 12: 20 hours**

---

# WEEKS 13-16: GRADIENT DESCENT, OPTIMIZATION & CAPSTONES

# WEEK 13: GRADIENT DESCENT FROM SCRATCH

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

**TOTAL WEEK 13: 18 hours**
---

# WEEK 14: ADVANCED OPTIMIZATION & ALGORITHMS

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

**TOTAL WEEK 14: 21 hours**

---
# WEEK 15: MACHINE LEARNING APPLICATIONS

**Combines:** Patrick Loeber (Algorithms) + Zero to Mastery (Tools) + Your understanding (Math)

- Algorithms 5-10 (Decision Trees, Random Forest, KNN, SVM, etc.)
- Zero to Mastery sections 4-6
- Full ML pipeline projects
- Real datasets and evaluation metrics

**TOTAL WEEK 15: 20 hours**

---
# WEEK 16: CAPSTONE 1 - COMPLETE NLP SYSTEM

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

**TOTAL WEEK 16: 25 hours**

---
# WEEK 17: OPTIMIZATION & PERFORMANCE

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

**TOTAL WEEK 17: 20 hours**

---

# WEEK 18: LARGE-SCALE COMPUTING

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

**TOTAL WEEK 18: 20 hours**

---

# WEEK 19: CAPSTONE 2 - YOUR CHOICE PROJECT

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

**TOTAL WEEK 19: 28 hours**

---

# WEEKS 20-27: EXTENDED SPECIALIZATION (OPTIONAL)

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
| 3 | Transpose/Dot ⭐ | 3 | 4 | - | - | 14 |
| 4 | Systems | 4 | 4 | - | - | 15 |
| 5 | Rank | 3.5 | 3.5 | - | - | 13.5 |
| 6 | Eigenvalues ⭐ | 4 | 5 | - | - | 15 |
| 7 | Diagonalization | 3.5 | 4 | - | - | 13.5 |
| 8 | SVD | 3.5 | 4 | - | - | 14 |
| **9** | **PCA ⭐** | - | 2 | 5 | 8 | 22 |
| 9.5 | Calculus | 2 | 5 | - | - | 10.5 |
| **10** | **Doc Embeddings** | - | 2 | 4 | 8 | 20 |
| **11** | **Recommendations** | - | - | 5 | 8 | 20 |
| **12** | **Applied ML** | - | 2 | 5 | 8 | 20 |
| 13 | Grad Descent | - | 4 | 4 | - | 18 |
| 14 | Adv Optim | - | 3 | 5 | 8 | 21 |
| 15 | Capstone 1 | 10 | - | 5 | 10 | 25 |
| 16 | Capstone 2 | 8 | - | 2 | 18 | 28 |
| **TOTAL** | | **70** | **58** | **48** | **92** | **397 hrs** |

---

# COMPLETE RESOURCE DIRECTORY

## Main Courses
- DeepLearning.AI: https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science
- Patrick Loeber: https://www.youtube.com/@patloeber
- Zero to Mastery: https://zerotomastery.io/courses/machine-learning-and-data-science-bootcamp/

## Supplementary (FREE)
- 3Blue1Brown: https://www.youtube.com/c/3blue1brown
- MIT 18.06: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- Khan Academy: https://www.khanacademy.org
- StatQuest: https://www.youtube.com/@statquest

## Tools
- Anaconda: https://www.anaconda.com/download
- VS Code: https://code.visualstudio.com
- GitHub: https://github.com
- Google Colab: https://colab.research.google.com

---

# YOUR COMPLETE LEARNING JOURNEY

✅ **Weeks 1-8 (111 hours):** Linear Algebra Foundations
- Master vectors, matrices, eigenvalues, SVD

✅ **Week 9 (22 hours):** PCA - First Real Application!
- Combines Week 6 eigenvalues + real use case

✅ **Week 9.5 (10.5 hours):** Calculus Bridge
- Derivatives, gradients, optimization theory

✅ **Week 10 (20 hours):** Document Embeddings & NLP
- Your Week 3 example evolved with TF-IDF

✅ **Week 11 (20 hours):** Recommendation Systems
- Your Week 8 SVD applied to real problem

✅ **Week 12 (20 hours):** Applied ML Pipeline
- Everything from Weeks 1-11 combined

✅ **Weeks 13-16 (92 hours):** Gradient Descent & Capstones
- Optimization algorithms
- Real-world projects

**TOTAL: 397 hours → Professional ML/AI Engineer!** 🎓

---

# START TODAY!

**Week 1, Day 1:**
1. Download Anaconda
2. Watch 3Blue1Brown Episode 1 (10 min)
3. Create your first vector
4. Start journey to mastery

**By Week 12: You'll understand not just HOW but WHY ML works!**
**By Week 16: You'll have a professional portfolio!**

🚀 Download and start Week 1 today! 📚💻🎓