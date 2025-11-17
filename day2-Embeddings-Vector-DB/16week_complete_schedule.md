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

## NUMPY CODING (4 hours)
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

## PROJECT (2 hours)
- Build vector calculator
- Test: addition, subtraction, scaling, dot product
- Save: `Week1_VectorCalculator.ipynb`

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

## PROJECT (2.5 hours)
- Image rotation using matrices
- Transform shapes geometrically
- Visualize before/after
- Save: `Week2_ImageRotation.ipynb`

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

## PROJECT (2.5 hours)
- Calculate document similarity matrix
- Implement 2×3=6 correctly!
- Display with Pandas
- Save: `Week3_DocumentSimilarity.ipynb`

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

## PROJECT (2 hours)
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

## PROJECT (2.5 hours)
- Create document embeddings
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

## PROJECT (2.5 hours)
- Iris dataset visualization
- Reduce 4D to 2D
- Calculate variance explained
- Save: `Week9_IrisVisualization.ipynb`

---

# WEEK 10: DOCUMENT EMBEDDINGS & NLP

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

## PROJECT (2.5 hours)
- Build document search engine
- Find most similar documents
- Save: `Week10_DocumentSearchEngine.ipynb`

---

# WEEK 11: RECOMMENDATION SYSTEMS

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

## PROJECT (2.5 hours)
- Movie recommendation system
- Predict unrated movies
- Save: `Week11_MovieRecommender.ipynb`

---

# WEEK 12: APPLIED LINEAR ALGEBRA IN ML

## THEORY (3 hours)
**Topics:**
- Linear regression from scratch
- Gradient descent (matrix calculus)
- Neural networks (matrix operations)
- Word embeddings

## NUMPY CODING (5 hours)

```python
import numpy as np

# Linear regression from scratch
X = np.random.randn(100, 1) * 10
y = 2.5 * X.ravel() + 1.0 + np.random.randn(100) * 2

# Add bias term
X_with_bias = np.column_stack([np.ones(len(X)), X])

# Normal equation: β = (X^T X)^-1 X^T y
beta = np.linalg.solve(X_with_bias.T @ X_with_bias, 
                       X_with_bias.T @ y)

y_pred = X_with_bias @ beta

print("Coefficients:", beta)
print("Predictions match?", np.allclose(y_pred[:5], y[:5], atol=0.5))

# Gradient descent
def gradient_descent(X, y, lr=0.01, iterations=1000):
    X = np.column_stack([np.ones(len(y)), X])
    w = np.zeros(X.shape[1])
    m = len(y)
    
    for i in range(iterations):
        y_pred = X @ w
        error = y_pred - y
        w = w - (lr/m) * X.T @ error
    
    return w

w_gd = gradient_descent(X, y)
y_pred_gd = X_with_bias @ w_gd
print("Gradient descent coefficients:", w_gd)
```

## PROJECT (2 hours)
- Build ML pipeline
- Linear regression + classification
- Save: `Week12_MLPipeline.ipynb`

---

# WEEK 13: CAPSTONE PROJECT 1 - NLP ANALYSIS

## PROJECT: COMPLETE NLP PIPELINE

**Goal:** Combine everything from Weeks 1-12

```python
# What to build:
# 1. Load documents
# 2. Create TF-IDF vectors
# 3. Apply PCA dimensionality reduction
# 4. Calculate similarity
# 5. Cluster documents
# 6. Visualize results

# Your system should:
# - Process documents
# - Create embeddings
# - Find similar documents
# - Visualize relationships
# - Handle edge cases
```

**Deliverable:** `Week13_CapstoneNLP.ipynb`

---

# WEEK 14: OPTIMIZATION & PERFORMANCE

## THEORY (3 hours)
**Topics:**
- Vectorization vs loops
- NumPy broadcasting
- Memory efficiency
- Sparse matrices
- Parallel computing

## NUMPY CODING (4 hours)

```python
import numpy as np
import time

# Broadcasting
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = np.array([10, 20, 30])            # (3,)
result = A + b                         # Broadcasting!

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

## PROJECT (2.5 hours)
- Optimize previous projects
- Measure improvements
- Save: `Week14_Optimization.ipynb`

---

# WEEK 15: LARGE-SCALE COMPUTING

## THEORY (3.5 hours)
**Topics:**
- Randomized SVD
- Incremental learning
- GPU computing basics
- Distributed computing

## NUMPY CODING (4 hours)

```python
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import IncrementalPCA
import time

# Large matrix
n, m = 10000, 5000
A = np.random.randn(n, m)

# Standard SVD
start = time.time()
U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=False)
time_full = time.time() - start

# Randomized SVD
start = time.time()
U_rand, s_rand, Vt_rand = svds(A, k=100)
time_rand = time.time() - start

print(f"Standard SVD time: {time_full:.4f}s")
print(f"Randomized SVD time: {time_rand:.4f}s")
print(f"Speedup: {time_full/time_rand:.1f}x")

# Incremental PCA
ipca = IncrementalPCA(n_components=10)
for batch in range(5):
    X_batch = np.random.randn(1000, 100)
    ipca.partial_fit(X_batch)

print(f"Explained variance: {ipca.explained_variance_ratio_.sum()*100:.1f}%")
```

## PROJECT (2 hours)
- Process large datasets
- Compare SVD approaches
- Save: `Week15_LargeScale.ipynb`

---

# WEEK 16: CAPSTONE PROJECT 2 - YOUR CHOICE

## CHOOSE ONE:

### Option 1: Advanced Recommendation System
- Handle millions of ratings
- Handle cold start problem
- Production-ready code
- Proper evaluation

### Option 2: Image Compression & Analysis
- Compress images using SVD
- Analyze quality vs compression
- Visualize trade-offs
- Compare approaches

### Option 3: Stock Portfolio Analysis
- Price prediction
- Risk analysis (eigenanalysis)
- Correlation matrices
- Portfolio optimization

### Option 4: Large-Scale NLP
- Process large corpus
- Topic modeling
- Semantic search
- Performance optimized

## REQUIREMENTS:

✅ Use NumPy for calculations
✅ Apply 3+ concepts (matrices, eigenanalysis, SVD/PCA, applications)
✅ Include documentation
✅ Visualizations
✅ Performance metrics
✅ Production-ready code

**Deliverable:** `Week16_CapstoneProject.ipynb` + `README.md`

---

# SUMMARY

| Week | Focus | Project |
|------|-------|---------|
| 1 | Vectors | Vector Calculator |
| 2 | Matrices | Image Rotation |
| 3 | Transpose & Dot Product | Document Similarity |
| 4 | Systems | Linear Solver |
| 5 | Rank & Nullspace | Document Analysis |
| 6 | **EIGENVALUES** | Eigenvalue Viz |
| 7 | Diagonalization | Embeddings |
| 8 | SVD | Image Compression |
| 9 | PCA | Iris Visualization |
| 10 | NLP | Search Engine |
| 11 | Recommendations | Movie Recommender |
| 12 | ML Applications | ML Pipeline |
| 13 | Capstone 1 | NLP System |
| 14 | Optimization | Performance |
| 15 | Large-Scale | Big Data |
| 16 | Capstone 2 | Your Choice |

---

# RESOURCES

**3Blue1Brown:** https://youtube.com/c/3blue1brown
**MIT 18.06:** https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
**DataCamp:** https://datacamp.com ($35/mo)
**Coursera:** https://coursera.org (FREE audit)
**NumPy Docs:** https://numpy.org

---

# START NOW!

**Week 1, Day 1:**
1. Download Anaconda: https://www.anaconda.com/download
2. Watch 3Blue1Brown Episode 1 (10 min)
3. Open Jupyter and create first vector
4. Follow the schedule!

**In 16 weeks: You'll be a NumPy master!** 🚀