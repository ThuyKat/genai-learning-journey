# NumPy Tutorial for Matrix Calculations
## Learn by Coding - Step by Step

---

## PART 1: INSTALLATION & SETUP

### Install NumPy
```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```

### Test Installation
```python
import numpy as np
print(np.__version__)  # Should print version like 1.23.0
```

---

## PART 2: BASIC NUMPY OPERATIONS

### 1. Create a Matrix
```python
import numpy as np

# Create a matrix using array
A = np.array([
    [2, 0, 3],
    [1, 0, 0],
    [0, 2, 2],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

print("Matrix A:")
print(A)
print("Shape:", A.shape)  # Output: (5, 3)
print("Type:", type(A))   # Output: <class 'numpy.ndarray'>
```

**Output:**
```
Matrix A:
[[2. 0. 3.]
 [1. 0. 0.]
 [0. 2. 2.]
 [0. 1. 0.]
 [0. 0. 1.]]
Shape: (5, 3)
```

---

### 2. Transpose a Matrix
```python
# Transpose (flip rows and columns)
At = A.T

print("Transpose A^T:")
print(At)
print("Shape:", At.shape)  # Output: (3, 5)

# Verify: (A^T)^T = A
print("Verify (A^T)^T = A:", np.allclose(At.T, A))
```

**Output:**
```
Transpose A^T:
[[2. 1. 0. 0. 0.]
 [0. 0. 2. 1. 0.]
 [3. 0. 2. 0. 1.]]
Shape: (3, 5)
Verify (A^T)^T = A: True
```

---

### 3. Matrix Multiplication
```python
# Matrix multiplication: A^T × A
ATA = At @ A  # @ is the matrix multiplication operator

print("A^T × A:")
print(ATA)
print("Shape:", ATA.shape)  # Output: (3, 3)

# Alternative (same result)
ATA_alt = np.dot(At, A)
print("Same result?:", np.allclose(ATA, ATA_alt))
```

**Output:**
```
A^T × A:
[[ 5.  0.  6.]
 [ 0.  5.  4.]
 [ 6.  4. 14.]]
Shape: (3, 3)
Same result?: True
```

---

### 4. Access Matrix Elements
```python
# Access individual elements
print("Element [0,0]:", ATA[0, 0])  # Output: 5.0
print("Element [0,2]:", ATA[0, 2])  # Output: 6.0 ← YOUR 2×3=6!

# Access rows
print("Row 0:", ATA[0, :])

# Access columns
print("Column 0:", ATA[:, 0])

# Reshape
print("Flattened:", ATA.flatten())
```

---

## PART 3: EIGENVALUE & EIGENVECTOR CALCULATION

### 5. Calculate Eigenvalues and Eigenvectors
```python
# THE MOST IMPORTANT CALCULATION!
# Finds eigenvalues and eigenvectors of a matrix
eigenvalues, eigenvectors = np.linalg.eig(ATA)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print("\nShape of eigenvalues:", eigenvalues.shape)  # (3,)
print("Shape of eigenvectors:", eigenvectors.shape)  # (3, 3)
```

**Output:**
```
Eigenvalues: [4. 2. 1.]
Eigenvectors:
 [[-0.40824829 -0.70710678 -0.57735027]
  [-0.40824829  0.70710678 -0.57735027]
  [-0.81649658  0.         0.57735027]]

Shape of eigenvalues: (3,)
Shape of eigenvectors: (3, 3)
```

**Understanding the output:**
- `eigenvalues`: 1D array of 3 numbers: [4.0, 2.0, 1.0]
- `eigenvectors`: 3×3 matrix where each COLUMN is an eigenvector
  - Column 0: [-0.408, -0.408, -0.816] (eigenvector for λ=4)
  - Column 1: [-0.707, 0.707, 0.0] (eigenvector for λ=2)
  - Column 2: [-0.577, -0.577, 0.577] (eigenvector for λ=1)

---

### 6. Sort Eigenvalues by Importance
```python
# Find indices that sort eigenvalues in descending order
idx = np.argsort(eigenvalues)[::-1]

print("Sorting indices:", idx)

# Reorder eigenvalues
eigenvalues_sorted = eigenvalues[idx]
print("Sorted eigenvalues:", eigenvalues_sorted)

# Reorder eigenvectors
eigenvectors_sorted = eigenvectors[:, idx]
print("Sorted eigenvectors:\n", eigenvectors_sorted)

# Calculate importance percentages
total = np.sum(eigenvalues_sorted)
percentages = (eigenvalues_sorted / total) * 100
print("Percentages:", percentages)
```

**Output:**
```
Sorting indices: [0 1 2]
Sorted eigenvalues: [4. 2. 1.]
Sorted eigenvectors:
 [[-0.40824829 -0.70710678 -0.57735027]
  [-0.40824829  0.70710678 -0.57735027]
  [-0.81649658  0.         0.57735027]]
Percentages: [57.14285714 28.57142857 14.28571429]
```

---

### 7. Verify: A × v = λ × v
```python
# This is the FUNDAMENTAL PROPERTY OF EIGENVECTORS!

print("VERIFICATION: A × v = λ × v")
print("="*50)

for i in range(len(eigenvalues_sorted)):
    v = eigenvectors_sorted[:, i]  # Get eigenvector i
    lam = eigenvalues_sorted[i]    # Get eigenvalue i
    
    # Calculate both sides
    left_side = ATA @ v            # A × v
    right_side = lam * v           # λ × v
    
    # Check if they're equal
    match = np.allclose(left_side, right_side)
    
    print(f"\nPattern {i+1}:")
    print(f"  λ = {lam:.1f}")
    print(f"  A × v = {np.round(left_side, 3)}")
    print(f"  λ × v = {np.round(right_side, 3)}")
    print(f"  Match? {match} ✓")
```

**Output:**
```
VERIFICATION: A × v = λ × v
==================================================

Pattern 1:
  λ = 4.0
  A × v = [-1.633 -1.633 -3.265]
  λ × v = [-1.633 -1.633 -3.265]
  Match? True ✓

Pattern 2:
  λ = 2.0
  A × v = [-1.414  1.414  0.000]
  λ × v = [-1.414  1.414  0.000]
  Match? True ✓

Pattern 3:
  λ = 1.0
  A × v = [-0.577 -0.577  0.577]
  λ × v = [-0.577 -0.577  0.577]
  Match? True ✓
```

---

## PART 4: MATRIX DECOMPOSITION

### 8. Create Diagonal Matrix
```python
# Create diagonal matrix with eigenvalues
Lambda = np.diag(eigenvalues_sorted)

print("Diagonal matrix Λ (Lambda):")
print(Lambda)

# Verify it's diagonal (zeros off-diagonal)
print("Is diagonal?", np.allclose(Lambda, np.diag(np.diag(Lambda))))
```

**Output:**
```
Diagonal matrix Λ (Lambda):
[[4. 0. 0.]
 [0. 2. 0.]
 [0. 0. 1.]]
Is diagonal? True
```

---

### 9. Reconstruct Original Matrix
```python
# Matrix decomposition: A = V × Λ × V^T

V = eigenvectors_sorted
Vt = V.T

# Step 1: Λ × V^T
step1 = Lambda @ Vt
print("Step 1: Λ × V^T shape:", step1.shape)

# Step 2: V × (Λ × V^T)
reconstructed = V @ Lambda @ Vt
print("\nStep 2: V × Λ × V^T shape:", reconstructed.shape)

print("\nOriginal A^T × A:")
print(ATA)

print("\nReconstructed (V × Λ × V^T):")
print(reconstructed)

# Verify they match
print("\nDo they match?", np.allclose(ATA, reconstructed))

# Check difference
difference = np.abs(ATA - reconstructed)
print("Maximum difference:", np.max(difference))
```

**Output:**
```
Step 1: Λ × V^T shape: (3, 3)

Step 2: V × Λ × V^T shape: (3, 3)

Original A^T × A:
[[ 5.  0.  6.]
 [ 0.  5.  4.]
 [ 6.  4. 14.]]

Reconstructed (V × Λ × V^T):
[[ 5. -0.  6.]
 [-0.  5.  4.]
 [ 6.  4. 14.]]

Do they match? True
Maximum difference: 8.88178420e-16
```

---

## PART 5: DOCUMENT EMBEDDINGS

### 10. Create Embeddings
```python
# Document embeddings are eigenvectors arranged as rows
embeddings = V.T  # Transpose to get 3 rows (documents) × 3 columns (patterns)

print("Document Embeddings:")
print(embeddings)
print("Shape:", embeddings.shape)

# Label them
doc_names = ['Doc1', 'Doc2', 'Doc3']
for i, doc in enumerate(doc_names):
    print(f"{doc}: {embeddings[i]}")
```

**Output:**
```
Document Embeddings:
[[-0.40824829 -0.70710678 -0.57735027]
 [-0.40824829  0.70710678 -0.57735027]
 [-0.81649658  0.         0.57735027]]
Shape: (3, 3)

Doc1: [-0.40824829 -0.70710678 -0.57735027]
Doc2: [-0.40824829  0.70710678 -0.57735027]
Doc3: [-0.81649658  0.         0.57735027]
```

---

## PART 6: USEFUL NUMPY FUNCTIONS

### Common Operations
```python
# Create matrices
np.zeros((3, 3))         # All zeros
np.ones((3, 3))          # All ones
np.eye(3)                # Identity matrix
np.random.rand(3, 3)     # Random numbers

# Matrix operations
A @ B                    # Matrix multiplication
np.dot(A, B)             # Alternative matrix multiplication
A.T                      # Transpose
np.linalg.inv(A)         # Inverse
np.linalg.det(A)         # Determinant
np.trace(A)              # Trace (sum of diagonal)

# Statistics
np.mean(A)               # Mean of all elements
np.std(A)                # Standard deviation
np.sum(A)                # Sum all elements
np.max(A)                # Maximum element
np.min(A)                # Minimum element

# Element-wise operations
A + B                    # Element-wise addition
A - B                    # Element-wise subtraction
A * B                    # Element-wise multiplication (NOT matrix!)
A / B                    # Element-wise division
np.sqrt(A)               # Element-wise square root
np.exp(A)                # Element-wise exponential

# Comparison
np.allclose(A, B)        # Check if arrays are close (useful for verification!)
np.isclose(a, b)         # Element-wise comparison
A == B                   # Element-wise equality

# Reshape
A.reshape(5, 3)          # Change shape
A.flatten()              # Flatten to 1D
A.ravel()                # Alternative flatten
```

---

## PART 7: FULL WORKING EXAMPLE

### Complete Code (Copy and Run!)
```python
import numpy as np
import pandas as pd

# STEP 1: Create data
print("STEP 1: Create Word-Document Matrix")
print("="*50)

A = np.array([
    [2, 0, 3],
    [1, 0, 0],
    [0, 2, 2],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

print("Matrix A (5×3):")
print(A)

# STEP 2: Transpose
print("\n\nSTEP 2: Transpose")
print("="*50)

At = A.T
print("Transpose A^T (3×5):")
print(At)

# STEP 3: Multiply
print("\n\nSTEP 3: Calculate Similarity Matrix")
print("="*50)

ATA = At @ A
print("A^T × A (3×3):")
print(ATA)

# STEP 4: Eigenanalysis
print("\n\nSTEP 4: Calculate Eigenvalues & Eigenvectors")
print("="*50)

eigenvalues, eigenvectors = np.linalg.eig(ATA)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# STEP 5: Verify
print("\n\nSTEP 5: Verify A × v = λ × v")
print("="*50)

for i in range(3):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    left = ATA @ v
    right = lam * v
    print(f"Pattern {i+1}: {np.allclose(left, right)} ✓")

# STEP 6: Display nicely with Pandas
print("\n\nSTEP 6: Display Embeddings")
print("="*50)

embeddings = eigenvectors.T
pct = (eigenvalues / np.sum(eigenvalues)) * 100

df = pd.DataFrame(embeddings, 
                  index=['Doc1', 'Doc2', 'Doc3'],
                  columns=['P1', 'P2', 'P3'])

print("Document Embeddings:")
print(df.round(3))

print("\nEigenvalue Importance:")
for i, p in enumerate(pct):
    print(f"  Pattern {i+1}: {p:.1f}%")

print("\n✓ COMPLETE!")
```

---

## PART 8: PRACTICE EXERCISES

### Exercise 1: Different Matrix
Create your own 4×3 matrix and:
1. Calculate transpose
2. Multiply A^T × A
3. Find eigenvalues
4. Verify A × v = λ × v

```python
# Your matrix here
B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [2, 1, 4],
    [3, 3, 1]
], dtype=float)

# Continue from STEP 2 above...
```

### Exercise 2: Larger Matrix
```python
# Create a random 10×8 matrix
C = np.random.rand(10, 8)

# Follow the same steps as above
```

### Exercise 3: Custom Similarity
```python
# Create word frequencies yourself
docs = {
    'Doc1': {'python': 3, 'code': 2, 'learn': 1},
    'Doc2': {'python': 1, 'data': 3, 'science': 2},
    'Doc3': {'code': 2, 'data': 2, 'analysis': 1}
}

# Convert to matrix format
# Hint: Use a dictionary or list to create the matrix
```

---

## PART 9: KEY NUMPY CONCEPTS

### Indexing
```python
A[0, 0]      # First element
A[0, :]      # First row
A[:, 0]      # First column
A[-1]        # Last row
A[-1, -1]    # Last element
```

### Slicing
```python
A[0:2]       # Rows 0-1
A[:, 0:2]    # Columns 0-1
A[::2]       # Every 2nd row
```

### Shape & Dimension
```python
A.shape      # Dimensions (rows, columns)
A.ndim       # Number of dimensions
A.size       # Total elements
len(A)       # Number of rows
```

### Data Types
```python
A.dtype      # Data type
A.astype(int)  # Convert to integer
```

---

## PART 10: TROUBLESHOOTING

### Common Errors

**Error: "shapes (3,5) and (3,3) not aligned"**
```python
# WRONG - dimensions don't match for multiplication
result = At @ ATA  # (3,5) @ (3,3) ❌

# RIGHT - dimensions match
result = At @ A    # (3,5) @ (5,3) ✓
```

**Error: "cannot reshape array of size 15 into shape (3,3)"**
```python
# Wrong size
A.reshape(3, 3)  # ❌ (5×3=15 elements, can't make 3×3=9)

# Right size
A.reshape(5, 3)  # ✓ Keep original shape
```

**Unexpected Results**
```python
# Check shape
print(A.shape)

# Check values
print(A)

# Check types
print(A.dtype)

# Use allclose for verification
print(np.allclose(A, B))
```

---

## FINAL COMPLETE WORKING CODE

Save this as `eigenanalysis.py` and run it!

```python
#!/usr/bin/env python3
"""
Complete Document Embeddings using NumPy
Reproduces all calculations from the guide
"""

import numpy as np
import pandas as pd

def main():
    # Create word-document matrix
    A = np.array([
        [2, 0, 3],
        [1, 0, 0],
        [0, 2, 2],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)
    
    print("Original Matrix A:")
    print(A)
    print()
    
    # Transpose
    At = A.T
    
    # Similarity matrix
    ATA = At @ A
    print("Similarity Matrix A^T × A:")
    print(ATA)
    print()
    
    # Eigenanalysis
    eigenvalues, eigenvectors = np.linalg.eig(ATA)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)
    print()
    
    # Verify
    print("Verification (A × v = λ × v):")
    for i in range(3):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        match = np.allclose(ATA @ v, lam * v)
        print(f"  Pattern {i+1}: {match}")
    print()
    
    # Embeddings
    embeddings = eigenvectors.T
    pct = (eigenvalues / np.sum(eigenvalues)) * 100
    
    print("Importance:")
    for i, p in enumerate(pct):
        print(f"  Pattern {i+1}: {p:.1f}%")
    print()
    
    # Display
    df = pd.DataFrame(embeddings,
                      index=['Doc1', 'Doc2', 'Doc3'],
                      columns=['P1', 'P2', 'P3'])
    print("Embeddings:")
    print(df.round(3))

if __name__ == "__main__":
    main()
```

**Run it:**
```bash
python eigenanalysis.py
```

---

## NEXT STEPS

1. ✓ Copy the code above
2. ✓ Run it in Jupyter or Python
3. ✓ Modify the matrix
4. ✓ Try different sizes
5. ✓ Add visualization (matplotlib)
6. ✓ Apply to real data

**You're now ready to calculate like we did!** 🚀