# GloVe Word Embeddings: Complete Q&A Guide

## Part 1: Understanding Co-occurrence Matrices

### Q1: What is a co-occurrence matrix?

A co-occurrence matrix is a table that tracks which words appear near each other in text. Both rows and columns represent words from your vocabulary, and each cell contains a count of how many times those two words appeared together within a context window (typically 5 words apart).

**Example Setup:**
Vocabulary: {dog, bark, cat, meow, tree}

|       | dog | bark | cat | meow | tree |
|-------|-----|------|-----|------|------|
| dog   | 0   | 500  | 45  | 3    | 12   |
| bark  | 500 | 0    | 8   | 2    | 1    |
| cat   | 45  | 8    | 0   | 480  | 20   |
| meow  | 3   | 2    | 480 | 0    | 5    |
| tree  | 12  | 1    | 20  | 5    | 0    |

The value 500 at (dog, bark) means: "In the corpus, 'dog' and 'bark' appear within a context window 500 times."

### Q2: Why do we need a co-occurrence matrix?

Because of the principle: "You shall know a word by the company it keeps." Words that frequently appear together are likely to be semantically related. The co-occurrence matrix captures these statistical relationships across the entire text corpus.

---

## Part 2: The Problem with Large Matrices

### Q3: Why is a 400,000 × 400,000 co-occurrence matrix a problem?

A vocabulary of 400,000 words creates a matrix with **160 billion cells**. This is computationally expensive to store and process. Most cells are zeros or very small numbers (sparse data), making this representation wasteful.

---

## Part 3: Matrix Factorization Solution

### Q4: How do we solve the size problem?

We use **matrix factorization**—specifically Singular Value Decomposition (SVD). The algorithm breaks down the massive 400,000 × 400,000 matrix into smaller, manageable matrices that can be multiplied back together to approximate the original.

### Q5: How exactly does matrix factorization work?

The original co-occurrence matrix C is decomposed as:

\[C = U \Sigma V^T\]

Where:
- U is a 400,000 × r matrix
- Σ is an r × r diagonal matrix of singular values
- V^T is an r × 400,000 matrix

The key insight: most information concentrates in just a few singular values. We keep only the **k largest singular values** (e.g., k = 300) and set the rest to zero.

### Q6: What do we end up with after factorization?

After truncation to keep only 300 dimensions:
- U becomes 400,000 × 300
- Σ becomes 300 × 300
- V^T becomes 300 × 400,000

We've compressed **160 billion cells down to approximately 240 million cells total**.

### Q7: Why keep 400,000 in the dimension?

The 400,000 represents your **vocabulary size**—the number of unique words. Each word needs its own representation. The 300 represents the **embedding dimensions**—the compressed features that capture each word's semantic meaning.

**Result:** Each of the 400,000 words gets a 300-dimensional word embedding vector.

---

## Part 4: Pointwise Mutual Information (PMI) - The Training Target

**IMPORTANT:** Before we can train embeddings, we must first calculate the PMI values from the co-occurrence matrix. These PMI values become the targets that the embeddings are trained to match.

### Q8: What is PMI and why do we need it?

PMI (Pointwise Mutual Information) measures whether two words co-occur more or less often than random chance would predict. This is more meaningful than raw co-occurrence counts because it accounts for individual word frequencies.

\[\text{PMI}(\text{dog, bark}) = \log\left(\frac{P(\text{dog, bark})}{P(\text{dog}) \times P(\text{bark})}\right)\]

### Q9: Why not just use P(dog) × P(bark)?

Great question! Let me explain why we need the full PMI formula instead of just multiplying the marginal probabilities.

**The Problem with Just P(dog) × P(bark):**

If we only used P(dog) × P(bark), we'd be calculating the probability that dog and bark co-occur **if they were completely independent** (if their appearance had nothing to do with each other).

**Example to show why this matters:**

Imagine two scenarios:

**Scenario 1: "dog" and "bark" are semantically related**
- They naturally appear together because they're related concepts
- Actual co-occurrence: 500 times
- If they were random/independent: would only appear together ~146 times (based on P(dog) × P(bark))
- Result: They appear together **much more than chance** predicts

**Scenario 2: "dog" and "the" are not semantically related**
- "the" is a common word that appears everywhere
- Actual co-occurrence: 5,000 times (very high count!)
- If they were independent: would appear together ~2,800 times
- Result: They appear together roughly as much as chance would predict

**The key insight:** Raw co-occurrence counts are misleading. "dog" and "the" co-occur more frequently (5,000 vs 500), but "dog" and "bark" are more semantically related. We need PMI to distinguish this difference.

PMI fixes this by comparing:
- **Actual:** How often they really co-occur
- **Expected by chance:** P(dog) × P(bark) 

If actual > expected, PMI is positive (they're attracted to each other semantically).
If actual = expected, PMI is zero (they're independent).
If actual < expected, PMI is negative (they avoid each other).

---

## Part 5: Calculate PMI Step-by-Step

### Q10: How do we calculate PMI step-by-step?

**Step 1: Calculate total co-occurrences**

Sum all values in the co-occurrence matrix (excluding diagonal):
Total = 500 + 45 + 3 + 12 + 500 + 8 + 2 + 1 + 45 + 8 + 480 + 20 + 3 + 2 + 480 + 5 + 12 + 1 + 20 + 5 = 2,000

**Step 2: Calculate the joint probability P(dog, bark)**

\[P(\text{dog, bark}) = \frac{\text{co-occurrence count}}{\text{total co-occurrences}} = \frac{500}{2000} = 0.25\]

This is the probability that any randomly selected co-occurrence involves both "dog" and "bark."

**What this means:** Out of every 2,000 word pair co-occurrences we observe, 500 of them are "dog" paired with "bark." So there's a 25% chance any randomly selected co-occurrence is this pair.

**Step 3: Calculate marginal probability for dog**

Dog appears in co-occurrences with: bark (500) + cat (45) + meow (3) + tree (12) = 560 times

\[P(\text{dog}) = \frac{560}{2000} = 0.28\]

This is the probability that any co-occurrence involves "dog" (regardless of what it co-occurs with).

**What this means:** Out of every 2,000 co-occurrences, 560 of them involve the word "dog" (paired with anything). So there's a 28% chance any randomly selected co-occurrence includes "dog."

**Step 4: Calculate marginal probability for bark**

Bark appears in co-occurrences with: dog (500) + cat (8) + meow (2) + tree (1) = 511 times

\[P(\text{bark}) = \frac{511}{2000} = 0.26\]

This is the probability that any co-occurrence involves "bark" (regardless of what it co-occurs with).

**What this means:** Out of every 2,000 co-occurrences, 511 of them involve the word "bark" (paired with anything). So there's a 26% chance any randomly selected co-occurrence includes "bark."

**Step 5: Calculate the probability if they were independent**

If "dog" and "bark" appeared together purely by random chance (completely independent), the probability would be:

\[P(\text{dog}) \times P(\text{bark}) = 0.28 \times 0.26 = 0.073\]

**What this means:** If dog and bark had nothing to do with each other—if they just randomly happened to appear in the same context windows—they'd co-occur with probability 0.073. Out of 2,000 co-occurrences, we'd expect about 146 of them to be "dog" with "bark" (0.073 × 2,000 = 146).

**This is the crucial comparison:**
- **Actual co-occurrence:** 500 times (P(dog, bark) = 0.25)
- **Expected by chance:** 146 times (P(dog) × P(bark) = 0.073)

Dog and bark co-occur 500 ÷ 146 = 3.42 times MORE OFTEN than we'd expect if they were independent!

**Step 6: Calculate the ratio - Why do we divide?**

Now we form the ratio to see how much MORE frequently they co-occur than chance predicts:

\[\frac{P(\text{dog, bark})}{P(\text{dog}) \times P(\text{bark})} = \frac{0.25}{0.073} = 3.42\]

**This ratio tells us:**
- If ratio > 1: They co-occur MORE often than chance predicts (they're attracted to each other)
- If ratio = 1: They co-occur EXACTLY as often as chance would predict (they're independent)
- If ratio < 1: They co-occur LESS often than chance predicts (they avoid each other)

In our case, the ratio is 3.42, meaning dog and bark co-occur **3.42 times more often than random chance would predict**.

**Why divide instead of subtract?**

You might ask: "Why not just calculate P(dog, bark) - P(dog) × P(bark) = 0.25 - 0.073 = 0.177?"

Answer: Because subtraction doesn't scale well with different probability ranges.

**Example:**
- Scenario A: Actual = 0.3, Expected = 0.1 → Difference = 0.2 → Ratio = 3.0
- Scenario B: Actual = 0.003, Expected = 0.001 → Difference = 0.002 → Ratio = 3.0

Both scenarios have the same ratio (3.0 times more frequent than expected), but different differences (0.2 vs 0.002). Division preserves the relative relationship regardless of absolute probability values. Subtraction would incorrectly suggest Scenario A is more significant.

**Step 7: Apply logarithm to get PMI**

\[\text{PMI}(\text{dog, bark}) = \log(3.42) \approx 1.23\]

We apply logarithm because:
1. It compresses large ratio values (prevents extreme numbers)
2. It makes ratios comparable on a symmetric scale (log(3) and log(1/3) have equal magnitude but opposite signs)
3. It's mathematically convenient for optimization algorithms

**What the 1.23 value means:**
- It's the log of how much more often they co-occur than chance predicts
- Positive 1.23 means they're strongly semantically related
- The magnitude tells us the strength of the relationship

### Q11: What does this PMI value tell us?

- **PMI > 0:** The words co-occur more often than random chance predicts. They're semantically related and attracted to each other.
- **PMI = 0:** They co-occur exactly as random chance would predict. No meaningful relationship.
- **PMI < 0:** They co-occur less than random chance predicts. They tend to avoid each other.

For our example, PMI ≈ 1.23 is positive and relatively large, meaning "dog" and "bark" are genuinely semantically related—not just appearing together by coincidence.

**This 1.23 value is now our training target for the embeddings.**

---

## Part 6: Training Word Embeddings to Match PMI

### Q12: What is GloVe's training objective?

GloVe trains word embeddings so that the dot product between any two word vectors approximates their PMI value:

\[\text{dot product}(\text{embedding}_{\text{dog}}, \text{embedding}_{\text{bark}}) \approx \text{PMI}(\text{dog, bark}) = 1.23\]

### Q13: How does the training process work?

**Step 1: Initialize random embeddings**

Start with random 300-dimensional vectors for each word:
- embedding_dog = [0.1, 0.2, 0.3, 0.4, 0.5, ..., 0.15] (300 random numbers)
- embedding_bark = [0.1, 0.2, 0.3, 0.4, 0.5, ..., 0.20] (300 random numbers)

**Step 2: Calculate current dot product**

\[\text{dot product} = (0.1 \times 0.1) + (0.2 \times 0.2) + (0.3 \times 0.3) + ... + (0.15 \times 0.20)\]

Let's say this equals 0.55.

**Step 3: Compare to target PMI**

- Target: 1.23
- Current: 0.55
- Error: 1.23 - 0.55 = 0.68 (too low)

**Step 4: Adjust the embeddings**

The optimization algorithm (like gradient descent) adjusts the numbers in both embeddings to reduce the error. It might change:
- embedding_dog to [0.3, 0.15, 0.5, 0.2, 0.4, ..., 0.25]
- embedding_bark to [0.4, 0.2, 0.6, 0.25, 0.5, ..., 0.30]

**Step 5: Recalculate dot product**

\[\text{dot product} = (0.3 \times 0.4) + (0.15 \times 0.2) + (0.5 \times 0.6) + ... + (0.25 \times 0.30)\]

Now equals 0.70. Still not 1.23, so continue adjusting.

**Step 6: Repeat iteratively**

The algorithm continues adjusting the embedding values thousands of times. After many iterations, it might reach:
- embedding_dog = [0.48, -0.25, 0.70, 0.14, 0.58, ..., -0.34]
- embedding_bark = [0.55, -0.22, 0.74, 0.10, 0.52, ..., -0.29]

\[\text{dot product} = (0.48 \times 0.55) + (-0.25 \times -0.22) + (0.70 \times 0.74) + ... + (-0.34 \times -0.29)\]
\[= 0.264 + 0.055 + 0.518 + ... + 0.099\]
\[= 1.23\]

**Success!** The dot product now matches the target PMI.

### Q14: What do these embedding numbers represent?

The numbers in the embeddings (like 0.48, -0.25, 0.70, etc.) are **learned parameters discovered through optimization**. They are:

- **NOT** co-occurrence counts from the matrix
- **NOT** probabilities
- **NOT** logarithms of anything
- **NOT** mathematically derived from a formula

They are simply **values that the algorithm discovered through trial and error** to satisfy the constraint that their dot product equals the target PMI.

Think of it like solving a puzzle:
- **The puzzle:** Find two sets of 300 numbers that multiply together (dot product) to give 1.23
- **The optimizer:** Tries different number combinations repeatedly and measures the error
- **The solution:** After many iterations, finds one valid combination

There's nothing inherently special about the specific values [0.48, -0.25, 0.70, ...]. A different optimization run might produce different numbers. **As long as the dot product equals 1.23, any set of numbers works.**

### Q15: Does GloVe train embeddings for just one word pair?

No! GloVe simultaneously trains embeddings to satisfy the PMI constraint for **all word pairs** in the vocabulary.

For our 5-word vocabulary, it needs to satisfy:
- dot product(dog, bark) ≈ 1.23
- dot product(dog, cat) ≈ PMI(dog, cat) 
- dot product(dog, meow) ≈ PMI(dog, meow)
- dot product(dog, tree) ≈ PMI(dog, tree)
- dot product(bark, cat) ≈ PMI(bark, cat)
- ...and all other pairs

The algorithm adjusts all embeddings simultaneously to minimize the total error across all word pairs.

---

## Part 7: Using the Trained Embeddings

### Q16: How do we approximate co-occurrence relationships from trained embeddings?

After training completes, we can use the **dot product** between two word embedding vectors to approximate their **PMI value**:

\[\text{dot product}(\text{embedding}_{\text{dog}}, \text{embedding}_{\text{bark}}) \approx \text{PMI}(\text{dog, bark}) = 1.23\]

**With the trained embeddings:**
- embedding_dog = [0.48, -0.25, 0.70, 0.14, 0.58, ..., -0.34] (300 numbers)
- embedding_bark = [0.55, -0.22, 0.74, 0.10, 0.52, ..., -0.29] (300 numbers)

**The dot product calculation:**

\[\text{dot product} = (0.48 \times 0.55) + (-0.25 \times -0.22) + (0.70 \times 0.74) + (0.14 \times 0.10) + (0.58 \times 0.52) + ... + (-0.34 \times -0.29)\]

\[= 0.264 + 0.055 + 0.518 + 0.014 + 0.302 + ... + 0.099\]

\[= 1.23\]

This 1.23 tells us that "dog" and "bark" are strongly semantically related (positive PMI = they co-occur more than chance).

### Q17: Can we get back the original co-occurrence count?

Yes, but it requires reversing the PMI calculation:

**Step 1: Exponentiate the dot product (PMI)**

\[e^{1.23} = 3.42\]

This gives us the ratio \(\frac{P(\text{dog, bark})}{P(\text{dog}) \times P(\text{bark})}\)

**Step 2: Multiply by marginal probabilities**

\[3.42 \times 0.28 \times 0.26 = 0.249 \approx 0.25\]

This gives us P(dog, bark).

**Step 3: Multiply by total co-occurrences**

\[0.25 \times 2000 = 500\]

We've recovered the original co-occurrence count from the co-occurrence matrix!

**However, in practice, you rarely need to do this.** For most NLP tasks, you only need the PMI (semantic similarity), which the dot product gives you directly.

---

## Part 8: Complete Summary

### The GloVe Process, Step by Step

**Phase 1: Build the co-occurrence matrix**
Count how often words appear together in the corpus.
- Result: 400,000 × 400,000 matrix with 160 billion cells
- Example: dog and bark co-occur 500 times

**Phase 2: Calculate PMI values for all word pairs**
For each word pair, calculate how much more frequently they co-occur than chance would predict.
- Calculate P(dog, bark), P(dog), P(bark)
- Calculate ratio: P(dog, bark) / (P(dog) × P(bark)) = 0.25 / 0.073 = 3.42
- Apply log: log(3.42) = 1.23
- Result: PMI targets for all word pairs
- Example: PMI(dog, bark) = 1.23

**Phase 3: Initialize random embeddings**
Start with random 300-dimensional vectors for each word.
- Result: 400,000 × 300 word embedding matrix with random values

**Phase 4: Train embeddings via optimization**
Iteratively adjust embedding values so that dot products match PMI targets.
- Optimization method: Gradient descent or alternating least squares
- Objective: Minimize error between dot products and PMI values across all word pairs
- Result: Trained embeddings where dot product(dog, bark) ≈ 1.23

**Phase 5: Use embeddings**
After training, compute semantic relationships using simple dot products.
- Compression: 300 dimensions instead of 400,000
- Storage: 240 million cells instead of 160 billion cells
- Query: Simple dot product instead of matrix lookup

### Why This Works

- **Compression:** 300 dimensions capture the essential semantic information that would require 400,000 dimensions in the original matrix.
- **Efficiency:** Computing similarity is a simple dot product, not a matrix lookup.
- **Semantic meaning:** The dot product reflects meaningful semantic relationships captured by PMI, not just raw frequency.
- **Learned representations:** The embedding values are discovered through optimization to satisfy PMI constraints across all word pairs simultaneously.
- **Generalization:** Embeddings work for tasks beyond just co-occurrence because they encode deep semantic structure in the learned parameters.

### Key Insights

1. **PMI is the bridge:** PMI captures meaningful relationships by comparing actual co-occurrence to what random chance would predict. This is why we don't just use P(dog) × P(bark).

2. **Embedding values are discovered, not calculated:** The numbers in embeddings (0.48, -0.25, etc.) don't come from a formula. They're found through optimization to satisfy the constraint that their dot product equals the target PMI.

3. **Compression is powerful:** Despite having only 300 numbers per word instead of 400,000, embeddings capture semantic relationships because PMI already encodes meaningful information.

4. **Multiple valid solutions:** Different training runs produce different embedding values, but all valid solutions preserve the semantic relationships encoded in the PMI targets.