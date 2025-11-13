**Prompt:** 
Write a Python function to calculate the factorial of a number. No explanation, provide only the code.


**Response:**
```python
def factorial(n):
  """Calculates the factorial of a non-negative integer n."""
  if n == 0:
    return 1
  else:
    return n * factorial(n-1)
```


---

