# flow
Flow is a Python library that provides an alternative to creating classes.

### Key Features:
- Define functions and their dependencies.
- Automatically compute values based on dependencies.
- Cache results to avoid redundant calculations.
- Visualize the computation graph with type hints and descriptions.
- Provides an inheritance mechanism

### Example Usage:

```python
from flow import flow, gui
from typing import Annotated

# Define some functions with type hints and optional descriptions
def a() -> Annotated[int, "Constant value for a"]:
    """Returns a constant integer."""
    return 2

def b() -> int:
    """Returns another constant integer."""
    return 3

def add(a: int, b: int) -> Annotated[int, "Sum of inputs"]:
    """Adds two integers."""
    return a + b

def multiply(a: int, b: int) -> Annotated[int, "Product of inputs"]:
    """Multiplies two integers."""
    return a * b

# Create a flow with these functions
# It creates a dependency graph from the argument names
f = flow(
    name='FirstFlow',
    doc='It does arithmetic.',
    a=a, b=b,
    add=add, multiply=multiply
)

# f is an object
f.a         # 2
f.b         # 3
f.add       # 5
f.multiply  # 6

# f works as a function with cached values
f(b=7).add  # 9 (does not recompute a)

# In a notebook, the object displays the dependency graph
f  # Displays dependency graph with types and values

# Display buttons to evaluate each node
gui(f)