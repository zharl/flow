# flow
Flow is a python library that provides an alternative to creating classes. 

### Key Features:
- Define functions and their dependencies.
- Automatically compute values based on dependencies.
- Cache results to avoid redundant calculations.
- Visualize the computation graph.
- Provides an inheritance mechanism


### Example Usage:

```python
from flow import flow, gui

# Define some functions
def a():
    return 2

def b():
    return 3

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

# Create a flow with these functions
# it creates a dependency graph from the name of the arguments
f = flow(
    a=a, b=b
    add=add, multiply=multiply
)



# f is an object
f.a # 2
f.b # 3
f.add # 5
f.multiply # 6

# f works as a function with cached values
f(b=7).add # 9 does not recompute a

# in a notebook the object will display the dependency graph
f # displays dependency graph

# display buttons that evaluate each node
gui(f)
```
