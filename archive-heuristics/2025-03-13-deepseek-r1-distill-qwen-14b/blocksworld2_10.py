To solve this problem, we need to create a domain-dependent heuristic function for the blocksworld domain. The heuristic function will estimate the number of actions required to reach the goal state as accurately as possible while being efficiently computable. The heuristic does not need to be admissible, so it can overestimate the number of actions.

### Approach
The heuristic function for the blocksworld domain is designed to estimate the number of actions needed to stack blocks according to the goal configuration. Each block that is not in its correct position contributes two actions: one to pick it up and one to place it.

1. **Extract Goal Structure**: Parse the goal to determine the correct parent for each block. For example, if the goal is `(on b1 b2)`, then `b1` should be on top of `b2`.
2. **Current Sta