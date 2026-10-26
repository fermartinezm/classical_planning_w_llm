from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# BFS function to find shortest path distances in the location graph
def bfs(start_loc, graph):
    """
    Performs Breadth-First Search starting from start_loc to find shortest distances
    to all reachable locations in the graph.

    Args:
        start_loc (str): The starting location string.
        graph (dict): Adjacency list representation of the location graph.

    Returns:
        dict: A dictionary mapping reachable location strings to their shortest distances from start_loc.
              Returns None for locations that are unreachable.
    """
    distances = {start_loc: 0}
    queue = deque([start_loc])
    visited = {start_loc} # Keep track of visited nodes to avoid cycles and redundant processing

    while queue:
        current_loc = queue.popleft()
        current_dist = distances[current_loc]

        # Check if current_loc exists in the graph keys before iterating neighbors
        # This check is technically redundant if the graph is built correctly
        # from adjacent facts where all mentioned locations are keys or values.
        # But it's harmless.
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It primarily focuses on the distance each box
    needs to travel, multiplied by a factor to account for the robot's
    necessary repositioning effort per push.

    # Assumptions
    - The grid layout and adjacency are defined by the 'adjacent' static facts.
    - Each box has a unique goal location specified in the task goals.
    - The cost of moving the robot or pushing a box is 1 action.
    - Moving a box one step towards its goal typically requires one 'push' action
      and additional robot 'move' actions to get into the correct pushing position
      for the next step. A factor of 3 is used as a rough estimate for the total
      actions per box-grid-step (1 push + ~2 robot moves).
    - Deadlocks (boxes pushed into unrecoverable positions) are not explicitly detected.
      If a box's goal is unreachable from its current location, the heuristic returns infinity.

    # Heuristic Initialization
    - Build an undirected adjacency graph of locations based on 'adjacent' facts.
      The graph nodes are location strings.
    - Store the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of each box from the state.
    2. Initialize the total heuristic value to 0.
    3. For each box specified in the task goals:
       a. Get the box's current location from the state and its goal location from the pre-calculated goals.
       b. If the box is already at its goal location, it contributes 0 to the heuristic for this box.
       c. If the box is not at its goal location:
          i. Calculate the shortest path distance from the box's current location to its goal location using BFS on the location graph. This distance represents the minimum number of 'push' actions needed for this box if there were no obstacles and the robot was always in position.
          ii. If the goal location is unreachable from the box's current location (based on the graph connectivity), the problem is likely unsolvable from this state, and the heuristic returns `float('inf')`.
          iii. Multiply the calculated distance by a factor (e.g., 3) to account for the robot's movement required to reposition itself for each subsequent push. Add this value to the total heuristic.
    4. The final total heuristic value is the sum of the weighted distances for all boxes not at their goals. This value is 0 if and only if all boxes are at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts ('adjacent' relationships) to build the location graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the adjacency graph from 'adjacent' facts.
        # The graph maps location strings to lists of adjacent location strings.
        self.graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph.setdefault(loc1, []).append(loc2)
                # Assuming adjacency is symmetric in Sokoban grid
                self.graph.setdefault(loc2, []).append(loc1)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at boxX loc_Y_Z)
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Find current locations of boxes.
        box_locations = {}
        # Robot location is not used in this specific heuristic calculation.
        # robot_loc = None
        for fact in state:
            # if match(fact, "at-robot", "*"):
            #     robot_loc = get_parts(fact)[1]
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        total_heuristic = 0

        # Iterate through all boxes that have a goal location defined.
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # If a box is not found in the state's 'at' facts, it might indicate
            # an unexpected state representation or an unsolvable state.
            # Based on the domain, boxes are always 'at' a location.
            if current_loc is None:
                 # This case should ideally not happen in a valid Sokoban state
                 # generated by the domain's operators.
                 # If it does, it might indicate an issue or an unsolvable state.
                 # Returning infinity is a safe bet for unsolvable.
                 return float('inf')


            if current_loc != goal_loc:
                # Calculate shortest path distance from current box location to its goal.
                # BFS from the box's current location.
                distances = bfs(current_loc, self.graph)

                dist_to_goal = distances.get(goal_loc)

                if dist_to_goal is None:
                    # Goal location is unreachable from the box's current location.
                    # This state is likely a deadlock or unsolvable.
                    return float('inf')

                # Add weighted distance for this box.
                # Using a factor of 3 (1 push + ~2 robot moves to reposition).
                total_heuristic += 3 * dist_to_goal

        # If total_heuristic is 0, it means all boxes are at their goal locations,
        # which is the goal state. The heuristic should be 0 only at the goal.
        # Our calculation satisfies this.

        return total_heuristic
