# Required imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty before processing
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_4_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_distance(start_loc, end_loc, graph):
    """
    Calculates the shortest path distance between two locations using BFS.
    Returns float('inf') if the end location is unreachable.
    """
    if start_loc == end_loc:
        return 0

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # The graph stores adjacent locations regardless of direction
        # We need to iterate through all adjacent locations
        # Use .get() with a default empty set in case a location has no adjacencies listed
        for neighbor in graph.get(current_loc, set()):
            if neighbor == end_loc:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # End location is unreachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the shortest path distances between each misplaced box and its
    assigned goal location. It ignores the robot's position and the need for
    the robot to maneuver behind boxes.

    # Assumptions
    - The primary cost is moving boxes.
    - The cost of moving a box one step is proportional to the distance it needs
      to travel.
    - The grid structure and connectivity are defined by the 'adjacent' facts.
    - Each box specified in the goal has a unique goal location.
    - Location names are strings.
    - Box names are strings starting with "box".

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an adjacency graph of all locations based on the 'adjacent' static facts.
      This graph is used to compute shortest path distances between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box specified in the goal conditions:
       a. Identify its current location in the state by finding the fact `(at box_name location)`.
       b. Retrieve its goal location (stored during initialization).
       c. If the box is found in the state and is not at its goal location:
          i. Calculate the shortest path distance between the box's current
             location and its goal location using BFS on the pre-computed
             location graph.
          ii. If the goal is reachable, add this distance to the total heuristic value.
          iii. If the goal is unreachable from the box's current location, the state
               is likely a dead end or unsolvable. Return a very large value
               (infinity) to penalize this state heavily.
    2. The total heuristic value is the sum of these distances for all
       misplaced boxes.
    3. If all boxes are at their goal locations, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the location graph.
        """
        # Assuming task.goals is a frozenset of goal facts
        self.goals = task.goals
        # Assuming task.static is a frozenset of static facts
        static_facts = task.static

        # Store goal locations for each box.
        # Example: {'box1': 'loc_2_4', 'box2': 'loc_11_11', ...}
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at box_name location)
            if match(goal, "at", "*", "*"):
                 _, box, location = get_parts(goal)
                 self.goal_locations[box] = location

        # Build the adjacency graph from static facts.
        # graph[loc1] = {loc2, loc3, ...} where loc2, loc3 are adjacent to loc1
        # This creates an undirected graph where edges represent adjacency.
        self.location_graph = {}
        for fact in static_facts:
            # Adjacent facts are (adjacent loc1 loc2 dir)
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Adjacency is symmetric

        # Note: Robot location is not needed for this heuristic calculation,
        # only box locations and goal locations.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # Find current location of all boxes that are in our goals.
        # Example: {'box1': 'loc_4_4', 'box2': 'loc_4_10', ...}
        current_box_locations = {}
        # Robot location is not needed for this heuristic
        # robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty or invalid facts
                continue

            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj_name = parts[1]
                location = parts[2]
                # Check if the object is one of the boxes we care about (i.e., in our goals)
                if obj_name in self.goal_locations:
                     current_box_locations[obj_name] = location
            # We don't need robot location for this heuristic
            # elif predicate == "at-robot" and len(parts) == 2:
            #      robot_location = parts[1]


        total_heuristic = 0

        # For each box that has a goal location defined in the task
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If the box is found in the current state and is not at its goal location
            if current_location and current_location != goal_location:
                # Calculate shortest distance from current box location to its goal location
                dist = get_distance(current_location, goal_location, self.location_graph)

                # If the goal is unreachable from the box's current location,
                # this state is likely unsolvable or a deadlock for this box.
                # Return a very large value to strongly penalize this state.
                if dist == float('inf'):
                    return float('inf') # Indicate an effectively infinite cost

                # Otherwise, add the distance (number of required box pushes) to the total.
                total_heuristic += dist

            # If the box is not found in the current state but is in the goals,
            # this might indicate an invalid state representation or an unsolvable problem.
            # Treat this as an unreachable goal.
            elif current_location is None:
                 return float('inf')


        # If the loop completes without returning infinity, it means all boxes
        # in the goal are either at their goal or reachable.
        # If total_heuristic is 0, all boxes were at their goals (goal state).
        # Otherwise, it's the sum of distances for misplaced but reachable boxes.
        return total_heuristic
