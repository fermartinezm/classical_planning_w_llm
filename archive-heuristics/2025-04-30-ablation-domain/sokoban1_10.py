# Import necessary modules
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions (adapted from provided examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Use fnmatch for flexible pattern matching on each corresponding part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs_distance(start, end, graph):
    """
    Computes the shortest path distance between two locations in the graph.
    Returns float('inf') if no path exists.
    """
    if start == end:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()

        # Ensure current_loc is a valid key in the graph (should be if built correctly)
        # If a location from the state/goals isn't in the graph built from statics,
        # it implies an invalid problem definition or state. Treat as unreachable.
        if current_loc not in graph:
             return float('inf')

        for neighbor in graph[current_loc]:
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor == end:
                    return dist + 1
                queue.append((neighbor, dist + 1))
    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its goal location.
    This represents a lower bound on the number of pushes required if the robot
    could instantly move to the correct pushing position for each box step.

    # Assumptions
    - Each box has a unique goal location specified in the task goals.
    - The traversable grid structure and connectivity are defined by the 'adjacent' predicates.
    - The cost of moving a box one step towards its goal is estimated by the
      shortest path distance on the location graph defined by 'adjacent' facts.
    - The robot's position and the need to maneuver the robot are ignored in the distance calculation for boxes.
    - Deadlocks (boxes pushed into unrecoverable positions) are detected if a box cannot reach its goal location via the graph, resulting in an infinite heuristic value.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an undirected adjacency graph of locations based on the 'adjacent' static facts. This graph represents the traversable paths for both the robot and boxes (when pushed).

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the current location of each box that has a corresponding goal location.
    3. Initialize the total heuristic value to 0.
    4. For each box that is not currently at its goal location:
       - Retrieve the box's current location and its target goal location.
       - Calculate the shortest path distance between the box's current location
         and its goal location using a Breadth-First Search (BFS) on the pre-built
         location graph.
       - If no path exists (distance is infinite), the state is likely a dead end
         or unsolvable from here; return infinity immediately.
       - Add the calculated distance to the total heuristic value.
    5. Return the accumulated total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency graph from static facts
        self.graph = {}
        # Collect all locations mentioned in adjacent facts to ensure they are nodes
        all_locations = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "adjacent" and len(parts) == 4: # Ensure it's a valid adjacent fact
                  loc1, loc2, direction = parts[1:]
                  all_locations.add(loc1)
                  all_locations.add(loc2)

        # Initialize graph with all locations as keys
        for loc in all_locations:
             self.graph[loc] = []

        # Add bidirectional edges based on adjacent facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4: # Ensure it's a valid adjacent fact
                loc1, loc2, direction = parts[1:]
                # Add edge loc1 -> loc2
                if loc2 not in self.graph.get(loc1, []): # Use .get for safety, though keys should exist
                    self.graph[loc1].append(loc2)
                # Add edge loc2 -> loc1 (graph is undirected for distance calculation)
                if loc1 not in self.graph.get(loc2, []):
                    self.graph[loc2].append(loc1)

        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goal predicates are only (at boxX locY) for boxes
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                 box, location = args
                 self.box_goals[box] = location
            # Ignore other potential goal predicates if any exist and are not relevant to boxes

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        # Use the task's goal_reached method for robustness
        # This requires access to the task object, which is available via node.task
        # However, the Heuristic base class doesn't seem to provide node.task.
        # The simple check `self.goals <= state` is correct if goals are just a conjunction of facts.
        # Given the examples, this assumption is safe.
        if self.goals <= state:
             return 0

        # 2. Find current box locations
        current_box_locations = {}
        # Iterate through state facts to find box locations
        for fact in state:
            parts = get_parts(fact)
            # Check for facts like (at box1 loc_X_Y)
            if parts[0] == "at" and len(parts) == 2 and parts[1].startswith("box"):
                 box, location = parts[1], parts[2] # Correctly extract box and location
                 current_box_locations[box] = location

        # 3. Initialize total heuristic
        total_cost = 0

        # 4. Calculate sum of distances for misplaced boxes
        # Iterate through the boxes we know have goals
        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box)

            # If a box from the goals is not found in the current state,
            # something is wrong, or it implies the box was removed (not possible in Sokoban).
            # Assuming valid states contain all boxes from the initial state.
            if current_location is None:
                 # This shouldn't happen in a standard Sokoban state graph
                 # where objects don't disappear. Treat as unreachable goal.
                 return float('inf')

            if current_location != goal_location:
                # Calculate shortest path distance from current box location to goal location
                dist = bfs_distance(current_location, goal_location, self.graph)

                # 5. Handle unreachable goals
                if dist == float('inf'):
                    # If a box cannot reach its goal location in the graph,
                    # this state is a dead end. Assign infinite heuristic.
                    return float('inf')

                total_cost += dist

        # 5. Return the accumulated total heuristic value.
        # If total_cost is 0 here, it means all boxes iterated over were at their goals.
        # This implies all boxes with goals are at their targets.
        # The initial check `self.goals <= state` handles the case where
        # there might be other goal conditions (like robot position),
        # but for standard Sokoban, box positions are the only goals.
        # So, if total_cost is 0 here, the initial check should have returned 0.
        # The logic is consistent.
        return total_cost
