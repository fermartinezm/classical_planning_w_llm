from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

# Define helper functions (can be outside the class or inside if preferred)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at obj loc)" -> ["at", "obj", "loc"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts is at least the number of pattern arguments
    # and if each part matches the corresponding pattern argument using fnmatch.
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    # Assumes location strings are always in the format 'loc_R_C'
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # This indicates an unexpected format for the row/column numbers.
            # print(f"Warning: Could not parse row/column from location string '{loc_str}'")
            return None # Indicate parsing failure
    else:
        # This indicates the string does not start with 'loc_' or doesn't have 3 parts.
        # print(f"Warning: Unexpected location string format '{loc_str}'")
        return None # Indicate parsing failure


# The heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    for each box from its current location to its goal location. The distances
    are precomputed on the static grid graph, ignoring dynamic obstacles
    (other boxes, robot) and the robot's position. This estimates the minimum
    number of pushes required in a relaxed version of the problem.

    # Assumptions
    - The grid structure is defined by 'adjacent' facts, which are assumed
      to be symmetric (if A is adjacent to B, B is adjacent to A).
    - Locations are named using the format 'loc_R_C' where R and C are integers.
    - Each box specified in the goal has a unique target location.
    - The shortest path distance on the empty grid provides a useful,
      non-admissible estimate of the effort to move a box.
    - States where a box's goal is unreachable on the static grid are considered
      dead ends or unsolvable from that point.

    # Heuristic Initialization
    - Parses 'adjacent' facts from `task.static` to build an undirected graph
      representing the grid connectivity. Locations are stored as (R, C) tuples.
    - Computes all-pairs shortest paths (APSP) on this grid graph using BFS
      starting from every node. The results are stored in `self.distances_map`.
    - Parses goal facts from `task.goals` to create a mapping from each box
      object name to its target location string.

    # Step-By-Step Thinking for Computing Heuristic
    1. Retrieve the current state from the search node.
    2. Identify the current location string for each box object present in the state
       by examining facts like `(at box_name loc_name)`.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each box and its corresponding goal location string
       as determined during initialization (from `self.goal_locations`).
    5. For a given box, get its current location string from the state.
    6. If the box's current location is different from its goal location:
       a. Convert both the current location string and the goal location string
          into (R, C) tuple representations using `parse_location`.
       b. Check if parsing was successful. If not, return `float('inf')`.
       c. Look up the precomputed shortest path distance between the current
          location tuple and the goal location tuple in `self.distances_map`.
       d. If either the current location tuple or the goal location tuple is not
          a valid node in the graph (i.e., not a key in `self.distances_map`),
          or if the distance is `float('inf')` (meaning unreachable on the static
          grid), the state is likely a dead end. Return `float('inf')` as the
          heuristic value.
       e. Otherwise, add the retrieved shortest distance to the total heuristic cost.
    7. Return the accumulated total heuristic cost. If the state is a goal state
       (all boxes are at their goals), the total cost will be 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the graph and computing APSP."""
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph from adjacent facts
        self.graph = {}
        # Collect all unique location strings first to ensure all potential nodes are considered
        all_location_strings = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "adjacent" and len(parts) == 4:
                 all_location_strings.add(parts[1])
                 all_location_strings.add(parts[2])

        # Initialize graph with all potential nodes
        for loc_str in all_location_strings:
             loc_tuple = parse_location(loc_str)
             if loc_tuple is not None:
                 self.graph[loc_tuple] = []

        # Add edges based on adjacent facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1_str, loc2_str, direction = parts[1], parts[2], parts[3]
                loc1_tuple = parse_location(loc1_str)
                loc2_tuple = parse_location(loc2_str)

                # Ensure both locations were parsed successfully and are in our graph nodes
                if loc1_tuple is not None and loc2_tuple is not None and loc1_tuple in self.graph and loc2_tuple in self.graph:
                    # Add edges in both directions as adjacency is assumed symmetric
                    self.graph[loc1_tuple].append(loc2_tuple)
                    self.graph[loc2_tuple].append(loc1_tuple)
                # else: print(f"Warning: Skipping adjacent fact with invalid locations: {fact}")


        # Compute All-Pairs Shortest Paths (APSP) using BFS from each node
        self.distances_map = {}
        # Ensure we only run BFS from nodes actually present in the graph (those with edges)
        graph_nodes = list(self.graph.keys())
        for start_node in graph_nodes:
             self.distances_map[start_node] = self._bfs(start_node)

        # Store goal locations for each box
        self.goal_locations = {} # Maps box_name -> location_string
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Check for '(at box_name loc_name)' goal facts
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args[0], args[1]
                self.goal_locations[box] = location
            # Note: Other goal predicates like 'clear' are ignored by this heuristic

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        # Initialize distances for all nodes known in the graph
        distances = {node: float('inf') for node in self.graph}

        # Check if start_node is a valid graph node before starting BFS
        if start_node not in self.graph:
             # This start node is not part of the connected grid defined by 'adjacent' facts
             # print(f"Warning: BFS started from node {start_node} not in graph.")
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # current_node must be in self.graph if it was added to the queue
            for neighbor in self.graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current locations of all boxes
        current_box_locations = {} # Maps box_name -> location_string
        for fact in state:
            parts = get_parts(fact)
            # Check for '(at box_name loc_name)' facts
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                 box, location = parts[1], parts[2]
                 current_box_locations[box] = location

        total_cost = 0

        # Sum distances for all boxes not at their goal
        for box, goal_location_str in self.goal_locations.items():
            current_location_str = current_box_locations.get(box)

            # If a box required by the goal is not found in the current state,
            # it's an invalid state or a state representation issue.
            # Assuming valid states always contain all relevant box locations.
            if current_location_str is None:
                 # This should not happen in a valid Sokoban state where boxes
                 # are always located somewhere. Treat as unsolvable.
                 # print(f"Error: Box {box} required by goal not found in state.")
                 return float('inf')

            if current_location_str != goal_location_str:
                current_loc_tuple = parse_location(current_location_str)
                goal_loc_tuple = parse_location(goal_location_str)

                # Check if parsing was successful
                if current_loc_tuple is None or goal_loc_tuple is None:
                     # print(f"Error: Failed to parse location string for box {box}: current='{current_location_str}', goal='{goal_location_str}'")
                     return float('inf') # Indicate error/unsolvable state

                # Check if both locations are valid nodes in the graph and reachable
                # Check if current_loc_tuple is a key in distances_map
                if current_loc_tuple not in self.distances_map:
                     # print(f"Error: Current location tuple {current_loc_tuple} not found in distances_map.")
                     return float('inf') # Current location is not a known grid node

                # Check if goal_loc_tuple is reachable from current_loc_tuple
                dist = self.distances_map[current_loc_tuple].get(goal_loc_tuple, float('inf'))

                if dist == float('inf'):
                    # Goal is unreachable from current location on the static grid
                    # print(f"Debug: Goal {goal_loc_tuple} unreachable from {current_loc_tuple} for box {box}.")
                    return float('inf') # State is likely unsolvable or a deadlock

                total_cost += dist

        # If total_cost is 0, it means all boxes are at their goal locations.
        # This matches the goal condition for the boxes.
        # The heuristic is 0 iff the box goal conditions are met.

        return total_cost
