from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to build adjacency list from static facts
def build_adjacency_list(static_facts):
    adj_list = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            locations.add(loc1)
            locations.add(loc2)
            if loc1 not in adj_list:
                adj_list[loc1] = set()
            if loc2 not in adj_list:
                adj_list[loc2] = set()
            # Add bidirectional edges
            adj_list[loc1].add(loc2)
            adj_list[loc2].add(loc1)

    # Ensure all locations mentioned in adjacent facts are keys in adj_list
    for loc in locations:
        if loc not in adj_list:
            adj_list[loc] = set()

    return adj_list, list(locations) # Return list of locations for BFS starting points

# Helper function to compute shortest path distances using BFS
def bfs_distances(start_node, adj_list):
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        curr = queue.popleft()
        if curr in adj_list:
            for neighbor in adj_list[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of pushes required to move each box
    to its goal location, summed over all boxes. It uses shortest path
    distances on the grid graph defined by adjacent locations.

    # Assumptions:
    - The grid structure is defined by the 'adjacent' static facts.
    - Goals are specified as '(at ?box ?location)'.
    - The heuristic ignores the robot's position and the need for clear paths
      or robot positioning, focusing only on the minimum box movement distance.
    - It assumes that each box has a unique goal location specified in the goals.
    - It assumes the grid is connected such that all relevant locations (initial box locations, goal locations) are reachable from each other.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an undirected graph representing the grid connectivity based on
      'adjacent' static facts.
    - Precomputes shortest path distances between all pairs of locations
      on this grid graph using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of each box that has a goal specified
       by iterating through the state facts and finding predicates of the form
       '(at box_name location_name)'.
    2. Initialize the total heuristic cost to 0.
    3. For each box that has a specified goal location:
       - Retrieve the box's current location from the state.
       - Retrieve the box's goal location from the precomputed goal mapping.
       - If the current location is different from the goal location:
         - Look up the precomputed shortest path distance between the current
           location and the goal location on the grid graph.
         - Add this distance to the total heuristic cost.
         - If the goal location is unreachable from the current location
           (according to the precomputed distances), return infinity, as this
           state is likely unsolvable or on a path to an unsolvable state.
    4. After summing distances for all misplaced boxes, the total cost represents
       the heuristic value. This value is 0 if and only if all boxes are at their
       goals (i.e., the state is a goal state). Return the calculated total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid graph for distance calculations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at obj loc)
            if parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # Assuming objects in goal 'at' predicates are the boxes we need to move
                self.goal_locations[obj] = location

        # 2. Build the adjacency graph from static facts.
        self.adj_list, self.all_locations = build_adjacency_list(static_facts)

        # 3. Precompute shortest path distances between all pairs of locations.
        # self.distances[loc1][loc2] will store the shortest distance from loc1 to loc2.
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = bfs_distances(start_loc, self.adj_list)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required pushes
        (sum of box-to-goal distances).
        """
        state = node.state

        # Find current locations of all boxes that have a goal specified.
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if it's an (at obj loc) fact and if the object is one of the boxes we care about (i.e., has a goal)
            if parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_locations:
                 box, location = parts[1], parts[2]
                 current_box_locations[box] = location

        total_cost = 0

        # Sum distances for all boxes not at their goal.
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box from the goals isn't found in the current state,
            # this indicates an inconsistency or error in the state representation
            # or problem definition. Return infinity to signal a bad state.
            if current_location is None:
                 return float('inf')

            if current_location != goal_location:
                 # Add the shortest path distance from current location to goal location
                 # Check if both locations are valid and reachable in the precomputed distances.
                 # If goal_location is not reachable from current_location, bfs_distances won't
                 # have an entry for goal_location under distances[current_location].
                 if current_location in self.distances and goal_location in self.distances.get(current_location, {}):
                     total_cost += self.distances[current_location][goal_location]
                 else:
                     # If the goal is unreachable from the current location, this state is likely unsolvable.
                     # Return infinity to prune this path.
                     return float('inf')

        # The heuristic is 0 iff it's a goal state.
        # Our sum is 0 iff all boxes are at their goals (since distances are non-negative).
        # If total_cost is 0, it means all boxes are at their goals, which is the goal condition.
        # If total_cost is > 0, at least one box is not at its goal.
        # The case where total_cost is infinity is handled inside the loop.
        return total_cost
