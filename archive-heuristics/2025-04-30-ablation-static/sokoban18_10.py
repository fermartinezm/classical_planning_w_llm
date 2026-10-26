# from heuristics.heuristic_base import Heuristic # This import is needed in the actual planner environment

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for BFS
from collections import deque

def bfs(start_loc, graph):
    """
    Performs BFS from start_loc on the graph to find distances to all reachable locations.
    Returns a dictionary mapping location names to distances.
    """
    distances = {start_loc: 0}
    queue = deque([start_loc])

    while queue:
        current_loc = queue.popleft()
        current_dist = distances[current_loc]

        # Iterate through neighbors in all directions
        # graph[current_loc] is a dict {direction: neighbor_loc}
        for neighbor_loc in graph.get(current_loc, {}).values():
             # Check if neighbor_loc is a valid location and not already visited
             # The graph structure already ensures neighbor_loc is valid if it exists
             if neighbor_loc not in distances:
                 distances[neighbor_loc] = current_dist + 1
                 queue.append(neighbor_loc)

    return distances

# Assuming Heuristic base class is available from heuristics.heuristic_base
# If not, define a mock class or adjust import path
# from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest path distances
    for each box from its current location to its goal location on the grid graph.
    The distance is calculated using Breadth-First Search (BFS) on the graph defined by
    the 'adjacent' predicates. This represents the minimum number of 'push' actions
    required for each box if the path were clear and the robot were optimally positioned.

    # Assumptions
    - The goal state specifies the target location for each box using the '(at ?box ?location)' predicate.
    - The grid structure and connectivity are fully defined by the 'adjacent' predicates.
    - Each box has a unique goal location. (This is typical for Sokoban instances).
    - The heuristic only considers the box-goal distance and does not explicitly model
      robot movement cost or potential deadlocks beyond reachability.

    # Heuristic Initialization
    - Parses the 'adjacent' facts from the static information to build a graph representation
      of the locations. The graph is treated as undirected for distance calculation.
    - Parses the goal conditions to identify the target location for each box.
    - Precomputes the shortest path distances from each goal location to all other reachable
      locations using BFS. This allows for fast distance lookups during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current location of the robot and each box from the state facts.
    2.  Check if the current state is the goal state. If yes, return 0.
    3.  Initialize the total heuristic cost to 0.
    4.  For each box specified in the goal:
        a.  Determine the box's current location from the state.
        b.  Determine the box's goal location (precomputed during initialization).
        c.  If the box is not already at its goal location:
            i.  Look up the precomputed shortest distance from the box's current location
                to its goal location using the distance map calculated during initialization
                (BFS was run from the goal location).
            ii. If the goal location is unreachable from the box's current location
                (i.e., the current location is not found in the precomputed distance map
                for that goal), the state is likely a deadlock or unsolvable path.
                Return infinity (`float('inf')`) as the heuristic value.
            iii. Otherwise, add this distance to the total heuristic cost.
    5.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and precomputing
        distances from goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        # Graph structure: {loc1: {direction1: loc2, direction2: loc3}, ...}
        self.graph = {}
        # Map directions to their reverses
        reverse_directions = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.graph:
                    self.graph[loc1] = {}
                self.graph[loc1][direction] = loc2

                # Add reverse direction for undirected movement on grid
                reverse_dir = reverse_directions.get(direction)
                if reverse_dir:
                     if loc2 not in self.graph:
                         self.graph[loc2] = {}
                     self.graph[loc2][reverse_dir] = loc1


        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is (at boxX loc_Y_Z)
                if len(args) == 2 and args[0].startswith('box'):
                     box, location = args[0], args[1] # Correctly assign box and location
                     self.goal_locations[box] = location

        # Precompute distances from each goal location to all other locations
        self.goal_distances = {}
        # Use set to handle multiple boxes going to the same goal (unlikely in Sokoban but safe)
        for goal_loc in set(self.goal_locations.values()):
             # Ensure goal location exists in the graph before running BFS
             if goal_loc in self.graph:
                 self.goal_distances[goal_loc] = bfs(goal_loc, self.graph)
             else:
                 # Handle case where a goal location is not in the graph (e.g., unreachable)
                 # This shouldn't happen in valid PDDL, but as a safeguard:
                 # print(f"Warning: Goal location {goal_loc} not found in graph.")
                 self.goal_distances[goal_loc] = {} # Empty map means all locations are inf distance


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is already reached
        if self.goals <= state:
             return 0

        # Track current box locations
        current_box_locations = {}
        # Find robot location (optional for this heuristic, not used in calculation)
        # robot_location = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                # Check if it's a box location fact
                if len(parts) == 3 and parts[1].startswith('box'):
                    box, location = parts[1], parts[2]
                    current_box_locations[box] = location
                # Check if it's the robot location fact (not used by this heuristic)
                # elif len(parts) == 2 and parts[0] == 'at-robot':
                #     robot_location = parts[1]


        total_heuristic = 0  # Initialize action cost counter.

        # Iterate through boxes that have a goal location
        for box, goal_location in self.goal_locations.items():
            # Get the current location of the box
            # Use .get() in case a box isn't in the state (shouldn't happen in valid states)
            current_location = current_box_locations.get(box)

            # If the box is not at its goal, add its distance to the heuristic
            if current_location and current_location != goal_location:
                # Look up the precomputed distance from current_location to goal_location
                # Note: BFS was run FROM the goal, so we look up current_location in the goal's distance map
                distance_map = self.goal_distances.get(goal_location)

                if distance_map is None:
                     # This goal location was not in the graph during init, likely unreachable
                     return float('inf')

                dist = distance_map.get(current_location, float('inf'))

                if dist == float('inf'):
                    # Current location cannot reach the goal location for this box
                    # This state is likely unsolvable or a deadlock
                    return float('inf')

                total_heuristic += dist

        # If we reached here, all boxes are either at their goal or have a finite path.
        # The heuristic is the sum of box-goal distances.
        return total_heuristic
