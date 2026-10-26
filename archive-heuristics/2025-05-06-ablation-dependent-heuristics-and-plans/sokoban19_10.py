from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total effort required to move all boxes to their
    respective goal locations. It sums the minimum number of pushes required for
    each misplaced box (approximated by grid distance) and the minimum number
    of robot moves required to reach a position adjacent to the box for the
    first push (also approximated by grid distance).

    # Assumptions
    - The problem takes place on a grid-like structure defined by `adjacent` facts.
    - Each box has a unique goal location specified in the problem file.
    - The grid graph defined by `adjacent` facts is connected.
    - The heuristic uses shortest path distances on the grid graph, ignoring
      state-dependent obstacles (other boxes, clear status) for simplicity and speed.
      This makes the heuristic non-admissible but potentially effective for greedy search.

    # Heuristic Initialization
    - Parses `adjacent` facts to build a graph representation of the locations.
    - Precomputes all-pairs shortest path distances between all locations on this graph
      using BFS. This represents the minimum number of moves/pushes between locations
      on an empty grid.
    - Parses `goal` facts to identify the target location for each box.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize the total heuristic value to 0.
    3. For each box that is not currently at its designated goal location:
       a. Get the box's current location (`l_box`) and its goal location (`g_box`).
       b. Add the precomputed shortest path distance from `l_box` to `g_box` on the grid graph
          to the total heuristic. This estimates the minimum number of pushes required
          for this box, ignoring obstacles.
       c. Find all locations (`l_push`) that are adjacent to `l_box` according to the
          grid graph.
       d. Calculate the minimum precomputed shortest path distance from the robot's
          current location (`l_robot`) to any of these adjacent locations (`l_push`).
          This estimates the minimum robot movement cost to get into a position
          to push the box, ignoring state-dependent obstacles.
       e. Add this minimum robot distance to the total heuristic.
    4. The final total heuristic value is the sum of these costs for all misplaced boxes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, precomputing
        distances, and extracting goal locations.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # Data structures to represent the location graph
        self.adj_list = {} # map loc_name -> {direction: loc_name}
        self.all_locations = set()

        # Build adjacency list from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.all_locations.add(l1)
                self.all_locations.add(l2)
                if l1 not in self.adj_list:
                    self.adj_list[l1] = {}
                self.adj_list[l1][direction] = l2

        # Precompute all-pairs shortest path distances on the location graph (empty grid)
        self.location_distances = {}
        for start_loc in self.all_locations:
            self.location_distances[start_loc] = self._bfs_distances(start_loc)

        # Store goal locations for each box
        self.goal_locations = {} # map box_name -> goal_loc_name
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at boxX loc_Y_Z)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def _bfs_distances(self, start_loc):
        """
        Performs BFS from a start location to find shortest path distances
        to all other locations on the grid graph defined by adj_list.
        Ignores obstacles.
        """
        distances = {loc: float('inf') for loc in self.all_locations}
        if start_loc not in self.all_locations:
             # Start location is not in the graph, cannot reach anywhere
             return distances

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc has neighbors in the graph
            if current_loc in self.adj_list:
                for neighbor_loc in self.adj_list[current_loc].values():
                    # Ensure neighbor_loc is a valid location in our set
                    if neighbor_loc in self.all_locations and distances[neighbor_loc] == float('inf'):
                        distances[neighbor_loc] = current_dist + 1
                        queue.append(neighbor_loc)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of strings)

        # Find robot and box locations in the current state
        robot_loc = None
        box_locations = {} # map box_name -> current_loc_name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                 box_name, loc_name = parts[1], parts[2]
                 box_locations[box_name] = loc_name

        # If robot location is not found, it's an invalid state for this heuristic
        if robot_loc is None or robot_loc not in self.all_locations:
             return float('inf') # Cannot solve without robot or if robot is in invalid location

        total_cost = 0  # Initialize heuristic cost

        # Sum costs for each misplaced box
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # Get current location

            # If box is not found in state or already at goal
            if current_loc is None or current_loc == goal_loc:
                continue

            # Ensure current_loc and goal_loc are in our graph
            if current_loc not in self.location_distances or goal_loc not in self.location_distances.get(current_loc, {}):
                 # Should not happen in valid problems, but handle defensively
                 return float('inf')

            # Cost 1: Minimum pushes for the box (grid distance)
            box_to_goal_dist = self.location_distances[current_loc][goal_loc]

            if box_to_goal_dist == float('inf'):
                 # Box cannot reach the goal on the empty grid. This state is likely unsolvable.
                 return float('inf')

            total_cost += box_to_goal_dist

            # Cost 2: Minimum robot moves to get adjacent to the box for the first push (grid distance)
            min_robot_dist_to_adj_box = float('inf')

            # Find locations adjacent to the box's current location
            # Ensure current_loc is in adj_list (it should be if it's in all_locations)
            if current_loc in self.adj_list:
                for neighbor_loc in self.adj_list[current_loc].values():
                    # Calculate distance from robot to this adjacent location
                    # Ensure robot_loc is in location_distances (checked above)
                    # Ensure neighbor_loc is in location_distances[robot_loc] (checked by BFS result)
                    robot_to_neighbor_dist = self.location_distances[robot_loc].get(neighbor_loc, float('inf'))
                    min_robot_dist_to_adj_box = min(min_robot_dist_to_adj_box, robot_to_neighbor_dist)

            if min_robot_dist_to_adj_box == float('inf'):
                 # Robot cannot reach any location adjacent to the box on the empty grid.
                 # This state is likely unsolvable.
                 return float('inf')

            total_cost += min_robot_dist_to_adj_box

        # The heuristic is 0 if and only if all boxes are at their goal locations.
        # If there are no misplaced boxes, the loop finishes with total_cost = 0.
        # If there are misplaced boxes, total_cost will be > 0 (assuming connected grid).

        return total_cost
