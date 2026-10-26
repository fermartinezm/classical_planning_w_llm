from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_4_4)" -> ["at", "box1", "loc_4_4"]
    return fact[1:-1].split()

# BFS function
def bfs_distance(start_loc, target_locs, adj_list):
    """
    Find the shortest path distance from start_loc to any location in target_locs
    using BFS on the adjacency list.
    Returns float('inf') if no target is reachable.
    """
    if start_loc in target_locs:
        return 0
    queue = deque([(start_loc, 0)])
    visited = {start_loc}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc in adj_list:
            for neighbor in adj_list[current_loc]:
                if neighbor in target_locs:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Not reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the
    minimum number of pushes required for each misplaced box (approximated
    by shortest path distance on the grid) and adding the minimum number
    of robot moves required to reach the closest misplaced box.

    # Assumptions
    - The grid structure is defined by `adjacent` facts, and movement is
      possible between adjacent locations in both directions unless explicitly
      prevented by the absence of a reverse `adjacent` fact. The heuristic
      assumes symmetric adjacency for distance calculation.
    - Shortest path distance on the grid is a reasonable approximation for
      the minimum number of pushes for a box and robot moves.
    - The cost of moving the robot between pushes for the same box is
      ignored or assumed to be small relative to box movement and initial
      robot positioning.
    - The heuristic does not consider dead-end states (where a box is pushed
      into a corner it cannot leave) beyond detecting simple graph unreachability.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an undirected adjacency list representation of the grid from
      the `adjacent` static facts to enable shortest path calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify the goal location for each box from the pre-computed goal information.
    3. Find all boxes that are not currently at their goal location (misplaced boxes).
    4. If there are no misplaced boxes, the state is a goal state, and the heuristic is 0.
    5. For each misplaced box, calculate the shortest path distance from its current
       location to its goal location using BFS on the grid. Sum these distances.
       This sum represents a lower bound on the total number of pushes required.
    6. Calculate the shortest path distance from the robot's current location to the
       location of the closest misplaced box. This estimates the cost for the robot
       to start working on a box.
    7. The total heuristic value is the sum of the total box distances (step 5)
       and the robot-to-closest-box distance (step 6). If any required location
       is unreachable via BFS, a large penalty is returned.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the grid adjacency list from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Build the adjacency list for the grid from 'adjacent' facts.
        # We build an undirected graph for distance calculations, assuming
        # movement is possible in both directions between adjacent locations.
        self.adj_list = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2 = parts[1], parts[2]
                if loc1 not in self.adj_list:
                    self.adj_list[loc1] = set()
                self.adj_list[loc1].add(loc2)
                # Assuming adjacency is symmetric for movement distance
                if loc2 not in self.adj_list:
                    self.adj_list[loc2] = set()
                self.adj_list[loc2].add(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot and box locations in the current state.
        robot_location = None
        box_locations = {} # Map box name to its location string

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_location = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'): # Assuming box objects start with 'box'
                box, location = parts[1], parts[2]
                box_locations[box] = location

        # Identify misplaced boxes and calculate sum of their distances to goals.
        total_box_distance = 0
        misplaced_boxes = [] # List of (box_name, current_loc)

        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box) # Get goal location for this box

            # Only consider boxes that have a specified goal location
            if goal_loc:
                 if current_loc != goal_loc:
                     # Calculate distance for this box
                     dist = bfs_distance(current_loc, {goal_loc}, self.adj_list)
                     if dist == float('inf'):
                         # Box cannot reach its goal - this state is likely a dead end
                         # Return a large value to prune this branch in GBFS.
                         return 1000000 # Penalty for unreachable goal

                     total_box_distance += dist
                     misplaced_boxes.append((box, current_loc))
            # Note: Boxes present in state but not in goals are ignored by this heuristic.
            # This is standard if the goal only specifies a subset of objects.

        # If all relevant boxes are at their goals, the heuristic is 0.
        if not misplaced_boxes:
            return 0

        # Calculate distance from robot to the closest misplaced box.
        # This estimates the cost for the robot to start working on the first box.
        closest_box_distance = float('inf')
        misplaced_box_locations = {loc for _, loc in misplaced_boxes}

        if robot_location: # Robot location should always exist in a valid state
             dist_robot_to_closest_box = bfs_distance(robot_location, misplaced_box_locations, self.adj_list)
             if dist_robot_to_closest_box == float('inf'):
                 # Robot cannot reach any location where a misplaced box is.
                 # This state is likely a dead end.
                 return 1000000 # Penalty for unreachable boxes

             closest_box_distance = dist_robot_to_closest_box
        else:
             # Should not happen in valid Sokoban states, but handle defensively
             # If robot location is missing, it cannot move boxes.
             return 1000000 # Penalty for invalid state (missing robot)


        # The heuristic is the sum of minimum pushes for all boxes
        # plus the cost for the robot to reach the first box.
        heuristic_value = total_box_distance + closest_box_distance

        return heuristic_value
