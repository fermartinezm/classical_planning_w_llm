import math
from collections import deque

# Assuming heuristic_base.py is available in the same structure or path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to compute all-pairs shortest paths using BFS
def compute_distances(graph, locations):
    """Computes shortest path distances between all pairs of locations in the graph."""
    distances = {}
    for start_loc in locations:
        distances[start_loc] = {}
        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()
            distances[start_loc][current_loc] = dist

            # Iterate through neighbors
            for neighbor_loc in graph.get(current_loc, {}).values():
                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))
    return distances

# Helper function for opposite directions
def get_opposite_direction(direction):
    """Returns the opposite direction."""
    opposite_map = {
        'up': 'down',
        'down': 'up',
        'left': 'right',
        'right': 'left'
    }
    return opposite_map.get(direction)

# Main heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each box to reach its goal location and
    adding the estimated cost for the robot to get into a position
    to push one of the boxes towards its goal.

    # Assumptions
    - The grid structure is implied by location names like 'loc_R_C'.
    - Shortest path distances between locations are precomputed on the full grid,
      ignoring dynamic obstacles (other boxes, robot, clear status). This provides
      an optimistic estimate for both box movement (minimum pushes) and robot movement.
    - The cost of moving a box is estimated by the shortest path distance on the full grid.
    - The cost of robot movement is estimated by the shortest path distance on the full grid
      to a valid pushing position for *any* box that needs moving.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Parses goal conditions to map boxes to their target locations.
    - Builds adjacency graphs (forward and reverse) from static 'adjacent' facts.
    - Collects all possible locations.
    - Computes all-pairs shortest path distances between locations on the full grid graph using BFS.
    - Sets up a mapping for opposite directions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify which boxes are not currently at their goal locations. If all boxes are at their goals, the heuristic is 0.
    3. Initialize the total heuristic value `h` to 0.
    4. For each box that is not at its goal:
       - Get the box's current location and its goal location.
       - Calculate the shortest path distance between the box's current location and its goal location using the precomputed distances on the full grid. This distance represents the minimum number of push actions required for this box in an ideal scenario (ignoring obstacles and robot positioning). Add this distance to `h`. If the goal is unreachable on the full grid, return infinity (or a large value) as the state is likely unsolvable or leads to a deadlock.
    5. Estimate the robot's contribution to the cost: The robot must move to a position from which it can push a box towards its goal.
       - Find the robot's current location.
       - Initialize a variable `min_robot_dist_to_push` to infinity.
       - Iterate through each box that is not at its goal:
          - Get the box's current location and its goal location.
          - Iterate through all possible directions (up, down, left, right).
          - For each direction, check if pushing the box in that direction is possible (i.e., there is an adjacent location in that direction according to the graph) and if pushing in that direction moves the box strictly closer to its goal (using precomputed distances).
          - If it's a valid and useful push direction:
             - Determine the required robot location (`R_push`) which is adjacent to the box's current location in the opposite direction of the push. Use the reverse adjacency graph to find this location.
             - If `R_push` exists:
                - Calculate the shortest path distance from the robot's current location to `R_push` using precomputed distances on the full grid.
                - Update `min_robot_dist_to_push` with the minimum distance found so far across all valid push positions for all boxes.
       - If `min_robot_dist_to_push` remains infinity (robot cannot reach any useful pushing position for any box on the full grid), return infinity (or a large value).
       - Add `min_robot_dist_to_push` to `h`.
    6. Return the total calculated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # 1. Parse goal conditions to map boxes to their target locations.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location

        # 2. Build adjacency graphs (forward and reverse) and collect all locations.
        self.adj_graph = {}
        self.rev_adj_graph = {}
        self.all_locations = set()

        # Map for opposite directions, needed for building rev_adj_graph
        temp_opposite_direction = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

                if loc1 not in self.adj_graph:
                    self.adj_graph[loc1] = {}
                self.adj_graph[loc1][direction] = loc2

                # Build reverse graph: loc1 is reachable from loc2 by moving opposite_direction[direction]
                opposite_dir = temp_opposite_direction.get(direction)
                if opposite_dir: # Ensure direction has an opposite
                    if loc2 not in self.rev_adj_graph:
                        self.rev_adj_graph[loc2] = {}
                    self.rev_adj_graph[loc2][opposite_dir] = loc1

        # 3. Compute all-pairs shortest path distances on the full grid graph.
        self.dist = compute_distances(self.adj_graph, list(self.all_locations))

        # 4. Store the mapping for opposite directions.
        self.opposite_direction = temp_opposite_direction


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current locations
        robot_curr = None
        box_locations = {}
        # clear_locations = set() # Not used in this heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_curr = parts[1]
            elif parts[0] == 'at' and parts[1] in self.box_goals: # Only track relevant boxes
                box_locations[parts[1]] = parts[2]
            # elif parts[0] == 'clear': # Not used
            #     clear_locations.add(parts[1])

        # Ensure robot location is found (should always be the case in valid states)
        if robot_curr is None:
            # This indicates an unexpected state structure. Return a high cost.
            return float('inf')

        # 2. Identify boxes not at goals.
        boxes_to_move = [
            box for box, goal_loc in self.box_goals.items()
            if box_locations.get(box) != goal_loc
        ]

        # 3. If all boxes are at goals, return 0.
        if not boxes_to_move:
            return 0

        # 4. Calculate box-to-goal distances
        h = 0
        for box_name in boxes_to_move:
            B_curr = box_locations.get(box_name) # Use .get() in case box is missing from state (shouldn't happen)
            G = self.box_goals[box_name]

            # If box location is not in the precomputed distances (e.g., disconnected part of graph), treat as unreachable
            if B_curr not in self.dist or G not in self.dist.get(B_curr, {}):
                 return float('inf')

            box_dist = self.dist[B_curr][G]

            # If a box goal is unreachable on the full grid, return infinity
            if box_dist is None: # This check is redundant with the one above, but kept for clarity
                return float('inf')

            h += box_dist

        # 5. Estimate robot movement cost
        min_robot_dist_to_push = float('inf')

        for box_name in boxes_to_move:
           B_curr = box_locations.get(box_name)
           G = self.box_goals[box_name]

           # Get current box-goal distance for comparison
           current_box_goal_dist = self.dist.get(B_curr, {}).get(G)
           if current_box_goal_dist is None:
               continue # Skip this box if its goal is unreachable (already handled in step 4)

           # Iterate through all possible required robot directions (up, down, left, right)
           for required_robot_dir in self.opposite_direction.keys():
               push_dir = self.opposite_direction[required_robot_dir]

               # Check if pushing in `push_dir` from B_curr is possible (location exists)
               B_next = self.adj_graph.get(B_curr, {}).get(push_dir)

               if B_next is not None: # Can push in this direction
                   dist_B_next_G = self.dist.get(B_next, {}).get(G)

                   # Check if pushing in this direction moves the box strictly closer to the goal
                   if dist_B_next_G is not None and dist_B_next_G < current_box_goal_dist:
                       # This is a valid and useful push direction (push_dir).
                       # Find the required robot location R_push.
                       # R_push is the location such that moving `required_robot_dir` from R_push gets to B_curr.
                       # This is found using the reverse adjacency graph: R_push = rev_adj_graph[B_curr][required_robot_dir].
                       R_push = self.rev_adj_graph.get(B_curr, {}).get(required_robot_dir)

                       if R_push is not None:
                           # Calculate robot distance to R_push using precomputed distances
                           # If robot_curr or R_push is not in the precomputed distances, treat as unreachable
                           if robot_curr in self.dist and R_push in self.dist.get(robot_curr, {}):
                               robot_dist = self.dist[robot_curr][R_push]
                               min_robot_dist_to_push = min(min_robot_dist_to_push, robot_dist)

        # If robot cannot reach any useful pushing position on the full grid, return infinity
        if min_robot_dist_to_push == float('inf'):
            # This can happen if all boxes are in dead-end positions or
            # the robot is in a disconnected part of the graph from all push positions.
            # Or if there are no boxes to move (handled earlier).
            # If boxes_to_move is not empty but min_robot_dist_to_push is inf,
            # it means no box can be pushed towards its goal from any reachable robot position
            # on the full grid. This suggests a deadlock or unsolvable state.
            return float('inf')

        h += min_robot_dist_to_push

        # 6. Return the total calculated heuristic value.
        return h
