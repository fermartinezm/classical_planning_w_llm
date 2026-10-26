import re
from collections import deque
from heuristics.heuristic_base import Heuristic
from task import Task # Import Task class for type hinting if needed, or just for clarity

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a fact string like '(predicate arg1 arg2)' into a tuple."""
    # Remove parentheses and split by spaces
    # Handle potential leading/trailing whitespace
    parts = fact_string.strip().strip('()').split()
    return tuple(parts)

# Helper function to parse location names like 'loc_R_C'
# Note: This function is included but not strictly used by the heuristic
# for distance calculations, which rely on the graph structure from 'adjacent' facts.
def parse_location_name(loc_name):
    """Parses a location name string 'loc_R_C' into a (row, col) tuple."""
    match = re.match(r'loc_(\d+)_(\d+)', loc_name)
    if match:
        # Return integers for row and column
        return (int(match.group(1)), int(match.group(2)))
    return None # Return None if the pattern doesn't match

# Helper function to get reverse direction
def reverse_direction(direction):
    """Returns the reverse direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen with valid directions

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing the minimum number
    of pushes required for each box to reach its goal location (approximated
    by shortest path distance on the grid graph), and adding the minimum
    distance the robot needs to travel to reach a position from which it can
    make the first push for any box that is not yet at its goal.

    Assumptions:
    - The problem involves a grid-like structure where locations are connected
      by 'adjacent' facts.
    - Location names follow the pattern 'loc_R_C' (though coordinates are not
      strictly used, the graph structure is derived from 'adjacent' facts).
    - Goal facts specify the target location for each box using '(at boxX locY)'.
      Assumes a one-to-one mapping between boxes and goal locations.
    - The graph defined by 'adjacent' facts is static.

    Heuristic Initialization:
    1. Parses static 'adjacent' facts to build a directed graph representing
       possible movements between locations and their directions. Also builds
       an undirected graph for shortest path calculations.
    2. Parses goal facts to map each box to its target goal location.
    3. Computes all-pairs shortest path distances between all locations
       on the undirected graph using BFS. This precomputation makes distance
       lookups efficient during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the robot's current location and the current location of each box
       from the state facts.
    3. Initialize total box-goal distance to 0 and minimum robot-to-push-position
       distance to infinity.
    4. Iterate through each box and its assigned goal location:
       a. Get the box's current location.
       b. If the box is not at its goal location:
          i. Calculate the shortest path distance from the box's current location
             to its goal location using the precomputed distances. Add this to
             the total box-goal distance. If the goal is unreachable, the
             heuristic is infinity.
          ii. Find all potential 'next' locations adjacent to the box's current
              location such that pushing the box to that 'next' location moves
              it strictly closer to its goal location (based on precomputed distance).
          iii. For each such 'next' location, determine the required robot
               pushing position. Based on the PDDL 'push' action, if the box
               moves from loc_b to loc_next in direction 'dir', the robot must
               be at loc_r such that adjacent(loc_r, loc_b, dir). This means
               loc_r is adjacent to loc_b in the *reverse* direction of 'dir'.
          iv. Calculate the shortest path distance from the robot's current
              location to each required robot pushing position found in step iii.
          v. Update the overall minimum robot-to-push-position distance with
             the minimum distance found for this box across all valid push
             directions/positions.
    5. The heuristic value is the sum of the total box-goal distance and the
       minimum robot-to-push-position distance (over all non-goal boxes).
       If any required distance calculation resulted in infinity (e.g., unreachable
       location), the heuristic is infinity.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_locations = {}  # Map box name to goal location name
        self.locations = set()    # Set of all location names
        # Directed graph: loc -> {direction -> adjacent_loc}
        self.directed_adj = {}
        # Undirected graph: loc -> [adjacent_loc1, adjacent_loc2, ...]
        self.undirected_adj = {}
        # Precomputed shortest path distances: (loc1, loc2) -> distance
        self.distances = {}

        # Parse static facts to build graphs
        for fact_string in task.static:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'adjacent':
                # Fact is (adjacent l1 l2 dir)
                _, l1, l2, direction = parsed
                self.locations.add(l1)
                self.locations.add(l2)

                # Build directed graph
                if l1 not in self.directed_adj:
                    self.directed_adj[l1] = {}
                self.directed_adj[l1][direction] = l2

                # Build undirected graph
                if l1 not in self.undirected_adj:
                    self.undirected_adj[l1] = []
                if l2 not in self.undirected_adj:
                    self.undirected_adj[l2] = []
                # Add edges in both directions for undirected graph
                if l2 not in self.undirected_adj[l1]:
                    self.undirected_adj[l1].append(l2)
                if l1 not in self.undirected_adj[l2]:
                    self.undirected_adj[l2].append(l1)


        # Parse goal facts to map boxes to goal locations
        for fact_string in task.goals:
            parsed = parse_fact(fact_string)
            # Goal facts are typically (at boxX locY)
            if parsed[0] == 'at' and len(parsed) == 3 and parsed[1].startswith('box'):
                 _, box_name, loc_name = parsed
                 self.goal_locations[box_name] = loc_name

        # Compute all-pairs shortest path distances using BFS
        self._compute_all_pairs_distances()

    def _compute_all_pairs_distances(self):
        """Computes shortest path distances between all pairs of locations."""
        for start_loc in self.locations:
            self.distances.update(self._bfs(start_loc))

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()
            dist = distances[current_loc]

            if current_loc in self.undirected_adj:
                for neighbor in self.undirected_adj[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = dist + 1
                        queue.append(neighbor)

        # Store distances in the self.distances dictionary using (start_node, end_node) as key
        return { (start_node, end_node): dist for end_node, dist in distances.items() }


    def get_distance(self, loc1, loc2):
        """Retrieves the precomputed shortest path distance between two locations."""
        # Handle cases where loc1 or loc2 might not be in the graph (shouldn't happen with valid states/goals)
        if loc1 not in self.locations or loc2 not in self.locations:
             return float('inf')
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Keyword arguments:
        node -- the current state node
        """
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        robot_loc = None
        box_locations = {} # Map box name to current location name

        # Parse current state to find robot and box locations
        for fact_string in state:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'at-robot' and len(parsed) == 2:
                robot_loc = parsed[1]
            elif parsed[0] == 'at' and len(parsed) == 3 and parsed[1].startswith('box'):
                _, box_name, loc_name = parsed
                box_locations[box_name] = loc_name

        # If robot or any expected box is not found, state is likely invalid or incomplete
        if robot_loc is None or len(box_locations) != len(self.goal_locations):
             # This indicates a problem with the state representation or parsing
             return float('inf')

        total_box_goal_dist = 0
        min_robot_push_dist = float('inf') # Minimum distance from robot to *any* valid push position for *any* non-goal box

        non_goal_boxes_exist = False

        # Calculate box-goal distances and find potential push positions
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            if current_loc is None:
                 # Box location not found in state, indicates invalid state
                 return float('inf')

            if current_loc != goal_loc:
                non_goal_boxes_exist = True
                box_goal_dist = self.get_distance(current_loc, goal_loc)

                # If a box cannot reach its goal, the state is likely a dead end
                if box_goal_dist == float('inf'):
                    return float('inf')

                total_box_goal_dist += box_goal_dist

                # Find valid push positions for this box towards its goal
                valid_push_positions_for_this_box = []
                current_box_goal_dist = self.get_distance(current_loc, goal_loc)

                # Iterate through possible push directions from current_loc
                if current_loc in self.directed_adj:
                    for push_dir, next_loc in self.directed_adj[current_loc].items():
                        # Check if pushing in this direction moves the box strictly closer to the goal
                        dist_to_goal_from_next = self.get_distance(next_loc, goal_loc)
                        if dist_to_goal_from_next < current_box_goal_dist:
                            # This is a step towards the goal.
                            # The required robot location to push from current_loc to next_loc in push_dir
                            # is adjacent to current_loc in the reverse direction of push_dir.
                            rev_dir = reverse_direction(push_dir)
                            if current_loc in self.directed_adj and rev_dir in self.directed_adj[current_loc]:
                                required_robot_loc = self.directed_adj[current_loc][rev_dir]
                                valid_push_positions_for_this_box.append(required_robot_loc)

                # Calculate robot distance to the closest valid push position for this box
                min_dist_to_push_pos_for_this_box = float('inf')
                for push_pos in valid_push_positions_for_this_box:
                    dist_robot_to_push_pos = self.get_distance(robot_loc, push_pos)
                    min_dist_to_push_pos_for_this_box = min(min_dist_to_push_pos_for_this_box, dist_robot_to_push_pos)

                # Update the overall minimum robot distance to any push position
                min_robot_push_dist = min(min_robot_push_dist, min_dist_to_push_pos_for_this_box)

        # If no non-goal boxes exist, the heuristic is 0 (already handled by goal_reached check)
        # If non-goal boxes exist but no valid push positions were found for any of them,
        # min_robot_push_dist will remain infinity. This indicates a potential deadlock.
        if non_goal_boxes_exist and min_robot_push_dist == float('inf'):
             return float('inf')

        # If no non-goal boxes, heuristic is 0. Otherwise, sum the components.
        if not non_goal_boxes_exist:
             return 0

        return total_box_goal_dist + min_robot_push_dist
