from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions outside the class for clarity and potential reuse
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for number of parts vs args (handles exact matches and simple wildcards)
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use fnmatch for pattern matching on each corresponding part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing:
    1. The shortest path distance for each box from its current location to its goal location on the grid graph.
    2. The shortest path distance from the robot's current location to the nearest box that needs to be moved.

    The grid graph is derived from the `adjacent` facts in the PDDL domain, representing all traversable locations and connections. The distances are precomputed using BFS. This heuristic is non-admissible as it ignores dynamic obstacles (other boxes, robot, clear status) when calculating box-goal distances and simplifies the robot's role.

    # Assumptions
    - The grid structure is defined by `adjacent` facts. All locations mentioned in the problem are part of this grid.
    - Shortest path distance on the grid is a reasonable, albeit simplified, estimate for box movement cost (ignoring dynamic obstacles and the push mechanic details).
    - Robot movement cost is estimated by the distance to the nearest box needing attention.
    - This heuristic is non-admissible.

    # Heuristic Initialization
    - Parses all `location` objects mentioned in the task (initial state, goals, static facts) to build a mapping from location names to internal integer identifiers.
    - Builds the undirected adjacency graph based on `adjacent` facts.
    - Precomputes all-pairs shortest paths on the grid graph using BFS. Stores distances in a dictionary `self.distances[loc1_name][loc2_name]`.
    - Extracts goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state (`node.state`), which is a frozenset of facts.
    2. Identify the current location of the robot and all boxes by iterating through the state facts.
    3. Initialize `total_cost = 0`.
    4. Create a list `boxes_to_move` containing boxes that are not currently at their goal locations (as determined during initialization).
    5. If `boxes_to_move` is empty, the current state is a goal state, and the heuristic is 0.
    6. If there are boxes to move:
       a. For each box in `boxes_to_move`:
          i. Find its current location `current_l`.
          ii. Find its goal location `goal_l` from `self.goal_locations`.
          iii. Look up the precomputed shortest path distance from `current_l` to `goal_l` in `self.distances`.
          iv. If the distance is infinity (meaning the goal is unreachable on the grid), return `float('inf')` as the state is likely a dead end.
          v. Add this distance to `total_cost`. This component estimates the total "box movement effort".
       b. Find the robot's current location `robot_l`.
       c. Calculate the minimum shortest path distance from `robot_l` to the location of any box in `boxes_to_move`.
       d. If this minimum distance is infinity (meaning the robot cannot reach any box), return `float('inf')`.
       e. Add this minimum distance to `total_cost`. This component estimates the robot's effort to get to where it can start pushing.
    7. Return the calculated `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph, precomputing
        distances, and extracting goal locations.
        """
        self.goals = task.goals
        static_facts = task.static # Static facts include 'adjacent' relations

        # 1. Build the grid graph and location mapping
        # Collect all unique location names from initial state, goals, and static facts
        all_locations_set = set()
        for fact in task.initial_state | task.goals | static_facts:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith('loc_'):
                    all_locations_set.add(part)

        self.location_names = sorted(list(all_locations_set)) # Ensure consistent ordering
        # Create a mapping from location name string to an integer index
        self.loc_to_idx = {loc: i for i, loc in enumerate(self.location_names)}
        self.idx_to_loc = {i: loc for i, loc in enumerate(self.location_names)}
        num_locations = len(self.location_names)

        # Adjacency list representation of the grid graph
        self.adj_list = [[] for _ in range(num_locations)]
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1_name, loc2_name, _ = get_parts(fact)
                # Ensure both locations exist in our collected set
                if loc1_name in self.loc_to_idx and loc2_name in self.loc_to_idx:
                    loc1_idx = self.loc_to_idx[loc1_name]
                    loc2_idx = self.loc_to_idx[loc2_name]
                    # The graph is undirected for distance calculation purposes
                    if loc2_idx not in self.adj_list[loc1_idx]:
                         self.adj_list[loc1_idx].append(loc2_idx)
                    if loc1_idx not in self.adj_list[loc2_idx]:
                         self.adj_list[loc2_idx].append(loc1_idx)

        # 2. Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc_name in self.location_names:
            start_idx = self.loc_to_idx[start_loc_name]
            self.distances[start_loc_name] = self._bfs(start_idx, num_locations)

        # 3. Extract goal locations for each box
        self.goal_locations = {}
        # Goals are a frozenset of facts, e.g., {..., '(at box1 loc_2_4)', ...}
        for goal_fact in self.goals:
            predicate, *args = get_parts(goal_fact)
            # Check if it's an (at ?box ?location) goal
            if predicate == "at" and len(args) == 2 and args[0].startswith('box'):
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_idx, num_locations):
        """
        Performs BFS starting from start_idx to find distances to all other nodes.
        Returns a dictionary mapping location names to distances.
        """
        distances_from_start = {self.idx_to_loc[i]: float('inf') for i in range(num_locations)}
        distances_from_start[self.idx_to_loc[start_idx]] = 0
        queue = deque([start_idx])

        while queue:
            current_idx = queue.popleft()
            current_loc_name = self.idx_to_loc[current_idx]

            for neighbor_idx in self.adj_list[current_idx]:
                neighbor_loc_name = self.idx_to_loc[neighbor_idx]
                if distances_from_start[neighbor_loc_name] == float('inf'):
                    distances_from_start[neighbor_loc_name] = distances_from_start[current_loc_name] + 1
                    queue.append(neighbor_idx)

        return distances_from_start


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # Find current robot and box locations
        robot_location = None
        box_locations = {} # {box_name: location_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3 and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                box_locations[box] = loc
            # We don't need 'clear' facts for this heuristic

        total_cost = 0
        boxes_to_move = []

        # Calculate sum of box-goal distances
        # Iterate through goal boxes to find which ones need moving
        for box, goal_l in self.goal_locations.items():
            current_l = box_locations.get(box) # Get current location, default None if box isn't in state (unexpected)

            # If box is not at its goal location
            # Check if current_l is None first, although in valid states it should exist
            if current_l is None or current_l != goal_l:
                # If current_l is None, it means a box required by the goal is not in the state.
                # This is an invalid state or problem definition issue. Treat as unreachable.
                if current_l is None:
                     return float('inf')

                boxes_to_move.append(box)
                # Add distance from current box location to goal location
                # Use precomputed distances. Handle cases where goal might be unreachable (dist is inf)
                # Ensure current_l and goal_l are valid locations in our precomputed distances
                if current_l not in self.distances or goal_l not in self.distances.get(current_l, {}):
                     # This should not happen if all locations were collected correctly in __init__
                     # but as a safeguard, treat as unreachable.
                     return float('inf')

                dist = self.distances[current_l][goal_l]

                if dist == float('inf'):
                    # Goal is unreachable for this box on the grid. This state is likely a dead end.
                    # A high heuristic value is appropriate.
                    return float('inf')
                total_cost += dist

        # If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        # Calculate robot distance to the nearest box that needs moving
        min_robot_box_dist = float('inf')
        # Ensure robot_location was found and is in the grid graph
        if robot_location is None or robot_location not in self.distances:
             # Robot location is missing or not in the grid graph. Likely an invalid state.
             return float('inf')

        for box in boxes_to_move:
            box_l = box_locations.get(box) # Get current box location

            # Ensure box location was found and is in the grid graph
            if box_l is None or box_l not in self.distances.get(robot_location, {}):
                 # Box location missing or unreachable from robot location on the grid.
                 return float('inf')

            # Distance from robot to box location
            dist = self.distances[robot_location][box_l]
            min_robot_box_dist = min(min_robot_box_dist, dist)

        # If robot cannot reach any box needing move (min_robot_box_dist is still inf),
        # this state is likely a dead end.
        if min_robot_box_dist == float('inf'):
             return float('inf')

        total_cost += min_robot_box_dist

        return total_cost
