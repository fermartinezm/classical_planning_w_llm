import re
from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_graph_and_directions(static_facts):
    """
    Builds adjacency list graph and direction mapping from static facts.
    Returns: graph (dict), dir_map ((loc1, loc2) -> dir), opposite_dir_map (dir -> dir), locations (set)
    """
    graph = defaultdict(list)
    dir_map = {}
    locations = set()
    opposite_dir_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent' and len(parts) == 4:
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            graph[loc1].append(loc2)
            dir_map[(loc1, loc2)] = direction
            locations.add(loc1)
            locations.add(loc2)

    return graph, dir_map, opposite_dir_map, locations

def precompute_distances(graph, locations):
    """
    Computes shortest path distances between all pairs of locations using BFS.
    Returns: dist_map ((loc1, loc2) -> distance)
    """
    dist_map = {}
    for start_node in locations:
        q = deque([(start_node, 0)])
        visited = {start_node}
        dist_map[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist_map[(start_node, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    return dist_map

def find_first_step_on_shortest_path(dist_map, graph, start_loc, end_loc):
    """
    Finds the location of the first step on a shortest path from start_loc to end_loc.
    Assumes dist_map contains shortest distances.
    Returns: first_step_loc or None if no path or start==end.
    """
    dist = dist_map.get((start_loc, end_loc), float('inf'))
    if dist == 0 or dist == float('inf'):
        return None # Already at goal or unreachable

    # Look for a neighbor v of start_loc such that dist(v, end_loc) == dist - 1
    # Iterate through neighbors in the order they appear in the graph adjacency list
    # This provides a deterministic choice if multiple neighbors are on a shortest path
    for neighbor in graph.get(start_loc, []):
        if dist_map.get((neighbor, end_loc), float('inf')) == dist - 1:
            return neighbor # Found the first step

    return None # Should not happen if dist > 0 and graph is correct


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. For each box not at its goal, it calculates the
    cost as the sum of:
    1. The shortest path distance for the robot to reach the required position
       to push the box for its first step towards the goal.
    2. The shortest path distance for the box from its current location to its
       goal location on the grid (representing the minimum number of pushes).

    The total heuristic is the sum of these costs for all off-goal boxes.
    If any box goal is unreachable from its current location or the required
    push position is unreachable by the robot, the heuristic returns infinity.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Locations are named `loc_R_C`.
    - The cost of moving the robot is 1 per step.
    - The cost of a push action is 1.
    - The primary costs are robot movement to push position and the pushes themselves.
    - Interactions between boxes (blocking each other) are not explicitly modeled
      beyond reachability on the static grid graph. This is a simplification
      that makes the heuristic non-admissible but potentially faster to compute
      and informative for greedy search.
    - Deadlocks (boxes pushed into irretrievable positions) are partially
      handled by checking grid reachability to the goal.

    # Heuristic Initialization
    - Parses static facts (`adjacent` predicates) to build the location graph.
    - Builds a mapping of adjacent locations to the direction of movement.
    - Precomputes shortest path distances between all pairs of locations on the grid graph using BFS.
    - Parses goal conditions to store the target location for each box.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize total heuristic cost to 0.
    3. For each box:
       a. Check if the box is already at its goal location. If yes, continue to the next box.
       b. Get the box's current location (`loc_b`) and its goal location (`goal_b`).
       c. Calculate the shortest path distance for the box from `loc_b` to `goal_b` using the precomputed distances (`box_dist`). This represents the minimum number of pushes required.
       d. If `goal_b` is unreachable from `loc_b` on the grid graph (`box_dist` is infinity), the state is likely unsolvable or requires complex maneuvers not captured by this heuristic; return infinity.
       e. Find the location (`next_loc`) that is the first step on a shortest path for the box from `loc_b` towards `goal_b`. This is a neighbor `v` of `loc_b` such that `dist(v, goal_b) == box_dist - 1`.
       f. Determine the direction of movement from `loc_b` to `next_loc` using the precomputed direction map.
       g. Determine the required robot push position (`push_pos`) which is adjacent to `loc_b` in the direction opposite to the box's first step (`loc_b` to `next_loc`). This is a location `p` such that the direction from `p` to `loc_b` is the opposite of the direction from `loc_b` to `next_loc`.
       h. Calculate the shortest path distance for the robot from its current location (`robot_loc_str`) to the required push position (`push_pos`) using the precomputed distances (`robot_dist_to_push_pos`).
       i. If `push_pos` is unreachable by the robot (`robot_dist_to_push_pos` is infinity), return infinity.
       j. The estimated cost for this box is `robot_dist_to_push_pos + box_dist`.
       k. Add this cost to the total heuristic.
    4. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goals.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build graph and direction maps from adjacent facts
        self.graph, self.dir_map, self.opposite_dir_map, self.locations = \
            build_graph_and_directions(static_facts)

        # Precompute all-pairs shortest path distances on the grid graph
        # This graph only considers locations and adjacency, ignoring obstacles initially.
        self.dist_map = precompute_distances(self.graph, self.locations)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # (at box loc)
                box, location = args
                self.goal_locations[box] = location

        # Basic check: Ensure all goal locations exist in the graph
        # If a goal location is not in the graph, it's unreachable.
        for goal_loc in self.goal_locations.values():
             if goal_loc not in self.locations:
                 # The dist_map will correctly show infinity for paths to this location.
                 pass


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot and box locations in the current state
        robot_loc_str = None
        box_locations = {} # {box_name: location_str}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc_str = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                 box = parts[1]
                 loc = parts[2]
                 box_locations[box] = loc

        # If robot location is unknown, state is invalid for this domain
        if robot_loc_str is None:
             return float('inf')

        # Check if the state is the goal state
        if self.goals <= state:
             return 0

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each box that has a goal location defined
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # If box is not in the state (shouldn't happen in valid states) or already at goal, skip
            # (The goal state check above handles the "already at goal" case for all boxes)
            if current_location is None or current_location == goal_location:
                continue

            # Ensure current location is a valid node in the graph
            if current_location not in self.locations:
                 # Box is in an invalid location not on the grid
                 return float('inf')

            # 1. Calculate box distance to goal (minimum pushes) on the static grid
            box_dist = self.dist_map.get((current_location, goal_location), float('inf'))

            # If box cannot reach its goal on the static grid, it's likely stuck/unsolvable
            if box_dist == float('inf'):
                return float('inf')

            # If box is already at goal (box_dist == 0), we already handled it with 'continue'
            # So, only proceed if the box needs to move.
            if box_dist > 0:
                # 2. Find the first step location on a shortest path for the box
                # This requires finding a neighbor v of current_location such that dist(v, goal_location) == box_dist - 1
                next_loc = find_first_step_on_shortest_path(self.dist_map, self.graph, current_location, goal_location)

                # If box_dist > 0, next_loc should be found. If not, something is wrong with graph/dist_map.
                if next_loc is None:
                     # This implies box_dist was 0 (handled), or an internal error.
                     return float('inf')

                # 3. Determine the required robot push position for the first step
                # The direction from current_location to next_loc
                dir_b_to_next = self.dir_map.get((current_location, next_loc))
                if dir_b_to_next is None:
                     # Should not happen if next_loc is a valid neighbor from graph
                     return float('inf')

                # The opposite direction
                opposite_dir = self.opposite_dir_map.get(dir_b_to_next)
                if opposite_dir is None:
                     # Should not happen for standard directions
                     return float('inf')

                # Find the location 'p' such that 'p' is adjacent to current_location in the opposite direction
                push_pos = None
                # Iterate through neighbors of current_location and check the direction from neighbor to current_location
                for neighbor_of_box in self.graph.get(current_location, []):
                     if self.dir_map.get((neighbor_of_box, current_location)) == opposite_dir:
                          push_pos = neighbor_of_box
                          break

                if push_pos is None:
                     # This means there is no location adjacent to the box from which it can be pushed
                     # in the required direction. This might indicate a deadlock or an invalid grid structure.
                     # Treat as unsolvable from this state.
                     return float('inf')

                # 4. Calculate robot distance to the required push position
                robot_dist_to_push_pos = self.dist_map.get((robot_loc_str, push_pos), float('inf'))

                # If robot cannot reach the push position
                if robot_dist_to_push_pos == float('inf'):
                     return float('inf')

                # 5. Add cost for this box: robot movement to push position + box pushes
                # The cost is robot_dist_to_push_pos + box_dist
                # robot_dist_to_push_pos: cost to get robot into position for the first push
                # box_dist: minimum number of pushes (each push moves box 1 step, robot 1 step)
                # This is the core non-admissible simplification.
                cost_for_box = robot_dist_to_push_pos + box_dist
                total_cost += cost_for_box

        # The goal state check at the beginning ensures we return 0 for goal states.
        # If we reach here, it's not a goal state, and total_cost reflects the sum
        # for all off-goal boxes.
        return total_cost
