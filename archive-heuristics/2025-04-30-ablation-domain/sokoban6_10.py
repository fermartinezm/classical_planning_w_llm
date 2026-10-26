from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None # Not a valid loc_R_C format
    return None

def bfs(start_node, adj_list, all_nodes):
    """Computes shortest distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not in the set of known locations, cannot compute distances
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # adj_list is loc_tuple -> {direction: neighbor_loc_tuple}
        # Iterate over neighbor locations regardless of the direction key
        for neighbor_loc_tuple in adj_list.get(current_node, {}).values():
            if distances[neighbor_loc_tuple] == float('inf'): # Check if visited
                distances[neighbor_loc_tuple] = current_dist + 1
                queue.append(neighbor_loc_tuple)
    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    number of pushes required for each box (graph distance to its goal) and
    the minimum number of robot moves required to reach a position adjacent
    to any box that still needs to be moved.

    # Assumptions
    - The locations form a graph defined by `adjacent` predicates.
    - Locations are named using the pattern `loc_R_C`.
    - The heuristic ignores dynamic obstacles (other boxes, robot blocking push paths)
      and `clear` predicates, except for defining the static graph structure.
    - It does not explicitly detect or penalize complex deadlocks (e.g., box in corner).
    - Assumes a one-to-one mapping between boxes and goal locations defined in the goal state.

    # Heuristic Initialization
    - Parses the goal state to find the target location for each box.
    - Identifies all unique locations present in the initial state and static facts.
    - Builds an adjacency list representation of the location graph based on `adjacent` facts.
    - Computes all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize the total heuristic cost `h` to 0.
    3. Create a list of boxes that are not currently at their designated goal location.
    4. For each box in this list:
       - Get its current location and its goal location.
       - Find the shortest path distance between the box's current location and its goal location using the precomputed distances. This distance represents the minimum number of pushes required for this box assuming free movement in the graph.
       - Add this distance to `h`. If the goal is unreachable from the box's current location in the graph, the state is likely unsolvable, and the heuristic returns infinity.
    5. If the list of boxes not at their goal is not empty:
       - Find the minimum shortest path distance from the robot's current location to *any* location that is adjacent to *any* box in the list of boxes not at their goal. This estimates the robot's cost to get into a position to start pushing *some* box.
       - Add this minimum distance to `h`. If the robot cannot reach any location adjacent to any box that needs moving in the graph, the state is likely unsolvable, and the heuristic returns infinity.
    6. Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing graph distances and goal locations."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find all locations

        # 1. Parse goal locations for each box
        self.goal_locations = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                loc_str = parts[2]
                loc_tuple = parse_location(loc_str)
                if loc_tuple:
                    self.goal_locations[box_name] = loc_tuple
                # else: Warning already printed by parse_location if it returns None


        # 2. Collect all unique locations
        self.locations = set()
        # Check static facts (adjacent defines most locations)
        for fact in self.static_facts:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith('loc_'):
                    loc_tuple = parse_location(part)
                    if loc_tuple:
                        self.locations.add(loc_tuple)
        # Check initial state (robot and box locations, clear locations)
        for fact in self.initial_state:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('loc_'):
                     loc_tuple = parse_location(part)
                     if loc_tuple:
                         self.locations.add(loc_tuple)

        # 3. Build adjacency list graph
        self.adj = {} # loc_tuple -> {direction_str: neighbor_loc_tuple}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                _, loc1_str, loc2_str, dir_str = parts
                loc1_tuple = parse_location(loc1_str)
                loc2_tuple = parse_location(loc2_str)
                if loc1_tuple and loc2_tuple:
                    self.adj.setdefault(loc1_tuple, {})[dir_str] = loc2_tuple
                # else: Warning already printed by parse_location if it returns None


        # 4. Compute all-pairs shortest path distances
        self.distances = {} # loc_tuple1 -> {loc_tuple2: distance}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(start_loc, self.adj, self.locations)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current robot and box locations
        robot_loc_tuple = None
        current_box_locations = {} # box_name -> loc_tuple

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot' and len(parts) == 2:
                loc_str = parts[1]
                robot_loc_tuple = parse_location(loc_str)
                if robot_loc_tuple is None:
                     # Should not happen in valid states if locations are well-formed
                     return float('inf') # Indicate unsolvable/invalid state
            elif parts and parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                loc_str = parts[2]
                loc_tuple = parse_location(loc_str)
                if loc_tuple:
                    current_box_locations[box_name] = loc_tuple
                else:
                     # Should not happen in valid states if locations are well-formed
                     return float('inf') # Indicate unsolvable/invalid state


        if robot_loc_tuple is None:
             # Robot location not found in state? Should not happen.
             return float('inf') # Indicate unsolvable/invalid state

        # 2. Initialize total heuristic cost
        h = 0
        boxes_not_at_goal = []

        # 3. + 4. Sum box distances to goals
        for box_name, goal_loc_tuple in self.goal_locations.items():
            current_loc_tuple = current_box_locations.get(box_name)

            if current_loc_tuple is None:
                 # Box not found in state? Should not happen in valid states.
                 return float('inf') # Indicate unsolvable/invalid state

            if current_loc_tuple != goal_loc_tuple:
                # Distance from box to goal
                box_distances_from_current = self.distances.get(current_loc_tuple)
                if box_distances_from_current is None:
                     # Box current location not in graph? Should not happen.
                     return float('inf') # Indicate unsolvable/invalid state

                d_box = box_distances_from_current.get(goal_loc_tuple, float('inf'))

                if d_box == float('inf'):
                    # Goal is unreachable for this box in the graph. State is likely unsolvable.
                    return float('inf')

                h += d_box
                boxes_not_at_goal.append(box_name)

        # 5. Add robot distance to the closest adjacent cell of any box not at goal
        if boxes_not_at_goal:
            min_robot_dist_to_any_adjacent = float('inf')
            robot_reachable_locations = self.distances.get(robot_loc_tuple)

            if robot_reachable_locations is None:
                # Robot current location not in graph? Should not happen.
                return float('inf') # Indicate unsolvable/invalid state

            found_adjacent_reachable = False
            for box_name in boxes_not_at_goal:
                current_loc_tuple = current_box_locations[box_name]
                # Iterate through all neighbors of the box's current location
                # self.adj is loc_tuple -> {direction: neighbor_loc_tuple}
                for adjacent_loc_tuple in self.adj.get(current_loc_tuple, {}).values():
                     d_robot_to_adjacent = robot_reachable_locations.get(adjacent_loc_tuple, float('inf'))
                     if d_robot_to_adjacent != float('inf'):
                         min_robot_dist_to_any_adjacent = min(min_robot_dist_to_any_adjacent, d_robot_to_adjacent)
                         found_adjacent_reachable = True

            if found_adjacent_reachable:
                h += min_robot_dist_to_any_adjacent
            else:
                # Robot cannot reach any location adjacent to any box that needs moving.
                # This state is likely unsolvable.
                return float('inf')

        # 6. Return total cost
        return h
