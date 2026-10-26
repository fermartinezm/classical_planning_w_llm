from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# If not provided, define a dummy for testing purposes
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError("Heuristic must implement __call__")

# Helper functions (can be outside the class or static methods)

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input gracefully
        return []
    return fact[1:-1].split()

def parse_location(loc_str):
    """Parses a location string like 'loc_x_y' into (x, y) coordinates."""
    if not isinstance(loc_str, str):
        return None
    try:
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            # PDDL locations are 1-indexed, convert to 0-indexed for easier calculation
            return (int(parts[1]) - 1, int(parts[2]) - 1)
        # print(f"Warning: Unexpected location string format: {loc_str}")
        return None
    except (ValueError, IndexError):
        # print(f"Warning: Could not parse location string: {loc_str}")
        return None

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (x, y) coordinate tuples."""
    if coords1 is None or coords2 is None:
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

def bfs_distance(start_loc, target_loc, blocked_locs, adj):
    """
    Calculates the shortest path distance from start_loc to target_loc
    avoiding locations in blocked_locs, using the adjacency graph.
    """
    if start_loc == target_loc:
        return 0
    # Robot cannot start a path *from* a location occupied by another box
    # (though it can start where it is).
    if start_loc in blocked_locs:
         return float('inf')

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == target_loc:
            return dist

        # Iterate through neighbors from the adjacency list
        # The second element of the tuple from adj is direction, which we don't need for BFS pathfinding
        for neighbor_loc, _ in adj.get(current_loc, []):
            # Robot cannot move into a location occupied by a box
            if neighbor_loc not in visited and neighbor_loc not in blocked_locs:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return float('inf') # Target not reachable

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum, for each box not at its goal, of:
    (shortest robot path to a position where it can push the box towards the goal)
    + (Manhattan distance from the box's current location to its goal location).
    Includes a penalty if a box seems stuck or a push position is unreachable.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Adjacency graph (adj, rev_adj) from static facts.
        - Location coordinates for Manhattan distance calculation.
        """
        super().__init__(task)

        self.goal_locations = {}
        self.adj = {} # location -> [(neighbor_loc, direction), ...]
        self.rev_adj = {} # location -> [(prev_loc, direction), ...]
        self.location_coords = {} # location_str -> (x, y)
        all_locations = set()

        # Process static facts to build graph and collect locations
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 4 and parts[0] == 'adjacent':
                l1, l2, d = parts[1], parts[2], parts[3]
                self.adj.setdefault(l1, []).append((l2, d))
                # Store reverse adjacency for finding push positions
                self.rev_adj.setdefault(l2, []).append((l1, d))
                all_locations.add(l1)
                all_locations.add(l2)

        # Process goal facts to find box goal locations
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal is typically (at boxN loc_x_y)
            if len(parts) == 3 and parts[0] == 'at' and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc
                all_locations.add(loc) # Add goal locations to known locations

        # Process initial state facts to ensure all locations are known
        # This is important if some locations only appear in the initial state
        # and not in static or goals (unlikely but safe)
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 2 and parts[0] == 'at-robot':
                  all_locations.add(parts[1])
             elif len(parts) == 3 and parts[0] == 'at' and parts[1].startswith('box'):
                  all_locations.add(parts[2])
             elif len(parts) == 2 and parts[0] == 'clear':
                  all_locations.add(parts[1])

        # Parse coordinates for all known locations
        for loc_str in all_locations:
             coords = parse_location(loc_str)
             if coords is not None:
                 self.location_coords[loc_str] = coords
             # else: location string format is unexpected, ignore or log warning

    def manhattan_distance(self, loc1_str, loc2_str):
        """Calculates Manhattan distance using pre-parsed coordinates."""
        coords1 = self.location_coords.get(loc1_str)
        coords2 = self.location_coords.get(loc2_str)
        # Return infinity if coordinates weren't parsed (e.g., invalid loc string)
        return manhattan_distance(coords1, coords2)

    def bfs_distance(self, start_loc, target_loc, blocked_locs):
         """
         Calculates the shortest path distance for the robot.
         Uses the pre-built adjacency graph.
         """
         return bfs_distance(start_loc, target_loc, blocked_locs, self.adj)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify robot and box locations
        robot_loc = None
        box_locs = {} # box -> location
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif len(parts) == 3 and parts[0] == 'at' and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                box_locs[box] = loc

        if robot_loc is None:
             # Robot location should always be in a valid state
             # print("Error: Robot location not found in state.")
             return float('inf') # Indicate invalid state

        # Locations occupied by boxes (robot cannot move into these)
        robot_blocked_locs = set(box_locs.values())

        total_h = 0
        stuck_penalty = 1000 # Penalty for boxes that seem stuck or unreachable

        # 2. Calculate heuristic for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locs.get(box)

            # If box is not in the state (shouldn't happen in valid states) or already at goal, ignore
            if current_loc is None or current_loc == goal_loc:
                continue

            # Box is not at its goal. Estimate cost for this box.
            # Box movement cost: Manhattan distance (minimum pushes if path is clear)
            box_dist = self.manhattan_distance(current_loc, goal_loc)

            # Find candidate robot push locations
            # A location P is a candidate if adjacent(P, current_loc, dir)
            # and adjacent(current_loc, next_loc, dir) for some dir,
            # AND next_loc is strictly closer to goal_loc than current_loc (Manhattan distance).
            push_pos_candidates = set()
            current_loc_coords = self.location_coords.get(current_loc)
            goal_loc_coords = self.location_coords.get(goal_loc)

            # If coordinates are missing for current or goal location, cannot calculate distance
            if current_loc_coords is None or goal_loc_coords is None:
                 total_h += stuck_penalty
                 continue

            # Iterate through directions the box *could* be pushed from current_loc
            # using the forward adjacency list
            for next_loc, dir in self.adj.get(current_loc, []):
                 next_loc_coords = self.location_coords.get(next_loc)
                 if next_loc_coords is not None:
                      # Check if pushing in this direction moves the box strictly closer to the goal
                      if manhattan_distance(next_loc_coords, goal_loc_coords) < box_dist:
                           # If so, find the location prev_loc such that adjacent(prev_loc, current_loc, dir)
                           # This prev_loc is the required robot position to push from current_loc to next_loc
                           # Use the reverse adjacency list to find prev_loc
                           for prev_loc, prev_dir in self.rev_adj.get(current_loc, []):
                                if prev_dir == dir: # Check if the reverse direction matches the push direction
                                     push_pos_candidates.add(prev_loc)
                                     break # Found the unique prev_loc for this dir

            # If no push position allows moving towards the goal, this box might be stuck
            if not push_pos_candidates:
                 total_h += stuck_penalty
                 continue # Move to the next box

            # Calculate minimum robot distance to any candidate push position
            min_robot_dist = float('inf')
            for candidate_loc in push_pos_candidates:
                 # BFS calculates path distance for the robot, avoiding locations occupied by *other* boxes.
                 # The robot's current location is the start, so it's handled correctly by BFS.
                 dist = self.bfs_distance(robot_loc, candidate_loc, robot_blocked_locs)
                 min_robot_dist = min(min_robot_dist, dist)

            # If robot cannot reach any push position, add penalty
            if min_robot_dist == float('inf'):
                 total_h += stuck_penalty
            else:
                 # Heuristic for this box:
                 # Cost to get robot to a valid push position + minimum pushes for the box
                 total_h += min_robot_dist + box_dist

        return total_h

