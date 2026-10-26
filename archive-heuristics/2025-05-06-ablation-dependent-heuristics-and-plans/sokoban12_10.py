# Assuming heuristic_base.py exists and defines a Heuristic base class
# Example:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError

# Define a dummy Heuristic base class if the actual one is not provided
# This is just for standalone testing or if the base class is simple
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError


from fnmatch import fnmatch
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string before stripping
    if not isinstance(fact, str):
        return [] # Or raise an error, depending on expected input
    return fact.strip('()').split()

# Map directions to their opposites
OPPOSITE_DIRECTIONS = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total number of actions (robot moves and pushes)
    required to move all boxes to their goal locations. It sums the estimated cost
    for each box that is not yet at its goal. The cost for a single box is estimated
    as the minimum number of pushes required to move it to its goal on an empty grid,
    plus the minimum number of robot moves required to reach a position from which
    it can perform the first necessary push in the current state.

    # Assumptions
    - The grid structure and connectivity are defined solely by the 'adjacent' facts.
    - Goal conditions only involve boxes being at specific locations.
    - The robot can only move into 'clear' locations or its current location.
    - Boxes can only be moved by the robot using the 'push' action.
    - The heuristic assumes a box can always be pushed towards its goal on an empty grid,
      unless no adjacent location is closer to the goal on the empty grid.
    - If the robot cannot reach any valid push position for a box in the current state,
      a large penalty is applied for that box.

    # Heuristic Initialization
    - Parses goal conditions to map each box to its target location.
    - Builds an adjacency graph and a reverse adjacency graph from 'adjacent' facts
      to represent the grid connectivity.
    - Identifies all possible location objects mentioned in static facts, goals, or initial state.
    - Precomputes shortest path distances between all pairs of locations on the empty
      grid graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location.
    2. Identify the current location of each box (only those that are goal objects).
    3. Determine the set of 'clear' locations in the current state.
    4. Define the set of locations the robot can move through: clear locations plus
       the robot's current location, excluding locations occupied by boxes.
    5. Initialize the total heuristic value to 0.
    6. For each box that is listed in the goals and is not currently at its goal location:
       a. Get the box's current location (`loc_b`) and its goal location (`loc_g`).
       b. Calculate the shortest path distance from `loc_b` to `loc_g` on the empty
          grid graph (`dist_box_to_goal`). This is the minimum number of pushes needed
          in an ideal scenario.
       c. Find potential next locations (`loc_next`) for the box: these are neighbors
          of `loc_b` such that the empty grid distance from `loc_next` to `loc_g` is
          strictly less than `dist_box_to_goal`.
       d. If no such `loc_next` exists (the box is stuck relative to the goal on the
          empty grid), add a large penalty (`UNREACHABLE_PENALTY`) to the total
          heuristic and proceed to the next box.
       e. For each potential `loc_next`, determine the required robot push position
          (`loc_r_push`). This is the location adjacent to `loc_b` in the same
          direction as the move from `loc_b` to `loc_next`.
       f. Identify the set of valid `loc_r_push` candidates: these are the required
          push positions that are included in the robot's currently allowed locations.
       g. If no valid `loc_r_push` candidate exists (the robot cannot reach any
          position to push the box towards the goal in the current state), add a
          large penalty (`UNREACHABLE_PENALTY`) to the total heuristic and proceed
          to the next box.
       h. Calculate the minimum shortest path distance from the robot's current
          location to any of the valid `loc_r_push` candidates, using BFS on the
          graph of robot-allowed locations in the current state (`min_robot_dist`).
       i. If `min_robot_dist` == float('inf') (robot cannot reach any valid push position
          from its current location), add a large penalty (`UNREACHABLE_PENALTY`).
       j. Otherwise, add `dist_box_to_goal + min_robot_dist` to the total heuristic value.
    7. Return the total heuristic value.
    """

    UNREACHABLE_PENALTY = 1000 # Penalty for boxes that are stuck or unreachable

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the graph."""
        # super().__init__(task) # Assuming Heuristic base class has __init__(self, task)

        self.task = task # Store task if base class doesn't handle it

        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        # Extract goal locations for boxes
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        self.adjacency_graph = {}
        self.reverse_adjacency_graph = {}
        self.locations = set()

        # Build adjacency graphs from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.adjacency_graph:
                    self.adjacency_graph[loc1] = {}
                if loc2 not in self.reverse_adjacency_graph:
                    self.reverse_adjacency_graph[loc2] = {}
                self.adjacency_graph[loc1][loc2] = direction
                # Build reverse graph: maps loc2 -> {loc1: dir} if (adjacent loc1 loc2 dir)
                # This means loc1 is adjacent to loc2 in direction dir.
                # If box moves loc_b -> loc_next in dir, robot needs to be at loc_r_push
                # such that (adjacent loc_r_push loc_b dir) is true.
                # This means loc_r_push is a neighbor of loc_b in the reverse graph with direction 'dir'.
                self.reverse_adjacency_graph[loc2][loc1] = direction


        # Ensure all locations mentioned in goals or initial state are in self.locations
        # This handles cases where some locations might be isolated or only mentioned
        # in initial/goal states but not in any 'adjacent' facts.
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "at" and len(parts) == 3:
                 self.locations.add(parts[2]) # goal location
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at-robot" and len(parts) == 2:
                 self.locations.add(parts[1]) # robot location
             elif parts and parts[0] == "at" and len(parts) == 3:
                 self.locations.add(parts[2]) # box location
             elif parts and parts[0] == "clear" and len(parts) == 2:
                 self.locations.add(parts[1]) # clear location


        # Precompute all-pairs shortest paths on the empty grid
        self.empty_grid_distances = self._precompute_distances()

    def _precompute_distances(self):
        """Computes shortest path distances between all pairs of locations on the empty grid."""
        distances = {loc: {l: float('inf') for l in self.locations} for loc in self.locations}
        for loc in self.locations:
            distances[loc][loc] = 0
            queue = deque([loc])
            visited = {loc}

            while queue:
                curr = queue.popleft()
                # Iterate through neighbors using the adjacency graph
                if curr in self.adjacency_graph:
                    for neighbor in self.adjacency_graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[loc][neighbor] = distances[loc][curr] + 1
                            queue.append(neighbor)
        return distances

    def bfs_distance(self, start_loc, end_loc, allowed_locations):
        """
        Computes the shortest path distance between two locations using BFS,
        considering only moves to locations within allowed_locations.
        """
        if start_loc == end_loc:
            return 0
        # If start or end is not in the set of locations the robot can potentially traverse, it's unreachable
        # This check is important if allowed_locations is a strict subset of self.locations
        if start_loc not in allowed_locations or end_loc not in allowed_locations:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            curr, dist = queue.popleft()

            if curr in self.adjacency_graph:
                for neighbor in self.adjacency_graph[curr]:
                    # Robot can only move to allowed locations
                    if neighbor in allowed_locations and neighbor not in visited:
                        if neighbor == end_loc:
                            return dist + 1
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # End location is unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        robot_loc = None
        box_locations = {}
        clear_locations = set()
        occupied_locations = set() # Locations occupied by boxes

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif parts and parts[0] == "at" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Only track boxes that are goals
                if box in self.goal_locations:
                    box_locations[box] = loc
                    occupied_locations.add(loc)
            elif parts and parts[0] == "clear" and len(parts) == 2:
                clear_locations.add(parts[1])

        # Locations the robot can move through (clear locations + robot's current location)
        # Robot cannot move *into* a location occupied by a box.
        # The set of locations the robot can *be* at or move *into* is clear_locations | {robot_loc}.
        # BFS should only traverse edges where the destination is in this set.
        robot_allowed_locations = clear_locations | {robot_loc}


        total_heuristic = 0

        # Iterate through boxes that need to reach a goal
        for box, goal_loc in self.goal_locations.items():
            loc_b = box_locations.get(box) # Get current box location

            # If box is not in state (shouldn't happen if state includes all objects)
            # or already at goal, continue
            if loc_b is None or loc_b == goal_loc:
                continue

            # 1. Box distance to goal on empty grid (minimum pushes)
            # Use .get() with default empty dict to handle cases where loc_b or goal_loc
            # might not be in self.locations (e.g., isolated locations)
            dist_box_to_goal = self.empty_grid_distances.get(loc_b, {}).get(goal_loc, float('inf'))

            if dist_box_to_goal == float('inf'):
                 # Box is unreachable from its goal on the empty grid - implies unsolvable
                 total_heuristic += self.UNREACHABLE_PENALTY
                 continue

            # 2. Robot distance to a push position for the first step
            min_robot_dist = float('inf')
            found_valid_push_pos_candidate = False

            # Find potential first steps for the box towards the goal
            # These are neighbors of loc_b that are strictly closer to goal_loc on the empty grid
            candidate_loc_nexts = []
            if loc_b in self.adjacency_graph:
                for loc_next in self.adjacency_graph[loc_b]:
                    # Check if moving to loc_next gets closer to the goal on the empty grid
                    # Use .get() with default empty dict for safety
                    if self.empty_grid_distances.get(loc_next, {}).get(goal_loc, float('inf')) < dist_box_to_goal:
                         candidate_loc_nexts.append(loc_next)

            if not candidate_loc_nexts:
                 # Box is stuck relative to goal on empty grid (e.g., cornered or surrounded by walls)
                 total_heuristic += self.UNREACHABLE_PENALTY
                 continue

            # For each potential first step, find the required robot push position
            valid_loc_r_push_candidates = set()
            for loc_next in candidate_loc_nexts:
                 # Find direction from loc_b to loc_next
                 direction = self.adjacency_graph[loc_b][loc_next]
                 # Find loc_r_push such that (adjacent loc_r_push loc_b direction) is true.
                 # This means loc_r_push is a neighbor of loc_b in the reverse graph with direction 'dir'.
                 if loc_b in self.reverse_adjacency_graph:
                     for potential_r_push, d in self.reverse_adjacency_graph[loc_b].items():
                         if d == direction:
                             # Check if this potential push position is reachable by the robot in the current state
                             # The push position itself must be a location the robot can be AT.
                             # This means it must be in the set of robot_allowed_locations.
                             if potential_r_push in robot_allowed_locations:
                                 valid_loc_r_push_candidates.add(potential_r_push)
                                 found_valid_push_pos_candidate = True


            if not found_valid_push_pos_candidate:
                 # Robot cannot reach *any* valid push position for *any* beneficial first step
                 total_heuristic += self.UNREACHABLE_PENALTY
                 continue

            # Calculate minimum robot distance to any valid push position
            # The BFS must consider the current state's allowed locations for robot movement
            for push_pos in valid_loc_r_push_candidates:
                 dist = self.bfs_distance(robot_loc, push_pos, robot_allowed_locations)
                 min_robot_dist = min(min_robot_dist, dist)

            if min_robot_dist == float('inf'):
                 # Robot cannot reach any valid push position from its current location
                 total_heuristic += self.UNREACHABLE_PENALTY
            else:
                 # Heuristic for this box: pushes needed + robot moves to get into position for the first push
                 total_heuristic += dist_box_to_goal + min_robot_dist

        return total_heuristic
