from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def _bfs(start_loc, target_locs, graph, allowed_locs=None):
    """
    Performs BFS to find the shortest distance from start_loc to any target_loc.

    Args:
        start_loc: The starting location.
        target_locs: A set of target locations.
        graph: The location graph {loc: {dir: adj_loc, ...}} representing directed edges.
        allowed_locs: Optional set of locations the path is allowed to traverse.
                      If None, all locations in the graph are allowed.

    Returns:
        The shortest distance, or float('inf') if no path exists.
    """
    if start_loc in target_locs:
        return 0

    # If allowed_locs is provided, start_loc must be in it to begin the search
    if allowed_locs is not None and start_loc not in allowed_locs:
         return float('inf') # Cannot start from here if not allowed

    queue = collections.deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Check neighbors based on the directed graph
        if current_loc in graph:
            for neighbor_loc in graph[current_loc].values():
                # Check if neighbor is allowed (if filter is active)
                if allowed_locs is not None and neighbor_loc not in allowed_locs:
                    continue

                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    if neighbor_loc in target_locs:
                        return dist + 1
                    queue.append((neighbor_loc, dist + 1))

    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing two components:
    1. The minimum number of pushes required for each box to reach its goal location,
       calculated as the shortest path distance on the location graph ignoring obstacles.
    2. The minimum number of robot moves required to reach a location adjacent to
       any off-goal box, considering current obstacles (other boxes, walls).

    # Assumptions
    - The location graph defined by 'adjacent' predicates is static and represents
      directed movement possibilities.
    - Boxes need to reach specific goal locations.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic assumes that moving a box one step towards its goal is always
      beneficial and doesn't explicitly model complex interactions or deadlocks,
      except by returning a large value if a box cannot reach its goal location
      even in an empty grid, or if the robot cannot reach any box.

    # Heuristic Initialization
    - Parses the goal conditions to map each box to its target location.
    - Builds a graph representation of the locations based on 'adjacent' predicates.
      This graph is directed as per the PDDL definition.
    - Pre-computes all-pairs shortest path distances between all locations on this
      static directed graph (ignoring obstacles) for efficient box-goal distance lookups.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is the goal state. A state is a goal state if
       all boxes are at their specified goal locations. If it is the goal, return 0.
    2. Identify the robot's current location and the current location of each box
       from the state facts.
    3. Calculate the total minimum pushes needed for all off-goal boxes:
       - For each box that is not currently at its goal location:
         - Look up the pre-computed shortest path distance from its current location
           to its goal location on the static directed location graph (ignoring obstacles).
         - If no path exists (distance is infinity), this box cannot reach its goal
           even in an empty grid. This state is likely unsolvable or requires complex
           preparatory steps not captured by simple distance. Return a large penalty value.
         - Add this distance to a running total for box pushes. This sum represents
           a lower bound on the total number of 'push' actions needed.
    4. Calculate the minimum robot moves needed to get into a position to push *any* off-goal box:
       - Identify all locations that are directly adjacent (as per the graph) to any
         box that is not yet at its goal. These are potential target locations for the robot
         to reach in order to be next to a box.
       - Perform a Breadth-First Search (BFS) from the robot's current location
         to find the shortest path to any of these potential target locations.
       - This BFS for robot movement must only traverse locations that are currently
         marked as 'clear' in the state, or the robot's current location itself
          (as the robot can start its movement from its current spot). Locations
         occupied by boxes are not 'clear' and act as obstacles for robot movement.
       - If the robot cannot reach any location adjacent to an off-goal box (BFS
         returns infinity), return a large penalty value, as no box can be pushed.
    5. The final heuristic value is the sum of the total minimum box pushes (step 3)
       and the minimum robot moves required to reach a position adjacent to an
       off-goal box (step 4). This sum estimates the combined effort for boxes
       to reach goals and the robot to get into position to help.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)

        # Map boxes to their goal locations
        self.box_goals = {}
        for goal in task.goals:
            # Goal facts are like (at box1 loc_2_4)
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box, goal_loc = parts[1], parts[2]
                self.box_goals[box] = goal_loc

        # Build the location graph from adjacent facts (directed)
        self.location_graph = collections.defaultdict(dict)
        self.all_locations = set()
        for fact in task.static:
            # Adjacent facts are like (adjacent loc_1_1 loc_1_2 right)
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.location_graph[loc1][direction] = loc2
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Pre-compute all-pairs shortest path distances on the directed location graph
        self.location_distances = {}
        for start_loc in list(self.all_locations):
            # BFS from start_loc to all other locations on the static graph
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            self.location_distances[(start_loc, start_loc)] = 0

            bfs_queue = collections.deque([(start_loc, 0)])
            bfs_visited = {start_loc}

            while bfs_queue:
                current_loc, dist = bfs_queue.popleft()
                self.location_distances[(start_loc, current_loc)] = dist

                if current_loc in self.location_graph:
                    # Iterate over adjacent locations reachable from current_loc
                    for neighbor_loc in self.location_graph[current_loc].values():
                         if neighbor_loc not in bfs_visited:
                             bfs_visited.add(neighbor_loc)
                             bfs_queue.append((neighbor_loc, dist + 1))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        LARGE_PENALTY = 1_000_000 # Use a large integer penalty

        # 1. Check if goal state
        is_goal = True
        for box, goal_loc in self.box_goals.items():
            if f"(at {box} {goal_loc})" not in state:
                is_goal = False
                break
        if is_goal:
            return 0

        # 2. Identify robot and box locations
        robot_loc = None
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                obj_name, obj_loc = parts[1], parts[2]
                # Check if the object is one of the boxes we care about
                if obj_name in self.box_goals:
                    current_box_locations[obj_name] = obj_loc

        # Should not happen in valid states, but as a safeguard
        if robot_loc is None:
             return LARGE_PENALTY

        # 3. Calculate total minimum pushes for off-goal boxes
        total_box_distance = 0
        off_goal_boxes_present = False
        robot_target_locations = set() # Locations adjacent to off-goal boxes

        for box, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box)
            if current_loc is None:
                 # Box is not in the state? Problematic state.
                 return LARGE_PENALTY

            if current_loc != goal_loc:
                off_goal_boxes_present = True
                box_dist = self.location_distances.get((current_loc, goal_loc), float('inf'))

                if box_dist == float('inf'):
                    # Box cannot reach goal even on empty grid
                    return LARGE_PENALTY # Large penalty

                total_box_distance += box_dist

                # Add locations adjacent to this off-goal box as potential robot targets
                if current_loc in self.location_graph:
                    for adj_loc in self.location_graph[current_loc].values():
                         robot_target_locations.add(adj_loc)

        # If no off-goal boxes, heuristic is 0 (already handled by is_goal check)
        if not off_goal_boxes_present:
             return 0 # Should be caught by is_goal check, but safe fallback

        # 4. Calculate minimum robot moves to reach a pushable position
        # Robot can move to any 'clear' location or its current location.
        # It cannot move into locations occupied by boxes (as they are not clear).
        robot_allowed_locations = {loc for loc in self.all_locations if f"(clear {loc})" in state}
        robot_allowed_locations.add(robot_loc) # Robot can start from its current spot

        min_robot_distance_to_push = _bfs(robot_loc, robot_target_locations, self.location_graph, robot_allowed_locations)

        if min_robot_distance_to_push == float('inf'):
            # Robot cannot reach any location adjacent to an off-goal box
            return LARGE_PENALTY # Large penalty

        # 5. Total heuristic value
        # The heuristic is the sum of the estimated box pushes and the robot's effort
        # to get to a position where it can start pushing.
        return total_box_distance + min_robot_distance_to_push

