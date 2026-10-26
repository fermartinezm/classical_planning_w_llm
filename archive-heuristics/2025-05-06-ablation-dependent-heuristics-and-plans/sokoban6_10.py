from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts (Included based on example heuristics, though not strictly used in the logic below)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed). The number of args must match the number of elements in the fact.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box not at its goal:
    1. The shortest path distance (in pushes) for the box to reach its goal location, ignoring obstacles for the box itself.
    2. The shortest path distance (in robot moves) for the robot to reach *any* location adjacent to the box, considering current obstacles (other boxes and non-clear locations).

    # Assumptions
    - Each box has a unique goal location specified in the problem.
    - The grid structure implied by location names (loc_row_col) is not strictly necessary; we rely only on the explicit 'adjacent' facts to build the graph.
    - Robot moves and box pushes have a cost of 1.

    # Heuristic Initialization
    - Build a graph representation of the locations and their adjacencies from the static facts.
    - Store the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box that has a goal.
    3. Identify the set of clear locations.
    4. Initialize total heuristic cost to 0.
    5. For each box that has a goal:
        a. Check if the box is already at its goal location. If yes, continue to the next box.
        b. If not at the goal, get the box's current location (l_b) and its goal location (g_b).
        c. Calculate the shortest path distance from l_b to g_b in the location graph (ignoring obstacles for the box). This is the minimum number of pushes required for this box. Add this distance to the total cost.
        d. Calculate the shortest path distance from the robot's current location (l_r) to *any* location l_adj that is adjacent to l_b. This BFS must only traverse locations that are currently *clear*. This estimates the cost for the robot to reach a position from which it *could* potentially push the box. Add this minimum distance to the total cost.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - The location graph from 'adjacent' facts.
        """
        # Assuming task object has 'goals' and 'static' attributes as shown in example
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        # Graph: {location: {neighbor: direction, ...}, ...}
        self.location_graph = {}
        # Also store reverse adjacencies for finding locations adjacent *to* a box
        self.location_graph_reverse = {}

        for fact in static_facts:
            parts = get_parts(fact)
            # Check if the fact is an 'adjacent' predicate with 3 arguments after the predicate name
            if len(parts) == 4 and parts[0] == "adjacent":
                l1, l2, direction = parts[1], parts[2], parts[3]
                if l1 not in self.location_graph:
                    self.location_graph[l1] = {}
                self.location_graph[l1][l2] = direction

                # Add reverse adjacency: l1 is adjacent *to* l2
                if l2 not in self.location_graph_reverse:
                    self.location_graph_reverse[l2] = {}
                self.location_graph_reverse[l2][l1] = direction # Store direction from l1 to l2

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Check if the goal is an 'at' predicate with 2 arguments after the predicate name
            if len(args) == 2 and predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def bfs_distance(self, start_loc, end_loc, graph):
        """
        Performs BFS on the given graph to find the shortest path distance
        between start_loc and end_loc.
        Returns distance or float('inf') if unreachable.
        """
        if start_loc == end_loc:
            return 0

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # Check if current_loc is in the graph (handle potential inconsistencies)
            if current_loc not in graph:
                 continue

            for neighbor in graph.get(current_loc, {}): # Use .get() for safety
                if neighbor == end_loc:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # Target not reachable

    def bfs_min_distance_to_any(self, start_loc, target_locs, graph, traversable_locations):
        """
        Performs BFS on the given graph to find the shortest path distance
        from start_loc to any location in the target_locs set.
        Only traverses locations present in the traversable_locations set.
        Returns minimum distance or float('inf') if no target is reachable.
        """
        # If start_loc is one of the targets, distance is 0.
        if start_loc in target_locs:
            return 0

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # Check if current_loc is in the graph (handle potential inconsistencies)
            if current_loc not in graph:
                 continue

            for neighbor in graph.get(current_loc, {}): # Use .get() for safety
                 # Robot can only move into a neighbor if that neighbor is clear
                if neighbor not in traversable_locations:
                    continue

                if neighbor in target_locs:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # No target reachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Extract current robot and box locations and clear locations from the state
        robot_location = None
        box_locations = {} # {box_name: location}
        clear_locations = set() # Locations that are currently clear

        for fact in state:
            parts = get_parts(fact)
            # Check predicate name and number of arguments
            if len(parts) == 2 and parts[0] == "at-robot":
                robot_location = parts[1]
            elif len(parts) == 3 and parts[0] == "at":
                 obj, loc = parts[1], parts[2]
                 # Only track boxes that have goals
                 if obj in self.goal_locations:
                    box_locations[obj] = loc
            elif len(parts) == 2 and parts[0] == "clear":
                 clear_locations.add(parts[1])

        # If robot_location is not found (shouldn't happen in a valid state), return inf
        if robot_location is None:
             return float('inf')

        total_heuristic_cost = 0

        # The set of locations the robot can move *into*.
        # These are the locations currently marked as clear.
        robot_traversable_locations = clear_locations

        for box, goal_location in self.goal_locations.items():
            current_box_location = box_locations.get(box)

            # If box is not in the state or is already at its goal, skip it
            if current_box_location is None or current_box_location == goal_location:
                continue

            # 1. Estimate pushes for the box: BFS on the full location graph
            # This finds the minimum number of steps (pushes) for the box itself
            # to reach the goal, ignoring any obstacles or robot position.
            box_dist = self.bfs_distance(current_box_location, goal_location, self.location_graph)

            if box_dist == float('inf'):
                 # Box cannot reach goal location in the static graph structure
                 # This state is likely unsolvable or a dead end.
                 return float('inf')

            total_heuristic_cost += box_dist

            # 2. Estimate robot moves to reach a pushing position for the box
            # The robot needs to reach a location adjacent *to* the box's current location.
            # Find all locations l_adj such that adjacent(l_adj, current_box_location, some_dir).
            # These are the neighbors of current_box_location in the reverse graph.
            locations_adjacent_to_box = set(self.location_graph_reverse.get(current_box_location, {}).keys())

            if not locations_adjacent_to_box:
                 # No locations are adjacent *to* the box's current location.
                 # This implies a problem with the graph or state, likely unsolvable.
                 return float('inf')

            # BFS for robot from its current location to any location in locations_adjacent_to_box.
            # Robot can only move into clear locations.
            robot_dist = self.bfs_min_distance_to_any(robot_location, locations_adjacent_to_box, self.location_graph, robot_traversable_locations)

            if robot_dist == float('inf'):
                 # Robot cannot reach any location adjacent to the box's current location
                 # This state is likely unsolvable or a dead end.
                 return float('inf')

            total_heuristic_cost += robot_dist

        # If the loop finishes, all boxes are at their goals, and the cost is 0.
        # If any box/robot path was unreachable, total_heuristic_cost is inf.
        return total_heuristic_cost
