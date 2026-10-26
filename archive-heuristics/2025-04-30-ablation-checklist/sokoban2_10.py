import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of shortest path distances for each misplaced box to its goal location
       (assuming boxes can move freely on the static grid). This estimates the minimum
       number of pushes required.
    2. The minimum shortest path distance for the robot to reach any location
       adjacent to any misplaced box, provided that adjacent location is clear.
       This estimates the cost for the robot to get into a position to start
       pushing a box.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Shortest paths are calculated using BFS on this grid.
    - Box movement distance is calculated on the static grid (ignoring dynamic obstacles).
    - Robot movement distance is calculated on the grid, avoiding locations occupied by boxes.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Build the graph of locations based on `adjacent` facts.
    - Store the goal locations for each box.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Parse the current state to find the robot's location, box locations, and clear locations.
    2. Identify which boxes are not currently at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Calculate the 'box cost': For each misplaced box, find the shortest path distance
       from its current location to its goal location on the static grid graph. Sum these distances.
       This estimates the minimum number of pushes needed for each box independently.
    5. Calculate the 'robot cost': Find the minimum shortest path distance from the robot's
       current location to any location `adj_loc` such that `adj_loc` is adjacent to
       *any* misplaced box and `adj_loc` is currently clear. The robot's path must avoid
       locations occupied by boxes. This estimates the cost for the robot to get into
       a position to interact with a box.
    6. The total heuristic value is the sum of the 'box cost' and the 'robot cost'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (`adjacent` relationships).
        - Build the location graph from `adjacent` facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                obj_type = None
                # Need to determine if the object is a box.
                # PDDL type info isn't directly in task.goals or static,
                # but we can infer from predicate 'at ?o - box ?l - location'
                # or assume objects in 'at' goals are boxes in Sokoban.
                # Let's assume objects in 'at' goals are boxes for this domain.
                package, location = args
                self.goal_locations[package] = location

        # Build the graph of locations based on adjacent facts.
        # Graph is an adjacency dictionary: {location: [adjacent_location, ...]}
        self.location_graph = self._build_graph(static_facts)

    def _build_graph(self, static_facts):
        """Builds the location graph from adjacent facts."""
        graph = collections.defaultdict(list)
        locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                graph[loc1].append(loc2)
                graph[loc2].append(loc1) # Assuming adjacency is symmetric for movement
                locations.add(loc1)
                locations.add(loc2)

        # Ensure all mentioned locations are keys in the graph, even if they have no adjacencies listed
        # (though in Sokoban they usually do). This handles potential isolated locations.
        for loc in locations:
             if loc not in graph:
                 graph[loc] = []

        return dict(graph) # Convert defaultdict to dict

    def _shortest_path(self, start, end, graph, obstacles=None):
        """
        Performs BFS to find the shortest path distance between two locations.

        Args:
            start (str): The starting location.
            end (str): The target location.
            graph (dict): The location graph {location: [adjacent_locations]}.
            obstacles (set, optional): A set of locations that cannot be visited. Defaults to None.

        Returns:
            int: The shortest path distance, or float('inf') if the end is unreachable.
        """
        if obstacles is None:
            obstacles = set()

        if start == end:
            return 0

        queue = collections.deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Check if current_loc is an obstacle (shouldn't happen if start isn't obstacle,
            # but good practice, especially if start could be in obstacles)
            if current_loc in obstacles:
                 continue

            for neighbor in graph.get(current_loc, []):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # End is unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current locations of robot and boxes, and clear locations
        robot_loc = None
        box_locations = {} # {box_name: location}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at":
                box, loc = parts[1:]
                box_locations[box] = loc
            elif parts[0] == "clear":
                clear_locations.add(parts[1])

        # Identify misplaced boxes
        misplaced_boxes = [
            box for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        ]

        # If all relevant boxes are at their goals, heuristic is 0
        if not misplaced_boxes:
            return 0

        total_cost = 0

        # --- Component 1: Box distances to goals (minimum pushes) ---
        # Calculate sum of shortest paths for each box to its goal on the static grid.
        # We ignore dynamic obstacles (other boxes, robot) for this part, assuming
        # an optimistic path for the box itself.
        box_dist_sum = 0
        for box in misplaced_boxes:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]
            dist = self._shortest_path(current_loc, goal_loc, self.location_graph, obstacles=set()) # No dynamic obstacles for box path
            if dist == float('inf'):
                 # If a box goal is unreachable on the static graph, this state is likely a dead end
                 # or requires moving other boxes that are static obstacles in this view.
                 # A large penalty or infinity is appropriate.
                 return float('inf') # Or a large number like 1000

            box_dist_sum += dist

        total_cost += box_dist_sum

        # --- Component 2: Robot distance to a push position ---
        # Find the minimum distance from the robot to any clear location adjacent
        # to any misplaced box.
        min_robot_dist = float('inf')
        robot_obstacles = set(box_locations.values()) # Robot cannot move into a cell occupied by a box

        for box in misplaced_boxes:
            loc_b = box_locations[box]
            # Find locations adjacent to the box's current location
            adjacent_to_box = self.location_graph.get(loc_b, [])

            for adj_loc in adjacent_to_box:
                # A potential push position must be clear
                if adj_loc in clear_locations:
                    # Calculate robot distance to this potential push position, avoiding boxes
                    dist = self._shortest_path(robot_loc, adj_loc, self.location_graph, obstacles=robot_obstacles)
                    min_robot_dist = min(min_robot_dist, dist)

        # If the robot cannot reach any clear adjacent location to any box,
        # it might be stuck or requires complex maneuvers.
        # Add this minimum distance to the total cost.
        # If min_robot_dist is inf, it means the robot cannot reach *any*
        # clear adjacent cell to *any* misplaced box. This state is likely
        # unsolvable or requires significant path clearing.
        if min_robot_dist == float('inf'):
             return float('inf') # Or a large number

        total_cost += min_robot_dist

        return total_cost

