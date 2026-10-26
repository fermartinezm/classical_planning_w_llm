import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distance(graph, start, end, obstacles=None):
    """
    Performs a Breadth-First Search on the location graph to find the shortest
    path distance between a start and end location, optionally avoiding obstacles.

    Args:
        graph (dict): Adjacency list representation of the location graph.
                      {location: [neighbor1, neighbor2, ...]}
        start (str): The starting location.
        end (str): The target location.
        obstacles (set, optional): A set of locations that cannot be visited.
                                   Defaults to None.

    Returns:
        int: The shortest distance from start to end, or float('inf') if no path exists.
    """
    if obstacles is None:
        obstacles = set()

    if start == end:
        return 0
    if start in obstacles:
        return float('inf') # Cannot start in an obstacle

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # No path found

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distance
    for each misplaced box to its goal location and the shortest path distance
    for the robot to reach a location adjacent to any misplaced box (avoiding
    other boxes).

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - The goal is to move specific boxes to specific locations.
    - The robot's final position is not constrained by the goal.

    # Heuristic Initialization
    - Builds a graph representation of the locations based on `adjacent` facts.
    - Stores the goal location for each box.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes.
    2. Identify which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Calculate the sum of shortest path distances for each misplaced box
       from its current location to its goal location. This estimates the
       minimum number of pushes needed for the boxes themselves, ignoring
       robot movement and obstacles. Use BFS on the location graph, ignoring
       obstacles (as boxes can potentially be moved).
    5. Calculate the minimum shortest path distance from the robot's current
       location to *any* location that is adjacent to *any* of the misplaced
       boxes. This estimates the cost for the robot to get into a position
       to start interacting with a box. Use BFS on the location graph, treating
       locations occupied by *other* boxes as obstacles for the robot.
    6. The total heuristic value is the sum of the total box-to-goal distance
       and the minimum robot-to-box-vicinity distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The location graph from `adjacent` facts.
        - Goal locations for each box.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts.
        # The graph is an adjacency list where keys are locations and values
        # are lists of directly reachable neighboring locations.
        self.graph = collections.defaultdict(list)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                # Add edges in both directions as movement is bidirectional
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Assuming adjacency is symmetric

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot and box locations in the current state.
        robot_loc = None
        box_locations = {} # {box_name: location}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1] in self.goal_locations: # Only track boxes relevant to goals
                 box_locations[parts[1]] = parts[2]

        # Identify boxes that are not at their goal locations.
        misplaced_boxes = {
            box for box, loc in box_locations.items()
            if self.goal_locations.get(box) != loc # Use .get for safety, though task should define goals for all boxes
        }

        # If all relevant boxes are at their goals, the heuristic is 0.
        if not misplaced_boxes:
            return 0

        # --- Calculate Box-to-Goal Distance Sum ---
        # Sum of shortest path distances for each misplaced box to its goal.
        # We calculate this distance on the full location graph, ignoring
        # other objects, as they can potentially be moved out of the way.
        box_distance_sum = 0
        for box in misplaced_boxes:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]
            dist = bfs_distance(self.graph, current_loc, goal_loc)

            # If a box cannot reach its goal even on an empty grid, it's likely trapped.
            if dist == float('inf'):
                return float('inf') # Return infinity if any box is trapped

            box_distance_sum += dist

        # --- Calculate Robot Access Distance ---
        # Minimum distance from the robot to a location adjacent to any misplaced box.
        # The robot needs to reach a location next to a box to push it.
        # We treat locations occupied by *other* boxes as obstacles for the robot.
        robot_obstacles = set(box_locations.values()) - {robot_loc} # Locations occupied by boxes other than the robot's current location

        min_robot_access_distance = float('inf')

        # Consider all locations adjacent to any misplaced box
        potential_robot_targets = set()
        for box in misplaced_boxes:
            box_loc = box_locations[box]
            if box_loc in self.graph:
                 # Add all neighbors of the box's location as potential robot targets
                 potential_robot_targets.update(self.graph[box_loc])

        # Calculate distance from robot to the nearest potential target location
        for target_loc in potential_robot_targets:
             # Robot cannot move into a location occupied by another box
             if target_loc not in robot_obstacles:
                 dist = bfs_distance(self.graph, robot_loc, target_loc, robot_obstacles)
                 min_robot_access_distance = min(min_robot_access_distance, dist)

        # If the robot cannot reach any location adjacent to any misplaced box,
        # the state might be unsolvable or very difficult.
        if min_robot_access_distance == float('inf'):
             return float('inf')

        # --- Combine Distances ---
        # The heuristic is the sum of box distances and the robot's access distance.
        # This is non-admissible as it doesn't account for the robot needing
        # to move between pushing different boxes or clearing paths.
        total_heuristic = box_distance_sum + min_robot_access_distance

        return total_heuristic

# Note: This code assumes the existence of a Heuristic base class and a Task class
# as provided in the problem description. The `bfs_distance` function is a helper
# function specific to this heuristic's needs.
