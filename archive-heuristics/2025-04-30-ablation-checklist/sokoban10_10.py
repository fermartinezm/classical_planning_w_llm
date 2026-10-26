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
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start, goal, blocked_locations=None):
    """
    Performs a Breadth-First Search to find the shortest path distance
    between a start and a goal location in a graph, avoiding blocked locations.

    Args:
        graph (dict): Adjacency list representation of the graph.
                      {location: [adjacent_location1, adjacent_location2, ...]}
        start (str): The starting location.
        goal (str): The target location.
        blocked_locations (set, optional): A set of locations that cannot be traversed.
                                           Defaults to None.

    Returns:
        int: The shortest path distance, or float('inf') if the goal is unreachable
             or start/goal are blocked (unless start == goal).
    """
    if blocked_locations is None:
        blocked_locations = set()

    # If start or goal are blocked (and not the same location), they are unreachable
    if start != goal and (start in blocked_locations or goal in blocked_locations):
        return float('inf')
        
    # If start is not in the graph, it's isolated or invalid
    if start not in graph and start != goal:
         return float('inf')

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == goal:
            return dist

        # Get neighbors from the graph. Handle locations not in the graph (e.g., walls).
        neighbors = graph.get(current_loc, [])

        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in blocked_locations:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    # Goal is unreachable
    return float('inf')


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the
    minimum number of pushes required for each misplaced box to reach its
    goal location, plus the minimum number of robot moves required to reach
    the nearest misplaced box.

    # Assumptions:
    - The goal specifies the final location for each box.
    - The grid structure is defined by `adjacent` predicates.
    - The heuristic assumes boxes can be pushed along shortest paths on the
      grid graph, ignoring complex deadlocks or the need for complex robot
      repositioning maneuvers beyond reaching the box initially.
    - Robot movement is blocked by boxes.

    # Heuristic Initialization
    - Extracts goal locations for each box.
    - Builds the grid graph from `adjacent` static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes.
    2. Identify which boxes are not currently at their specified goal location (misplaced boxes).
    3. If no boxes are misplaced, the heuristic value is 0.
    4. Build a graph representing the grid connectivity from the `adjacent` facts.
    5. For each misplaced box:
       a. Calculate the shortest path distance from the box's current location to its goal location on the grid graph (ignoring other boxes/robot as obstacles for this box-centric distance). This estimates the minimum number of pushes needed for this box.
       b. Sum these distances for all misplaced boxes. This is the 'box_push_distance'.
    6. Determine locations currently occupied by boxes. These locations are blocked for robot movement.
    7. Calculate the shortest path distance from the robot's current location to the current location of *each* misplaced box, considering box locations as obstacles for the robot.
    8. Find the minimum of these robot-to-box distances. This estimates the robot's cost to reach the nearest box it needs to push. This is the 'robot_approach_distance'.
    9. The heuristic value is the sum of the 'box_push_distance' and the 'robot_approach_distance'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box. Assuming a single goal location per box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box_name, location = args
                self.goal_locations[box_name] = location

        # Build the grid graph from adjacent facts.
        # The graph is an adjacency list: {location: [adjacent_location1, ...]}
        self.grid_graph = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.grid_graph[loc1].append(loc2)
                # Assuming adjacency is symmetric if not explicitly stated both ways,
                # but the example PDDL lists both directions, so we rely on that.
                # If the PDDL didn't list both, we might add the reverse here:
                # self.grid_graph[loc2].append(loc1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        if robot_loc is None:
             # Should not happen in a valid Sokoban state, but handle defensively
             return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "box*", "*"):
                box_name, loc = get_parts(fact)[1:]
                box_locations[box_name] = loc

        # Identify misplaced boxes
        misplaced_boxes = []
        for box_name, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box_name) # Use .get in case a box is missing from state facts (shouldn't happen)
            if current_loc is not None and current_loc != goal_loc:
                misplaced_boxes.append(box_name)

        # If all boxes are at their goals, the heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate sum of box-to-goal distances (minimum pushes)
        box_push_distance_sum = 0
        for box_name in misplaced_boxes:
            current_box_loc = box_locations[box_name]
            goal_box_loc = self.goal_locations[box_name]

            # Calculate shortest path for the box on the grid (ignoring obstacles for the box itself)
            # This is an optimistic estimate of pushes needed.
            dist = bfs(self.grid_graph, current_box_loc, goal_box_loc)

            if dist == float('inf'):
                # If any box is unreachable from its goal, the state is likely a dead end
                return float('inf')
            box_push_distance_sum += dist

        # Calculate minimum robot distance to reach any misplaced box
        min_robot_to_box_dist = float('inf')

        # Determine locations blocked for robot movement (locations occupied by boxes)
        blocked_for_robot = set(box_locations.values())

        for box_name in misplaced_boxes:
            current_box_loc = box_locations[box_name]
            # Calculate shortest path for the robot to the box location, avoiding other boxes
            dist = bfs(self.grid_graph, robot_loc, current_box_loc, blocked_for_robot)
            min_robot_to_box_dist = min(min_robot_to_box_dist, dist)

        # If the robot cannot reach any misplaced box, the state is likely a dead end
        if min_robot_to_box_dist == float('inf'):
             return float('inf')

        # The heuristic is the sum of box movement costs and the robot's cost to get started
        # This is non-admissible as it doesn't account for robot repositioning between pushes
        # or complex interactions/blockages between boxes and walls.
        return box_push_distance_sum + min_robot_to_box_dist

