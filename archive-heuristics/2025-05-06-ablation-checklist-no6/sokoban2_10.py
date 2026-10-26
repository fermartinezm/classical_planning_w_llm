from heuristics.heuristic_base import Heuristic

import collections
from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_location_graph(static_facts):
    """
    Builds a graph where nodes are locations and edges represent adjacency.
    Edges are bidirectional.
    """
    graph = collections.defaultdict(set)
    for fact in static_facts:
        if match(fact, "adjacent", "*", "*", "*"):
            _, loc1, loc2, _ = get_parts(fact)
            graph[loc1].add(loc2)
            graph[loc2].add(loc1) # Assuming adjacency is symmetric
    return graph

def shortest_path_distance(graph, start, end, obstacles=None):
    """
    Calculates the shortest path distance between start and end locations
    on the graph using BFS, avoiding obstacle locations.
    Returns infinity if no path exists.
    """
    if obstacles is None:
        obstacles = set()

    # Cannot start or end inside an obstacle (unless start == end and start is not an obstacle)
    if start in obstacles:
         return float('inf')
    if end in obstacles and start != end:
         return float('inf')
    if start == end:
        return 0

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        for neighbor in graph.get(current_loc, []):
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    Estimates the cost as the sum of shortest path distances for each box
    to its goal, plus the shortest path distance for the robot to the
    closest box that needs moving.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Shortest path distance on the grid is a reasonable estimate for movement cost.
    - Moving a box one step requires at least one push action.
    - The robot must reach a box to push it.

    # Heuristic Initialization
    - Build the location graph from static `adjacent` facts.
    - Extract goal locations for each box from the task goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Identify the goal location for each box (precomputed in init).
    4. Calculate the sum of shortest path distances for each box that is not
       at its goal location. The shortest path is calculated on the grid graph,
       ignoring other boxes as obstacles for this step (as they can be moved).
       If any box cannot reach its goal on the grid, the heuristic is infinity.
    5. If all boxes are at their goals, the heuristic is 0.
    6. If there are boxes not at their goals, calculate the shortest path distance
       from the robot's current location to the location of each box that needs
       moving. The robot's path must avoid locations occupied by *other* boxes.
    7. Find the minimum of these robot-to-box distances.
    8. If the robot cannot reach any box that needs moving, the heuristic is infinity.
    9. The total heuristic value is the sum from step 4 plus the minimum distance
       from step 7.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - The location graph from adjacent facts.
        """
        super().__init__(task) # Initialize the base class

        # Build the location graph from static adjacent facts
        self.graph = build_location_graph(self.static)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at box_name location_name)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of facts)

        # Find current robot location
        loc_robot = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                loc_robot = get_parts(fact)[1]
                break
        # Should always find robot location in a valid state, but handle defensively
        if loc_robot is None:
             # This should not happen in a well-formed problem/state
             return float('inf') # Indicate unsolvable state

        # Find current box locations
        box_locations = {}
        for fact in state:
            # Check for facts like (at box1 loc_...)
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                 box, loc = parts[1:]
                 box_locations[box] = loc

        total_heuristic = 0
        boxes_not_at_goal = []
        locations_of_all_boxes = set(box_locations.values())

        # Calculate sum of box-goal distances
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # Get current location, handle if box not found (shouldn't happen)
            if current_loc is None:
                 # This box is in the goal list but not in the state? Problematic state.
                 return float('inf')

            if current_loc != goal_loc:
                boxes_not_at_goal.append(box)
                # Distance for the box to reach its goal (ignoring other boxes as obstacles for this path)
                # A box can only move to a clear location. However, for a simple non-admissible heuristic,
                # we calculate distance on the full grid graph.
                dist_box_goal = shortest_path_distance(self.graph, current_loc, goal_loc, obstacles=set())
                if dist_box_goal == float('inf'):
                    # Box cannot reach its goal location on the grid
                    return float('inf')
                total_heuristic += dist_box_goal

        # If all boxes are at their goals, the state is a goal state
        if not boxes_not_at_goal:
            return 0

        # Calculate minimum robot distance to a box that needs moving
        min_robot_dist_to_box = float('inf')
        for box in boxes_not_at_goal:
            loc_b = box_locations[box]
            # Obstacles for the robot are the locations of *other* boxes
            obstacles_for_robot = locations_of_all_boxes - {loc_b}
            # Robot needs to reach a location adjacent to loc_b to push it.
            # Calculating distance to loc_b itself is a simple approximation.
            dist_robot_to_b = shortest_path_distance(self.graph, loc_robot, loc_b, obstacles=obstacles_for_robot)
            min_robot_dist_to_box = min(min_robot_dist_to_box, dist_robot_to_b)

        # If robot cannot reach any box that needs moving
        if min_robot_dist_to_box == float('inf'):
             return float('inf')

        total_heuristic += min_robot_dist_to_box

        return total_heuristic
