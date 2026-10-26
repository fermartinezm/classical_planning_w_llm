from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For infinity

# Helper function to extract the components of a PDDL fact
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse location strings like "loc_row_col"
def parse_location_str(loc_str):
    """Parses a location string 'loc_row_col' into a (row, col) tuple."""
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers if necessary
            return None
    return None # Return None for non-standard location strings

# Helper function to build the graph from adjacent facts
def build_graph(static_facts):
    """
    Builds an adjacency list graph from 'adjacent' static facts.
    Nodes are (row, col) tuples.
    """
    graph = {}
    for fact in static_facts:
        if match(fact, "adjacent", "*", "*", "*"):
            _, loc1_str, loc2_str, _ = get_parts(fact)
            loc1_tuple = parse_location_str(loc1_str)
            loc2_tuple = parse_location_str(loc2_str)
            if loc1_tuple and loc2_tuple:
                if loc1_tuple not in graph:
                    graph[loc1_tuple] = set()
                graph[loc1_tuple].add(loc2_tuple)
                # Assuming adjacency is symmetric, add the reverse edge
                if loc2_tuple not in graph:
                    graph[loc2_tuple] = set()
                graph[loc2_tuple].add(loc1_tuple)
    return graph

# Helper function to precompute all-pairs shortest paths using BFS
def precompute_distances(graph):
    """
    Computes shortest path distances between all pairs of nodes in the graph
    using BFS from each node.
    Returns a dictionary: ((start_node), (end_node)) -> distance.
    """
    distances = {}
    nodes = list(graph.keys())

    for start_node in nodes:
        q = deque([(start_node, 0)])
        visited = {start_node}
        distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            if current_node in graph: # Ensure the node exists in the graph keys
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The shortest path distance (on the static grid) for each box to its goal location.
    2. The shortest path distance (on the static grid) from the robot's current location
       to the location of the closest box that is not yet at its goal.
    3. A small constant cost (e.g., 1) for each box that needs to be moved,
       representing the initial effort to engage with the box.

    # Assumptions
    - The grid structure and adjacencies are static and defined by 'adjacent' facts.
    - Shortest path distances on the static grid provide a reasonable estimate,
      ignoring dynamic obstacles (other boxes, robot) and the push mechanic constraints
      during path calculation.
    - The number of boxes equals the number of goal locations, and each box has a unique goal.
      (The PDDL examples support this assumption).
    - All locations mentioned in the problem instance (initial state, goals, adjacencies)
      are part of a single connected component in the static graph.

    # Heuristic Initialization
    - Parses 'adjacent' facts from static information to build a graph of locations.
    - Precomputes shortest path distances between all pairs of locations on this static graph using BFS.
    - Extracts and stores the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box from the state.
    3. Determine which boxes are not yet at their goal locations by comparing current
       box locations with the stored goal locations.
    4. If the set of boxes not at their goals is empty, the heuristic value is 0
       (the goal state has been reached).
    5. Initialize a variable `box_distance_sum` to 0.
    6. Iterate through each box that is not at its goal:
       - Get the current location tuple of the box and its goal location tuple.
       - Look up the precomputed shortest path distance between these two location tuples
         in the `self.distances` dictionary.
       - If the distance is not found (implying locations are disconnected in the static graph,
         or an invalid state), return a very large value to indicate a likely unsolvable
         or very difficult state.
       - Add the found distance to `box_distance_sum`.
    7. Find the box (among those not at their goals) whose current location tuple is closest
       to the robot's current location tuple, using the precomputed shortest path distances.
       Initialize `min_robot_dist` to infinity and `closest_box_loc_tuple` to None.
       Iterate through boxes not at goal, get their location, look up the distance
       from the robot's location, and update `min_robot_dist` and `closest_box_loc_tuple`
       if a shorter distance is found.
    8. If no closest box location was found (e.g., no boxes need moving, or robot location
       is disconnected from all box locations), return a large value if there are still
       boxes not at goal (this case should ideally not happen in solvable problems).
       Otherwise, set `robot_distance_to_closest_box` to `min_robot_dist`.
    9. The total heuristic value is calculated as `box_distance_sum + robot_distance_to_closest_box + num_boxes_not_at_goal`.
       The `num_boxes_not_at_goal` term adds a base cost for each box that still requires
       manipulation, serving as a simple proxy for the initial effort (robot movement
       and first push) required for each box.
    10. Return the calculated total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the graph, and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the graph of locations from adjacent facts
        self.graph = build_graph(static_facts)

        # Precompute shortest path distances between all pairs of locations
        self.distances = precompute_distances(self.graph)

        # Store goal locations for each box.
        self.box_goals = {}
        # Also collect all box names
        self.all_boxes = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at ?b ?l)
                box_name, goal_loc_str = args
                self.box_goals[box_name] = parse_location_str(goal_loc_str)
                self.all_boxes.add(box_name)

        # If there are boxes in the initial state not mentioned in goals,
        # add them to self.all_boxes. This is defensive programming.
        # However, standard PDDL requires all objects to be declared and used.
        # We can rely on self.all_boxes from goals for typical Sokoban problems.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_loc_str = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc_str = get_parts(fact)[1]
                break
        robot_loc_tuple = parse_location_str(robot_loc_str)

        # Find box locations
        box_locations_str = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj_name, loc_str = get_parts(fact)
                 # Check if the object is one of the boxes we care about
                 if obj_name in self.all_boxes:
                     box_locations_str[obj_name] = loc_str

        box_locations_tuple = {
            box: parse_location_str(loc_str)
            for box, loc_str in box_locations_str.items()
        }

        # Identify boxes not at their goal
        boxes_not_at_goal = [
            box for box in self.all_boxes
            if box_locations_tuple.get(box) != self.box_goals.get(box)
        ]

        # If all boxes are at goal, heuristic is 0
        if not boxes_not_at_goal:
            return 0

        # Calculate sum of box-to-goal distances
        box_distance_sum = 0
        for box in boxes_not_at_goal:
            current_loc = box_locations_tuple[box]
            goal_loc = self.box_goals[box]

            # Ensure both locations are valid and in the graph
            if current_loc is None or goal_loc is None or current_loc not in self.graph or goal_loc not in self.graph:
                 # This state involves locations not in the precomputed graph, likely invalid or unsolvable
                 return 1000000 # Large value

            dist = self.distances.get((current_loc, goal_loc))

            if dist is None:
                 # If a box's current location and goal location are unreachable from each other
                 return 1000000 # Large value for unreachable goals

            box_distance_sum += dist

        # Find the closest box not at goal to the robot
        closest_box_loc_tuple = None
        min_robot_dist = math.inf

        # Ensure robot location is valid and in the graph
        if robot_loc_tuple is None or robot_loc_tuple not in self.graph:
             return 1000000 # Large value if robot is in an invalid location

        for box in boxes_not_at_goal:
            box_loc = box_locations_tuple[box]
            # Ensure box location is valid and in the graph
            if box_loc is None or box_loc not in self.graph:
                 return 1000000 # Large value if a box is in an invalid location

            dist = self.distances.get((robot_loc_tuple, box_loc))
            if dist is not None and dist < min_robot_dist:
                 min_robot_dist = dist
                 closest_box_loc_tuple = box_loc

        # If robot cannot reach any box needing move (should not happen in solvable puzzles on connected grid)
        if closest_box_loc_tuple is None or min_robot_dist == math.inf:
             # This implies robot cannot reach any box that needs moving
             return 1000000 # Large value

        robot_distance_to_closest_box = min_robot_dist

        # Total heuristic value
        # Sum of box distances + robot distance to closest box + base cost per box
        total_h = box_distance_sum + robot_distance_to_closest_box + len(boxes_not_at_goal)

        return total_h
