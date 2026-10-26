from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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

# Note: The bfs_distance function defined here is a general utility for graph distance.
# The main heuristic calculation in __call__ uses pre-computed distances on the full graph
# for efficiency, which is equivalent to BFS on the full graph ignoring obstacles.
# This function could be used for variations that consider obstacles dynamically.
def bfs_distance(graph, start, end, obstacles=None):
    """
    Computes the shortest path distance between start and end in the graph,
    optionally avoiding obstacle locations.
    Returns float('inf') if no path exists.
    """
    if obstacles is None:
        obstacles = set()

    if start == end:
        return 0
    # Cannot start inside an obstacle unless it's the target
    if start in obstacles and start != end:
        return float('inf')

    queue = deque([(start, 0)])
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


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its goal location, computed on the full location graph ignoring obstacles.
    2. The shortest path distance for the robot from its current location to
       the location of the box that is currently closest to the robot,
       also computed on the full location graph ignoring obstacles.

    # Assumptions
    - The location names are parseable as strings.
    - The graph defined by 'adjacent' facts is connected (at least the relevant parts).
    - The shortest path distance on the full graph is a reasonable lower bound
      on the number of pushes required for a box.
    - The robot needs to reach a box to push it, and the distance to the closest
      box is a relevant cost component.
    - The heuristic does not consider complex deadlocks (e.g., boxes trapped in corners).

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds the location graph from the 'adjacent' static facts. The graph is treated
      as undirected based on the PDDL structure providing adjacent facts for both
      directions (e.g., (adjacent l1 l2 dir) and (adjacent l2 l1 reverse_dir)).
    - Pre-computes all-pairs shortest path distances on the *full* location graph
      (ignoring obstacles) using BFS from every node. This pre-computed table is
      used for both the box-goal distance and the robot-box distance components
      in the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Calculate the first component (box-goal distance):
       - For each box not at its goal, find its current location L_b and its goal location G_b.
       - Use the pre-computed shortest path distance on the full graph (ignoring obstacles) between L_b and G_b.
       - Sum these distances for all boxes not at their goals. If any box cannot reach its goal
         even on the full graph, the state is likely unsolvable, and the heuristic returns infinity.
    5. Calculate the second component (robot-box distance):
       - Find the box among those not at their goals that is currently closest to the robot
         in terms of shortest path distance on the full graph (ignoring obstacles). Let this box be b_closest
         at location L_b_closest.
       - Use the pre-computed shortest path distance on the full graph (ignoring obstacles) between the robot's
         current location L_r and L_b_closest. If the robot cannot reach any box needing to move
         on the full graph, the state is likely unsolvable, and the heuristic returns infinity.
    6. The total heuristic value is the sum of the box-goal component and the robot-box component.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract box goals
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is (at boxX loc_Y_Z)
                box_name, goal_loc = args
                self.box_goals[box_name] = goal_loc

        # Build the location graph from adjacent facts
        self.graph = {}
        self.all_locations = set()
        for fact in static_facts:
             if match(fact, "adjacent", "*", "*", "*"):
                 _, loc1, loc2, _ = get_parts(fact)
                 self.all_locations.add(loc1)
                 self.all_locations.add(loc2)
                 if loc1 not in self.graph:
                     self.graph[loc1] = []
                 if loc2 not in self.graph:
                     self.graph[loc2] = []
                 # Add both directions explicitly as per PDDL structure
                 self.graph[loc1].append(loc2)


        # Pre-compute all-pairs shortest paths on the full graph (ignoring obstacles)
        self.full_graph_distances = {}
        for start_node in self.all_locations:
            # Run BFS from start_node to find distances to all other nodes
            queue = deque([(start_node, 0)])
            visited = {start_node}
            distances = {start_node: 0}

            while queue:
                current_loc, dist = queue.popleft()
                self.full_graph_distances[(start_node, current_loc)] = dist

                if current_loc in self.graph:
                    for neighbor in self.graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

            # Store infinity for locations not reachable from start_node
            for other_node in self.all_locations:
                 if (start_node, other_node) not in self.full_graph_distances:
                     self.full_graph_distances[(start_node, other_node)] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot and box locations
        robot_loc = None
        box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1] in self.box_goals: # Only track boxes we care about
                box_name = parts[1]
                box_loc = parts[2]
                box_locations[box_name] = box_loc

        # Identify boxes not at goals
        boxes_to_move = [
            b for b, l in box_locations.items()
            if self.box_goals.get(b) != l
        ]

        # If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        # Component 1: Sum of box-goal distances (relaxed)
        box_goal_h = 0
        for box_name in boxes_to_move:
            current_loc = box_locations[box_name]
            goal_loc = self.box_goals[box_name]
            dist = self.full_graph_distances.get((current_loc, goal_loc), float('inf'))
            if dist == float('inf'):
                 # If a box cannot reach its goal even in the full graph, it's unsolvable
                 return float('inf') # Return infinity for unsolvable states
            box_goal_h += dist

        # Component 2: Robot-box distance (relaxed)
        # Find the box closest to the robot (on the full graph) among those needing to move
        closest_box_name = None
        min_dist_to_box = float('inf')

        for box_name in boxes_to_move:
            box_loc = box_locations[box_name]
            dist = self.full_graph_distances.get((robot_loc, box_loc), float('inf'))
            if dist < min_dist_to_box:
                min_dist_to_box = dist
                closest_box_name = box_name

        # If robot cannot reach any box needing to move, it's unsolvable
        if closest_box_name is None or min_dist_to_box == float('inf'):
             return float('inf') # Return infinity for unsolvable states

        robot_box_h = min_dist_to_box

        # Total heuristic is the sum
        return box_goal_h + robot_box_h
