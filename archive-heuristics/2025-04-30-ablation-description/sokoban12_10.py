from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs(graph, start, end=None):
    """
    Performs BFS on the location graph to find shortest path distances.

    graph: adjacency list {location: [adjacent_locations]}
    start: starting location
    end: optional target location
    Returns: distance to end if end is specified, otherwise a dict of distances
             from start to all reachable locations. Returns float('inf') if end is unreachable.
    """
    queue = deque([(start, 0)])
    visited = {start}
    distances = {start: 0}

    while queue:
        current_loc, dist = queue.popleft()

        if end is not None and current_loc == end:
            return dist

        # Ensure current_loc exists in the graph keys before accessing neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

    if end is not None:
        return float('inf') # End not reachable
    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances of each box to its goal location and adding the shortest
    path distance from the robot to a location adjacent to the closest box
    that is not yet at its goal.

    # Assumptions
    - The locations form a graph defined by `adjacent` predicates.
    - Shortest path distance on this graph is a reasonable estimate for movement cost.
    - The cost of moving a box is approximated by the shortest path distance
      on the location graph (each step is one push).
    - The cost for the robot to get into a position to push a box is approximated
      by the shortest path distance from the robot to any location adjacent to
      the box.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph representation of the locations based on the
      `adjacent` static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box from the state.
    3. Identify the goal location for each box (pre-calculated during initialization).
    4. Check if all boxes are at their goal locations. If yes, the heuristic is 0.
    5. For each box that is not at its goal location:
       a. Calculate the shortest path distance from the box's current location
          to its goal location using BFS on the location graph. Sum these distances.
          This represents the minimum number of pushes required for the boxes
          if they could be pushed freely.
    6. Find the set of boxes that are not at their goal locations.
    7. If there are such boxes, find the box among them that is closest to the robot
       (using shortest path distance on the location graph). Let this be the `closest_box`
       at `closest_box_loc`.
    8. Find all locations `adj_loc` that are directly adjacent to `closest_box_loc`
       using the pre-built location graph.
    9. Calculate the minimum shortest path distance from the robot's current location
       to any of these `adj_loc`s. This estimates the cost for the robot to get into
       a position adjacent to the closest box, ready to potentially push it.
    10. The total heuristic value is the sum of the box-to-goal distances (step 5a)
        and this minimum robot-to-adjacent-location distance (step 9).
    11. If any required distance calculation results in infinity (meaning a location
        is unreachable), the heuristic should return infinity to indicate an
        unsolvable state (or a state from which the goal is unreachable).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically '(at box location)'
            if match(goal, "at", "*", "*"):
                box, location = get_parts(goal)[1:]
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        # The graph is represented as an adjacency list.
        self.location_graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                # Add edges in both directions as movement is bidirectional
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        if robot_location is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Robot location unknown

        # Find current box locations and identify boxes not at goal
        total_box_distance = 0
        boxes_not_at_goal = []
        current_box_locations_map = {} # Map box name to current location string

        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj_name, current_loc = get_parts(fact)[1:]
                  # Check if the object is one of the boxes we have a goal for
                  if obj_name in self.goal_locations:
                       current_box_locations_map[obj_name] = current_loc
                       goal_loc = self.goal_locations[obj_name]
                       if current_loc != goal_loc:
                           boxes_not_at_goal.append(obj_name)
                           # Calculate shortest path from box's current location to its goal
                           dist = bfs(self.location_graph, current_loc, goal_loc)
                           if dist == float('inf'):
                               # If any box goal is unreachable, the state is likely unsolvable
                               return float('inf')
                           total_box_distance += dist

        # If there are no boxes not at their goal, the goal is reached
        if not boxes_not_at_goal:
             return 0

        # Calculate distance from robot to a push position for the closest box not at goal
        min_robot_to_push_position_distance = float('inf')

        # Need robot distances from robot_location to all reachable locations
        robot_distances = bfs(self.location_graph, robot_location)

        # Find the closest box location (in terms of robot distance) among those not at goal
        min_dist_robot_to_any_box = float('inf')
        closest_box_loc = None

        for box_name in boxes_not_at_goal:
             box_loc = current_box_locations_map[box_name]
             # Check if the box location is reachable by the robot
             if box_loc in robot_distances:
                  if robot_distances[box_loc] < min_dist_robot_to_any_box:
                       min_dist_robot_to_any_box = robot_distances[box_loc]
                       closest_box_loc = box_loc

        # If the robot cannot reach any box that needs moving, the state is likely unsolvable
        if closest_box_loc is None:
             return float('inf')

        # Now calculate the minimum distance from the robot to any location adjacent to the closest box
        # This is the estimated cost for the robot to get into a position to push the box
        if closest_box_loc in self.location_graph:
             for adj_loc in self.location_graph[closest_box_loc]:
                  # Check if the adjacent location is reachable by the robot
                  if adj_loc in robot_distances:
                       min_robot_to_push_position_distance = min(min_robot_to_push_position_distance, robot_distances[adj_loc])

        # If the robot cannot reach any location adjacent to the closest box, the state is likely unsolvable
        if min_robot_to_push_position_distance == float('inf'):
             return float('inf')


        # The heuristic is the sum of box distances and the robot distance to a push position for the closest box
        heuristic_value = total_box_distance + min_robot_to_push_position_distance

        return heuristic_value
