from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
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

def shortest_path_distance(start_loc, end_loc, graph):
    """
    Computes the shortest path distance between two locations using BFS.
    Assumes the graph is an adjacency dictionary {location: [adjacent_locations]}.
    Returns float('inf') if no path exists.
    """
    if start_loc == end_loc:
        return 0

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Ensure current_loc is a key in the graph dictionary and has neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor == end_loc:
                    return dist + 1 # Found the target
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # If current_loc is not in graph or has no neighbors, BFS stops exploring from here

    return math.inf # No path found

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It sums the shortest path distance for each
    misplaced box to its goal and adds the shortest path distance from the
    robot to the nearest misplaced box.

    # Assumptions
    - The grid structure and adjacency are defined by the 'adjacent' static facts.
    - Boxes must reach specific goal locations defined in the task goals.
    - The heuristic uses shortest path distances on the grid graph, ignoring
      obstacles (other boxes, walls not defined by lack of adjacency) during
      distance calculations for simplicity and efficiency.
    - Deadlock states are not explicitly detected or penalized, potentially
      leading to misleading heuristic values in such cases.
    - The cost of moving the robot to a box and pushing it is simplified.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task goals.
    - Builds an undirected adjacency graph of the locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify the goal location for each box from the initialized goal mapping.
    3. Filter the boxes to find those that are currently not at their goal location (misplaced boxes).
    4. If there are no misplaced boxes, the state is a goal state, and the heuristic is 0.
    5. If there are misplaced boxes:
       a. Calculate the sum of shortest path distances for each misplaced box from its current location to its goal location using the pre-built grid graph. This estimates the minimum number of pushes required for all boxes collectively, ignoring robot position and other boxes.
       b. Calculate the shortest path distance from the robot's current location to the location of the *nearest* misplaced box. This estimates the cost for the robot to engage with a box that needs moving.
       c. The total heuristic value is the sum of the total box-to-goal distance and the minimum robot-to_box distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the grid graph.
        """
        # Assuming task object has 'goals' (frozenset of goal facts) and 'static' (frozenset of static facts)
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency graph from static facts
        # The graph is represented as a dictionary: location -> list of adjacent locations
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1:]
                # Add bidirectional edges
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1)

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                # Assuming goal predicates are always (at box location)
                # and each box has only one goal location specified in goals.
                self.box_goals[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # State is a frozenset of fact strings

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        if robot_loc is None:
             # Robot location must be known in a valid state
             return math.inf

        # Find box locations
        box_locs = {}
        # Collect all 'at' facts first
        at_facts = [get_parts(fact) for fact in state if match(fact, "at", "*", "*")]

        for parts in at_facts:
            obj, loc = parts[1:]
            # Check if the object is one of the boxes we care about (i.e., has a goal)
            if obj in self.box_goals:
                 box_locs[obj] = loc

        # Identify misplaced boxes
        misplaced_boxes = []
        for box in self.box_goals.keys():
             current_loc = box_locs.get(box)
             goal_loc = self.box_goals[box]
             # A box is misplaced if its current location is known and is not the goal location
             # Or if its current location is unknown (shouldn't happen in valid states)
             if current_loc is None or current_loc != goal_loc:
                  misplaced_boxes.append(box)


        # If all boxes are at their goals, the heuristic is 0.
        if not misplaced_boxes:
            return 0

        total_box_distance = 0
        min_robot_to_box_distance = math.inf

        for box in misplaced_boxes:
            current_box_loc = box_locs.get(box) # Use .get in case a box is missing from state (shouldn't happen)
            goal_box_loc = self.box_goals[box]

            if current_box_loc is None:
                 # A box is missing from the state facts - indicates an invalid state representation
                 return math.inf

            # Calculate distance for the box to its goal
            # This is the minimum number of pushes needed for this box in an empty grid
            box_to_goal_dist = shortest_path_distance(current_box_loc, goal_box_loc, self.graph)

            # If any box cannot reach its goal location in the grid graph, it's likely unsolvable
            if box_to_goal_dist == math.inf:
                 return math.inf

            total_box_distance += box_to_goal_dist

            # Calculate distance from robot to this box's current location
            # This is the minimum number of moves for the robot to reach the box
            robot_to_this_box_dist = shortest_path_distance(robot_loc, current_box_loc, self.graph)

            # Update minimum robot distance to any *reachable* misplaced box
            # If robot_to_this_box_dist is inf, it won't update min_robot_to_box_distance unless it was already inf
            min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_this_box_dist)


        # If the robot cannot reach *any* of the misplaced boxes, the state is likely unsolvable
        if min_robot_to_box_distance == math.inf:
             return math.inf

        # The heuristic is the sum of box-to-goal distances (pushes)
        # plus the minimum robot-to-box distance (robot movement to get to a box).
        # This is a simple sum of estimated costs for the two main types of "work".
        return total_box_distance + min_robot_to_box_distance
