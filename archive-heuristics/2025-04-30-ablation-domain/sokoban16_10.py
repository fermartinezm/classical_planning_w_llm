# Helper functions
from fnmatch import fnmatch
from collections import deque # For BFS
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # Ensure the number of parts is at least the number of args provided
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location and adding the shortest
    path distance from the robot to the closest box that is not yet at its goal.
    This heuristic is non-admissible and designed for greedy best-first search.

    # Assumptions
    - The locations form a grid graph defined by 'adjacent' predicates.
    - Shortest path distances on this grid graph are a reasonable estimate
      of movement costs for both boxes (pushes) and the robot (moves).
    - The heuristic does not consider obstacles (other boxes, walls) blocking
      paths, nor does it consider the specific robot positioning required for pushes.
    - All locations relevant to the problem (initial positions, goal positions,
      and intermediate path locations) are connected within the graph defined
      by 'adjacent' predicates.

    # Heuristic Initialization
    - Extracts all location objects mentioned in 'adjacent' predicates.
    - Builds an undirected graph representing the grid connectivity based on
      'adjacent' predicates.
    - Computes all-pairs shortest paths between all locations using BFS.
    - Stores the goal location for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box from the state.
    3. Initialize the total heuristic value to 0.
    4. Create a list to keep track of boxes that are not yet at their goal locations.
    5. Iterate through each box:
       - Get the box's current location and its goal location (from initialization).
       - If the box has a goal location and its current location is different from the goal:
         - Calculate the shortest path distance from the box's current location
           to its goal location using the precomputed distances.
         - Add this distance to the total heuristic value. This component
           represents a lower bound on the number of pushes required for this box
           in isolation.
         - Add the box to the list of boxes not at goal.
       - If the box's current location or goal location is not found in the
         precomputed distances (indicating an isolated or unreachable area),
         return infinity as the state is likely unsolvable.
    6. If the list of boxes not at goal is empty, it means all boxes are at their goals,
       so the total heuristic is 0. Return 0.
    7. If there are boxes not at goal, find the minimum shortest path distance
       from the robot's current location to the current location of any box
       in the list of boxes not at goal.
    8. If the robot's location or the closest box's location is not found in the
       precomputed distances, return infinity.
    9. Add this minimum robot-to-box distance to the total heuristic value. This
       component estimates the robot's effort to reach a box that needs pushing.
    10. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting locations, building the graph,
        computing distances, and storing goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract all location objects mentioned in adjacent facts
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) >= 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])

        # Build the undirected graph from adjacent facts
        self.graph = {loc: [] for loc in self.locations} # Initialize graph with all found locations

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) >= 3:
                l1, l2 = parts[1], parts[2]
                # Add edges in both directions for undirected graph
                if l2 not in self.graph[l1]:
                     self.graph[l1].append(l2)
                if l1 not in self.graph[l2]:
                     self.graph[l2].append(l1)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                current_node, dist = queue.popleft()
                self.distances[start_node][current_node] = dist
                for neighbor in self.graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Store goal locations for each box
        self.box_goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Check if the goal is an 'at' predicate for a 'box' object
            if predicate == "at" and args and len(args) >= 2 and args[0].startswith("box"):
                box, location = args[0], args[1]
                self.box_goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        # If robot location is not found in the state, this is an invalid state
        if robot_loc is None:
             return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("box"):
                 box, loc = get_parts(fact)[1], get_parts(fact)[2]
                 box_locations[box] = loc

        total_heuristic = 0
        boxes_not_at_goal = []

        # Calculate sum of box-to-goal distances
        for box, current_loc in box_locations.items():
            goal_loc = self.box_goal_locations.get(box)
            # Ensure goal exists and box is not already there
            if goal_loc and current_loc != goal_loc:
                # Check if current_loc and goal_loc are in our distance map
                # If not, it indicates an isolated or unreachable area
                if current_loc in self.distances and goal_loc in self.distances.get(current_loc, {}):
                    total_heuristic += self.distances[current_loc][goal_loc]
                    boxes_not_at_goal.append(box)
                else:
                    # Unreachable goal or box - likely unsolvable
                    return float('inf')

        # If all boxes are at goals, heuristic is 0
        if not boxes_not_at_goal:
            return 0

        # Calculate robot-to-closest-box distance (only for boxes not at goal)
        min_robot_box_dist = float('inf')
        robot_reachable = robot_loc in self.distances

        if robot_reachable:
            for box in boxes_not_at_goal:
                box_loc = box_locations[box]
                # Check if box_loc is reachable from robot_loc
                if box_loc in self.distances.get(robot_loc, {}):
                     min_robot_box_dist = min(min_robot_box_dist, self.distances[robot_loc][box_loc])
                else:
                     # Box is in an isolated area not reachable by robot - likely unsolvable
                     return float('inf')
        else:
             # Robot is in an isolated area - likely unsolvable
             return float('inf')


        # Add the minimum robot distance to the total heuristic
        # This check is redundant if the loops above handle unreachable locations correctly,
        # but kept for safety. If min_robot_box_dist is still inf, it means no box_loc
        # was reachable from robot_loc among boxes_not_at_goal.
        if min_robot_box_dist != float('inf'):
             total_heuristic += min_robot_box_dist
        else:
             # Should be caught by the loops above, but return inf if somehow missed
             return float('inf')

        return total_heuristic
