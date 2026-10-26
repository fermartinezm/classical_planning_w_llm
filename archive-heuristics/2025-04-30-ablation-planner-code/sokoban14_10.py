from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distance(graph, start, end):
    """
    Computes the shortest path distance between two locations in the graph using BFS.
    Returns float('inf') if the end is unreachable from the start.
    """
    if start == end:
        return 0
    queue = deque([(start, 0)]) # Stores tuples of (location, distance)
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        # Ensure current_loc exists in graph keys before iterating neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Target not reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box not at its goal,
    the shortest path distance from the box's current location to its goal location,
    and the shortest path distance from the robot's current location to the box's current location.
    The distances are computed on the location graph defined by 'adjacent' predicates. This heuristic
    is non-admissible as it ignores obstacles and the specific robot positioning required for pushes.

    # Assumptions
    - The location graph defined by 'adjacent' predicates is undirected (even though PDDL lists directions).
    - The shortest path distance in the location graph is a simplified estimate for movement cost, ignoring obstacles (other boxes, walls implicitly defined by lack of 'clear' or 'adjacent').
    - The heuristic is non-admissible.
    - Goal conditions are only of the form (at boxX locY).

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Builds an undirected graph representing the grid connectivity from the 'adjacent' static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state facts.
    2. Identify the current location of each box from the state facts.
    3. Initialize the total heuristic value to 0.
    4. Iterate through each box that has a specified goal location in the task goals:
       - Get the box's current location from the state.
       - Get the box's goal location from the pre-computed goal information.
       - If the box's current location is different from its goal location:
         - Calculate the shortest path distance between the box's current location and its goal location using BFS on the pre-computed location graph. This distance estimates the minimum number of push actions required for the box itself.
         - Calculate the shortest path distance between the robot's current location and the box's current location using BFS. This distance estimates the minimum number of move actions required for the robot to reach the box.
         - If either of these calculated distances is infinite (meaning the target location is unreachable in the graph), the state is considered effectively unsolvable by this heuristic; return infinity.
         - Add the sum of the box-to-goal distance and the robot-to-box distance to the total heuristic value.
    5. Return the accumulated total heuristic value. If all boxes were already at their goals, the value will be 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goals are only of the form (at boxX locY)
            if predicate == "at" and len(args) == 2 and args[0].startswith('box'):
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        self.graph = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent" and len(args) == 3:
                loc1, loc2, direction = args
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add undirected edge (assuming adjacency is symmetric)
                if loc2 not in self.graph[loc1]:
                     self.graph[loc1].append(loc2)
                if loc1 not in self.graph[loc2]:
                     self.graph[loc2].append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not found, the state is invalid or terminal (shouldn't happen in solvable states)
        if robot_location is None:
             return float('inf')

        # Find current box locations
        current_box_locations = {}
        for fact in state:
             # Match facts like (at box1 loc_X_Y)
             if match(fact, "at", "box*", "*"):
                 box, location = get_parts(fact)[1:]
                 current_box_locations[box] = location

        total_heuristic = 0

        # Calculate heuristic for each box that has a goal specified
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box with a goal is not found in the current state, something is wrong
            # This case should ideally not occur in valid problem states during search
            if current_location is None:
                 return float('inf')

            if current_location != goal_location:
                # Calculate box-to-goal distance
                dist_box_goal = bfs_distance(self.graph, current_location, goal_location)

                # Calculate robot-to-box distance
                dist_robot_box = bfs_distance(self.graph, robot_location, current_location)

                # If either distance is infinite, the state is likely unsolvable from here
                if dist_box_goal == float('inf') or dist_robot_box == float('inf'):
                    return float('inf')

                # Add to total heuristic: sum of box_dist + robot_dist
                # This counts the minimum pushes needed for the box and the minimum moves
                # needed for the robot to reach the box.
                total_heuristic += dist_box_goal + dist_robot_box

        return total_heuristic
