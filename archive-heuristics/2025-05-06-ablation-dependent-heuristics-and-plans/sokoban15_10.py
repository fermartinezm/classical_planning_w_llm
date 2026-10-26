from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re
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

def parse_location(location_str):
    """Parses a location string like 'loc_row_col' into (row, col) integers."""
    match = re.match(r'loc_(\d+)_(\d+)', location_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None # Should not happen in this domain if location format is consistent

def build_location_graph(static_facts):
    """Builds an adjacency list representation of the location graph from 'adjacent' facts."""
    graph = {}
    for fact in static_facts:
        if match(fact, "adjacent", "*", "*", "*"):
            _, loc1, loc2, _ = get_parts(fact)
            if loc1 not in graph:
                graph[loc1] = []
            graph[loc1].append(loc2)
            # Assuming adjacency is symmetric and both directions are listed in facts
    return graph

def shortest_path_bfs(graph, start, goals, traversable_locations=None):
    """
    Finds the shortest path distance from a start location to any of the goal locations
    using BFS on the graph.
    Optionally takes a set of traversable_locations. Only nodes in this set can be visited.
    Returns float('inf') if no path is found.
    """
    if start not in graph or (traversable_locations is not None and start not in traversable_locations):
        return float('inf') # Start is not in the graph or not traversable

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc in goals:
            return dist

        if current_loc not in graph:
             continue # Should not happen if start was in graph, but defensive check

        for neighbor in graph[current_loc]:
            if (traversable_locations is None or neighbor in traversable_locations) and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by summing the shortest path distances of each box from its goal location
    on the grid graph, and adding the shortest path distance from the robot
    to the nearest box that needs to be moved.

    # Assumptions
    - The locations are arranged in a grid and defined by 'adjacent' facts.
    - Shortest path distance on the grid graph provides a reasonable estimate of the minimum number of pushes required for a box (ignoring dynamic obstacles).
    - The heuristic considers the robot's distance to the closest box needing movement, considering clear locations.
    - This heuristic ignores potential deadlocks and complex interactions between boxes.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goal conditions.
    - Build the location graph from 'adjacent' static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the current location of the robot.
    3. Identify the current location of each box.
    4. Identify the set of clear locations.
    5. Initialize `total_box_goal_distance` to 0.
    6. Find the set of boxes that are not at their goal locations.
    7. For each box that is not at its goal location:
       - Calculate the shortest path distance for the box's current location to its goal location on the grid graph (using BFS on the full graph). This estimates the minimum pushes needed for this box.
       - Add this distance to `total_box_goal_distance`. If any box goal is unreachable on the grid, the total heuristic is infinity.
    8. If there are no boxes not at their goals (which should be caught by step 1 unless the goal includes non-'at' facts), the heuristic calculation is complete (though step 1 already returned 0).
    9. If there are boxes not at their goals, find the locations of these boxes.
    10. Find the set of locations adjacent to any box that needs to be moved. These are potential target locations for the robot to reach to initiate a push.
    11. Calculate the shortest path distance from the robot's current location to any of these potential robot target locations. This distance is calculated using BFS on the grid graph, traversing only locations that are currently marked as 'clear'. If the robot cannot reach any such location, the total heuristic is infinity.
    12. The total heuristic value is the sum of `total_box_goal_distance` and the calculated robot-to-nearest-box distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static # Static facts

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts
        self.location_graph = build_location_graph(static_facts)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if the goal is reached. If so, heuristic is 0.
        if self.goals <= state:
             return 0

        # 2. Identify the current location of the robot.
        current_robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                 current_robot_location = get_parts(fact)[1]
                 break # Assuming only one robot

        if current_robot_location is None:
             # Robot location not found, invalid state?
             return float('inf')

        # 3. Identify the current location of each box.
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "box*", "*"): # Assuming objects starting with 'box' are boxes
                 box, location = get_parts(fact)[1:]
                 current_box_locations[box] = location

        # 4. Identify the set of clear locations.
        current_clear_locations = set()
        for fact in state:
            if match(fact, "clear", "*"):
                 current_clear_locations.add(get_parts(fact)[1])

        # 5. Initialize total_box_goal_distance to 0.
        total_box_goal_distance = 0
        boxes_not_at_goal = []

        # 6 & 7. Calculate sum of shortest path distances for boxes not at goal.
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            if current_location is None:
                 # Box location not found, invalid state?
                 return float('inf')

            if current_location != goal_location:
                boxes_not_at_goal.append(box)

                # Calculate shortest path distance for the box location to its goal
                # This is distance on the full grid graph, ignoring dynamic obstacles.
                distance = shortest_path_bfs(self.location_graph, current_location, {goal_location})
                if distance == float('inf'):
                    # Box goal is unreachable on the grid graph
                    return float('inf')
                total_box_goal_distance += distance

        # 8. If no boxes need moving, this state should have been caught by step 1.
        # If we reach here, there are boxes not at their goals.

        # 9. Find the set of locations of boxes that need to be moved.
        locations_of_boxes_to_move = {current_box_locations[box] for box in boxes_not_at_goal}

        # 10. Find the set of locations adjacent to any box that needs to be moved.
        potential_robot_targets = set()
        for box_loc in locations_of_boxes_to_move:
             if box_loc in self.location_graph:
                 for neighbor in self.location_graph[box_loc]:
                     # The robot needs to reach a location adjacent to the box.
                     potential_robot_targets.add(neighbor)

        # 11. Calculate shortest path from robot's current location to any of the potential robot targets.
        # The robot can only traverse locations in robot_traversable_locations (which are the clear locations).
        robot_distance = shortest_path_bfs(self.location_graph, current_robot_location, potential_robot_targets, traversable_locations=current_clear_locations)

        if robot_distance == float('inf'):
             # Robot cannot reach any location adjacent to a box that needs moving
             return float('inf')

        # 12. The total heuristic value is the sum of box-goal distances and the robot's distance to a box.
        return total_box_goal_distance + robot_distance

