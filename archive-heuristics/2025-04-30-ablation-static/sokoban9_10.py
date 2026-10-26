from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_5)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location to its goal location.
    2. The shortest path distance from the robot's current location to the nearest box that is not yet at its goal.

    This heuristic is non-admissible as it does not account for potential blockages by other boxes
    or the need for the robot to move *around* boxes or obstacles to get into a pushing position.
    It also assumes the robot can immediately start moving towards the nearest box needing a push.

    # Assumptions
    - The locations form a graph connected by `adjacent` predicates.
    - The shortest path distance on this graph is a reasonable estimate for movement cost.
    - The heuristic ignores dynamic facts like `clear` for simplicity and efficiency, assuming paths are generally traversable.
    - The mapping from boxes to goal locations is fixed as specified in the goal state.

    # Heuristic Initialization
    - Build an undirected graph representing the locations and their adjacencies based on `adjacent` facts.
    - Extract the goal location for each box from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Determine which boxes are not currently at their assigned goal location.
    4. If all boxes are at their goal locations, the heuristic is 0.
    5. For each box not at its goal:
       a. Calculate the shortest path distance from the box's current location to its goal location using BFS on the location graph. Sum these distances (this estimates the minimum number of pushes required).
       b. Calculate the shortest path distance from the robot's current location to the box's current location using BFS. Keep track of the minimum such distance across all boxes not at their goal.
    6. The total heuristic value is the sum of the total box-to-goal distance and the minimum robot-to-box distance.
    7. If any required BFS distance calculation returns infinity (meaning a location is unreachable), the state is likely unsolvable, and the heuristic should return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and extracting
        box goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the undirected location graph from adjacent facts
        self.graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Assume adjacency is symmetric

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at boxX locY)
            if match(goal, "at", "*", "*"):
                _, box, goal_loc = get_parts(goal)
                self.goal_locations[box] = goal_loc

        # Store the set of all box names
        self.boxes = set(self.goal_locations.keys())


    def bfs_distance(self, start, end):
        """
        Calculate the shortest path distance between two locations using BFS.
        Returns float('inf') if the end is unreachable from the start.
        """
        if start == end:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc not in self.graph:
                 # Should not happen if graph is built correctly from all locations
                 continue

            for neighbor in self.graph[current_loc]:
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # End location is unreachable
        return float('inf')


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        if robot_loc is None:
             # Robot location not found, problem state is invalid or unsolvable
             return float('inf')

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.boxes: # Only track locations of objects that are boxes
                    current_box_locations[obj] = loc

        # Identify boxes not at their goal
        boxes_to_move = []
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)
            if current_loc != goal_loc:
                boxes_to_move.append(box)

        # If all boxes are at their goal, heuristic is 0
        if not boxes_to_move:
            return 0

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')

        # Calculate heuristic components for boxes that need moving
        for box in boxes_to_move:
            box_loc = current_box_locations.get(box)
            goal_loc = self.goal_locations[box]

            if box_loc is None:
                 # Box location not found, problem state is invalid or unsolvable
                 return float('inf')

            # Distance from box to its goal
            box_dist = self.bfs_distance(box_loc, goal_loc)
            if box_dist == float('inf'):
                # Box cannot reach its goal
                return float('inf')
            total_box_distance += box_dist

            # Distance from robot to this box
            robot_to_box_dist = self.bfs_distance(robot_loc, box_loc)
            if robot_to_box_dist == float('inf'):
                 # Robot cannot reach this box
                 return float('inf')
            min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_box_dist)

        # The heuristic is the sum of box distances and the robot's distance to the nearest box
        return total_box_distance + min_robot_to_box_distance

