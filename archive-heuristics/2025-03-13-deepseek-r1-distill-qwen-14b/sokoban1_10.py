from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_6_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move each box to its target location by summing the shortest path distances for both the boxes and the robot.

    # Assumptions:
    - The robot can move to any adjacent location if it's clear.
    - Each box can be pushed one step at a time.
    - The minimal number of actions is the sum of the shortest paths for each box to its goal and the robot to each box.

    # Heuristic Initialization
    - Extract the target location for each box from the goal conditions.
    - Build an adjacency map from static facts to enable shortest path calculations.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each box, determine its current location and target location.
    2. Compute the shortest path from the box's current location to its target location.
    3. Compute the shortest path from the robot's current location to the box's current location.
    4. Sum these distances for all boxes to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency map
        self.adjacency = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = []
                self.adjacency[loc2].append(loc1)

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and args[0].startswith("box"):
                box, loc = args
                self.box_goals[box] = loc

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Current locations of boxes and robot
        box_locations = {}
        robot_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                if obj.startswith("box"):
                    box_locations[obj] = loc
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]

        if not robot_location:
            return 0  # Robot's location not found, which shouldn't happen in valid states

        total_cost = 0

        # For each box, calculate the distance from its current location to the goal
        # and the distance from the robot's current location to the box's current location
        for box, current_loc in box_locations.items():
            if box in self.box_goals:
                goal_loc = self.box_goals[box]
                if current_loc == goal_loc:
                    continue  # No action needed for this box

                # Compute shortest path from current_loc to goal_loc
                distance_box = self.shortest_path(current_loc, goal_loc)
                if distance_box is None:
                    return float('inf')  # Unsolvable state

                # Compute shortest path from robot_location to current_loc
                distance_robot = self.shortest_path(robot_location, current_loc)
                if distance_robot is None:
                    return float('inf')  # Unsolvable state

                total_cost += distance_box + distance_robot

        return total_cost

    def shortest_path(self, start, end):
        """
        Compute the shortest path using BFS.

        Returns the number of steps or None if no path exists.
        """
        visited = set()
        queue = deque()
        queue.append((start, 0))

        while queue:
            current, steps = queue.popleft()
            if current == end:
                return steps
            if current in visited:
                continue
            visited.add(current)

            if current in self.adjacency:
                for neighbor in self.adjacency[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, steps + 1))

        return None  # No path found
