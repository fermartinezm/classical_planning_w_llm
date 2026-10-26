from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_4_7)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their target locations.

    # Assumptions:
    - The robot can move freely between locations.
    - Each box must be pushed individually to its target location.
    - Moving a box may require the robot to backtrack to move other boxes.

    # Heuristic Initialization
    - Extract the goal locations for each box and static facts (adjacent locations) from the task.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Extract the current positions of the robot and all boxes.
    2. For each box, calculate the shortest path from the robot's current position to the box's target location.
    3. Sum the distances for all boxes, adding extra steps for necessary backtracking.
    4. Return the total estimated actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (adjacent locations).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts about the domain

        # Build adjacency graph from static facts
        self.adjacent = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.adjacent:
                    self.adjacent[loc1] = {}
                self.adjacent[loc1][loc2] = dir
                if loc2 not in self.adjacent:
                    self.adjacent[loc2] = {}
                self.adjacent[loc2][loc1] = dir

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # Extract current positions
        current_positions = {}
        for fact in state:
            if match(fact, "at-robot", "*"):
                current_positions["robot"] = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                if obj.startswith("box"):
                    current_positions[obj] = loc

        # If no positions are found, return a high value
        if "robot" not in current_positions:
            return float('inf')

        total_actions = 0

        # For each box, calculate the minimal path
        for box in self.goal_locations:
            target = self.goal_locations[box]
            current_loc = current_positions.get(box, None)
            if current_loc is None:
                continue

            # Find the shortest path from robot's current position to the box's current location
            robot_loc = current_positions["robot"]
            path = self.breadth_first_search(robot_loc, current_loc)
            if path is None:
                # If no path exists, return infinity (unreachable state)
                return float('inf')
            distance_to_box = len(path) - 1  # Subtract start node

            # Find the shortest path from box's current location to target
            target_path = self.breadth_first_search(current_loc, target)
            if target_path is None:
                return float('inf')
            distance_to_target = len(target_path) - 1

            # Total actions for this box: distance to box + 1 (push action) + distance to target
            total_actions += distance_to_box + 1 + distance_to_target

            # Update robot's position for next box calculation
            robot_loc = target

        return total_actions

    def breadth_first_search(self, start, goal):
        """
        Perform BFS to find the shortest path between two locations.
        Returns the path as a list of locations, or None if no path exists.
        """
        if start == goal:
            return [start]

        visited = set()
        queue = deque()
        queue.append((start, [start]))

        while queue:
            current, path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path

            # Explore adjacent locations
            if current in self.adjacent:
                for neighbor, _ in self.adjacent[current].items():
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))

        return None
