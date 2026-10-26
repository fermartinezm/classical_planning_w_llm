from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

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
    This heuristic estimates the number of actions needed to move all boxes to their target locations.

    # Assumptions:
    - The robot can move to adjacent locations if they are clear.
    - Boxes can be pushed to adjacent clear locations.
    - The goal is to have all boxes in specific target locations.

    # Heuristic Initialization
    - Extract target locations for each box from the goal facts.
    - Build a grid map of the environment using static adjacency facts.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Extract the target location for each box from the goal facts.
    2. For each box, determine if it is already in the target location or needs to be moved.
    3. For boxes that need to be moved:
       a. Find the shortest path from the robot's current position to the box's current location.
       b. Find the shortest path from the box's current location to its target location.
       c. Calculate the total number of actions required to move the box to its target.
    4. Sum the required actions for all boxes and add any necessary movements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract target locations for each box
        self.box_targets = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box, loc = parts[1], parts[2]
                self.box_targets[box] = loc

        # Build a grid map of the environment
        self.locations = set()
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.adjacent:
                    self.adjacent[loc1] = []
                self.adjacent[loc1].append(loc2)
                if loc2 not in self.adjacent:
                    self.adjacent[loc2] = []
                self.adjacent[loc2].append(loc1)
            elif match(fact, "clear", "*"):
                loc = get_parts(fact)[1]
                self.locations.add(loc)

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Extract current positions
        robot_pos = None
        box_positions = {}
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                box, loc = get_parts(fact)[1], get_parts(fact)[2]
                box_positions[box] = loc

        if not robot_pos:
            return 0  # No robot position found, assume 0 (should not happen in valid state)

        total_actions = 0

        # For each box, determine if it needs to be moved and calculate required actions
        for box, target in self.box_targets.items():
            current = box_positions.get(box, None)
            if current == target:
                continue  # Box is already in target

            if current is None:
                continue  # Box not found in state (should not happen in valid state)

            # Calculate the shortest path from robot to current box position
            robot_to_box = self.breadth_first_search(robot_pos, current)
            if robot_to_box is None:
                continue  # No path found, assume maximum cost (should not happen in valid state)

            # Calculate the shortest path from current box position to target
            box_to_target = self.breadth_first_search(current, target)
            if box_to_target is None:
                continue  # No path found, assume maximum cost (should not happen in valid state)

            # Each step in the path is an action (move or push)
            total_actions += len(robot_to_box) + len(box_to_target)

        return total_actions

    def breadth_first_search(self, start, goal):
        """
        Perform a breadth-first search to find the shortest path between two locations.
        Returns the path as a list of locations, or None if no path exists.
        """
        visited = set()
        queue = [([start], start)]
        while queue:
            path, current = queue.pop(0)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.adjacent.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((new_path, neighbor))
        return None
