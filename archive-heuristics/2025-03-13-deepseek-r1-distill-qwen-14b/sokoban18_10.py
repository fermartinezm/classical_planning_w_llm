from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_2)".
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
    - Each box must be pushed to its target location.
    - The robot can only push a box if it is adjacent to it.
    - Boxes can block each other, so their movements are interdependent.

    # Heuristic Initialization
    - Extract the target location for each box from the goal conditions.
    - Build a map of static facts, including the adjacency relationships between locations.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box, calculate the Manhattan distance from its current location to the target location.
    2. Sum these distances to get the total steps needed if boxes could move independently.
    3. Calculate the minimum distance the robot needs to move to reach any box that isn't in its target position.
    4. Add a small penalty for each box that shares a row or column with another box to account for potential movement conflicts.
    5. Combine these values to estimate the total number of actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract target locations for each box
        self.box_targets = {}
        for goal in self.goals:
            predicate, box, loc = get_parts(goal)
            if predicate == "at":
                self.box_targets[box] = loc

        # Build adjacency map for static facts
        self.adjacent = {}
        for fact in self.static:
            if not match(fact, "adjacent", "*", "*", "*"):
                continue
            from_loc, to_loc, direction = get_parts(fact)
            if from_loc not in self.adjacent:
                self.adjacent[from_loc] = {}
            self.adjacent[from_loc][to_loc] = direction
            if to_loc not in self.adjacent:
                self.adjacent[to_loc] = {}
            self.adjacent[to_loc][from_loc] = opposite(direction)

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Extract current locations of boxes and the robot
        box_locations = {}
        robot_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                if obj == "robot":
                    robot_location = loc
                else:
                    box_locations[obj] = loc

        # If no boxes need to be moved, return 0
        if not box_locations:
            return 0

        # Calculate the sum of Manhattan distances for each box
        total_distance = 0
        for box, current_loc in box_locations.items():
            target_loc = self.box_targets.get(box, None)
            if target_loc is None:
                continue  # Box has no target, ignore it
            # Split location into coordinates
            current = current_loc.split('_')
            target = target_loc.split('_')
            x1, y1 = int(current[1]), int(current[2])
            x2, y2 = int(target[1]), int(target[2])
            total_distance += abs(x1 - x2) + abs(y1 - y2)

        # Calculate the minimum distance the robot needs to move to reach any box
        if robot_location is not None:
            min_robot_distance = float('inf')
            for box, current_loc in box_locations.items():
                target_loc = self.box_targets.get(box, None)
                if target_loc is None:
                    continue
                if current_loc == target_loc:
                    continue  # Box is already in target
                # Get coordinates
                r_x, r_y = map(int, robot_location.split('_')[1:])
                c_x, c_y = map(int, current_loc.split('_')[1:])
                # Calculate Manhattan distance
                distance = abs(r_x - c_x) + abs(r_y - c_y)
                if distance < min_robot_distance:
                    min_robot_distance = distance
            if min_robot_distance != float('inf'):
                total_distance += min_robot_distance

        # Add penalty for boxes in the same row or column
        # Check each box against others
        penalty = 0
        box_count = len(box_locations)
        for i in range(box_count):
            for j in range(i + 1, box_count):
                box1 = box_locations.keys()[i]
                box2 = box_locations.keys()[j]
                loc1 = box_locations[box1]
                loc2 = box_locations[box2]
                if loc1.startswith(loc2.split('_')[0]) or loc2.startswith(loc1.split('_')[0]):
                    penalty += 1  # Same row or column
        total_distance += penalty

        return total_distance

def opposite(direction):
    """Return the opposite direction."""
    return {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}[direction]
