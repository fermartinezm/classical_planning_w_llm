from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their target locations. It considers both the distance each box needs to travel and the robot's required movements to position itself for pushing.

    # Assumptions:
    - The robot can push one box at a time.
    - The robot must be adjacent to a box to push it.
    - The goal is to have each box in a specific target location.

    # Heuristic Initialization
    - Extracts target locations for each box from the goal facts.
    - Builds a grid map of locations and their adjacencies from static facts.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box, determine its current location and target location.
    2. Calculate the Manhattan distance between the box's current and target locations.
    3. Determine the minimum number of moves the robot needs to reach the correct side of the box to push it.
    4. Sum the distances for all boxes to estimate the total number of actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract target locations for each box
        self.box_targets = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box, loc = parts[1], parts[2]
                self.box_targets[box] = loc

        # Build a grid map of all locations and their adjacent locations
        self.location_map = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                from_loc, to_loc, direction = get_parts(fact)
                if from_loc not in self.location_map:
                    self.location_map[from_loc] = []
                self.location_map[from_loc].append((to_loc, direction))

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # If all goals are already achieved, return 0
        if self.goals.issubset(state):
            return 0

        # Track current locations of boxes and the robot
        current_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                current_locations[obj] = loc
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]

        heuristic_cost = 0

        # For each box, calculate the required actions
        for box, target in self.box_targets.items():
            if box not in current_locations:
                continue  # Box is already at target or doesn't exist

            current_loc = current_locations[box]
            if current_loc == target:
                continue  # Box is already at target

            # Calculate Manhattan distance between current and target locations
            # Extracting coordinates from location strings (e.g., loc_3_4 -> (3,4))
            def parse_loc(loc_str):
                x = int(loc_str.split('_')[1])
                y = int(loc_str.split('_')[2])
                return (x, y)

            current_coords = parse_loc(current_loc)
            target_coords = parse_loc(target)

            # Manhattan distance for the box to move
            box_distance = abs(current_coords[0] - target_coords[0]) + abs(current_coords[1] - target_coords[1])

            # Determine the direction the robot needs to be relative to the box to push it
            # The robot needs to be on the opposite side of the target
            if current_coords[0] < target_coords[0]:
                robot_side = 'down'
            elif current_coords[0] > target_coords[0]:
                robot_side = 'up'
            else:
                if current_coords[1] < target_coords[1]:
                    robot_side = 'left'
                else:
                    robot_side = 'right'

            # Calculate the distance the robot needs to move to get to the correct side
            # This is the minimum number of moves to reach the adjacent location in the required direction
            # We assume the robot can move optimally through adjacent locations
            # So we need to find the shortest path from robot's current location to the required side of the box

            # Get the location of the box
            box_loc = current_loc

            # Get the adjacent location of the box in the required direction
            required_adjacent = None
            for adj in self.location_map.get(box_loc, []):
                if adj[1] == robot_side:
                    required_adjacent = adj[0]
                    break

            if required_adjacent is None:
                # Cannot find a path, which shouldn't happen in Sokoban
                continue

            # Now find the shortest path from robot's location to required_adjacent
            # Using BFS for simplicity
            from collections import deque
            visited = set()
            queue = deque()
            queue.append((robot_loc, 0))
            visited.add(robot_loc)

            found = False
            while queue:
                current, dist = queue.popleft()
                if current == required_adjacent:
                    found = True
                    break
                for adj in self.location_map.get(current, []):
                    next_loc = adj[0]
                    if next_loc not in visited:
                        visited.add(next_loc)
                        queue.append((next_loc, dist + 1))

            if found:
                robot_moves = dist
            else:
                robot_moves = float('inf')  # Shouldn't happen in Sokoban

            heuristic_cost += box_distance + robot_moves

        return heuristic_cost
