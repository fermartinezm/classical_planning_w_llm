from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their respective target locations.

    # Assumptions:
    - Each box must be pushed from its current position to a target location.
    - The robot can move to any adjacent clear location.
    - The target location for each box is known and static.

    # Heuristic Initialization
    - Extract the target location for each box from the goal facts.
    - Build an adjacency graph from the static facts to represent possible movements.
    - Precompute the shortest path between every pair of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of the robot and each box from the state.
    2. For each box, determine its target location from the goal facts.
    3. If a box is already at its target location, contribute 0 to the heuristic.
    4. For each box not at its target, calculate:
       a. The shortest path from the robot's current position to the box's current position.
       b. The shortest path from the box's current position to its target location.
       c. Sum these distances to estimate the number of actions needed for that box.
    5. Sum the estimated actions for all boxes to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (adjacency relationships between locations).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build adjacency graph from static facts
        self.adjacency = defaultdict(set)
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, _ = get_parts(fact)
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)

        # Precompute shortest paths between all pairs of locations
        self.distances = defaultdict(dict)
        for loc in self.adjacency:
            queue = deque()
            queue.append((loc, 0))
            visited = set()
            visited.add(loc)
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) >= 3 and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Extract robot's current location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        # Extract current locations of boxes
        boxes = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                box = get_parts(fact)[0]
                loc = get_parts(fact)[1]
                boxes[box] = loc

        total_cost = 0

        # Calculate heuristic for each box
        for box, current_loc in boxes.items():
            if box not in self.goal_locations:
                continue  # Skip boxes that are not in the goals
            target_loc = self.goal_locations[box]

            if current_loc == target_loc:
                continue  # Already at target, no cost

            # Get distance from robot to current location of the box
            if robot_loc not in self.distances or current_loc not in self.distances[robot_loc]:
                # If no path exists (shouldn't happen in solvable problems), skip
                continue
            dist_robot_to_box = self.distances[robot_loc].get(current_loc, float('inf'))

            # Get distance from current location to target location
            if current_loc not in self.distances or target_loc not in self.distances[current_loc]:
                # If no path exists (shouldn't happen in solvable problems), skip
                continue
            dist_box_to_target = self.distances[current_loc].get(target_loc, float('inf'))

            total_cost += dist_robot_to_box + dist_box_to_target

        return total_cost
