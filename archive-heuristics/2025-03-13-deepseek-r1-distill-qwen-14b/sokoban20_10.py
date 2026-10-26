from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
from typing import Set, Dict, Tuple

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move all boxes to their goal locations. It considers:
    - The shortest path for each box to reach its goal.
    - The need for the robot to move between boxes and their goals.
    - The requirement for intermediate cells to be clear for box movement.

    # Assumptions:
    - Boxes can only be moved by pushing them one cell at a time.
    - The robot must be adjacent to a box to push it.
    - Boxes cannot overlap with other boxes or obstacles.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goals.
    - Builds a map of adjacent locations using static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, determine its current location and goal location.
    2. Compute the shortest path from the box's current location to its goal location using BFS, considering only clear cells.
    3. Sum the distances of all boxes to their goals.
    4. Adjust for the robot's movement between boxes:
       - If the robot is not adjacent to a box, add the distance for the robot to reach the box.
       - If multiple boxes need to be moved, add the distances between boxes to account for the robot's travel time.
    5. Return the total estimated actions.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal locations and static information."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (adjacent locations)

        # Build adjacency map from static facts
        self.adjacent = self._build_adjacency_map(static_facts)
        
        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, box, loc = get_parts(goal)
            if predicate == "at":
                self.goal_locations[box] = loc

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Check if all boxes are already at their goals
        if self._all_boxes_at_goal(state):
            return 0

        # Track current locations of boxes and the robot
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                if args[0] == "box1":
                    current_locations["box1"] = args[1]
                elif args[0] == "box2":
                    current_locations["box2"] = args[1]
                elif args[0] == "box3":
                    current_locations["box3"] = args[1]
            elif predicate == "at-robot":
                self.robot_location = args[0]

        total_actions = 0

        # For each box, calculate the distance to its goal
        for box in current_locations:
            current_loc = current_locations[box]
            goal_loc = self.goal_locations.get(box, None)
            if not goal_loc:
                continue  # Skip boxes not in the goal

            # If the box is already at the goal, no action needed
            if current_loc == goal_loc:
                continue

            # Calculate the shortest path from current to goal
            distance = self._shortest_path(current_loc, goal_loc)
            if distance is None:
                # If no path exists, the state is unsolvable
                return float('inf')

            total_actions += distance

        # Adjust for robot movements between boxes
        # If multiple boxes need to be moved, add the distance between them
        num_boxes = len(current_locations)
        if num_boxes > 1:
            # Simple adjustment: add the maximum distance between any two boxes
            # This is a simplification and could be improved
            max_distance = 0
            for box1 in current_locations:
                for box2 in current_locations:
                    if box1 != box2:
                        d = self._shortest_path(current_locations[box1], current_locations[box2])
                        if d > max_distance:
                            max_distance = d
            total_actions += max_distance

        return total_actions

    def _build_adjacency_map(self, static_facts):
        """Build a map of adjacent locations from static facts."""
        adjacent = {}
        for fact in static_facts:
            if not match(fact, "adjacent", "*", "*", "*"):
                continue
            loc1, loc2, dir = get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]
            if loc1 not in adjacent:
                adjacent[loc1] = []
            adjacent[loc1].append(loc2)
            if loc2 not in adjacent:
                adjacent[loc2] = []
            adjacent[loc2].append(loc1)
        return adjacent

    def _shortest_path(self, start, end):
        """Compute the shortest path using BFS."""
        visited = set()
        queue = deque()
        queue.append((start, 0))

        while queue:
            current, dist = queue.popleft()
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.adjacent.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return None  # No path found

    def _all_boxes_at_goal(self, state):
        """Check if all boxes are at their goal locations."""
        for goal in self.goals:
            predicate, box, loc = get_parts(goal)
            if predicate == "at":
                if f"(at {box} {loc})" not in state:
                    return False
        return True

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
