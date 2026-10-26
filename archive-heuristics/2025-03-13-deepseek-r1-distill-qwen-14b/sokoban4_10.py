from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their respective goal locations.

    # Assumptions:
    - The robot can move to adjacent locations if they are clear.
    - The robot can push a box if it is adjacent to the box and the target location is clear.
    - The goal is to have each box in a specific location.

    # Heuristic Initialization
    - Extract the goal locations for each box from the problem's goal conditions.
    - Extract static facts to determine the layout of the Sokoban world.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the goal location for each box.
    2. For each box, determine its current location.
    3. Calculate the Manhattan distance from the box's current location to its goal location.
    4. Add 1 to the distance for the robot to move next to the box to push it.
    5. Sum the distances for all boxes to get the total estimated number of actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts about the world

        # Extract the position of each location (assumes locations are in the form loc_x_y)
        self.location_positions = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                # For simplicity, we'll assume a grid layout where loc_x_y has coordinates (x, y)
                x1, y1 = int(loc1.split('_')[1]), int(loc1.split('_')[2])
                x2, y2 = int(loc2.split('_')[1]), int(loc2.split('_')[2])
                self.location_positions[loc1] = (x1, y1)
                self.location_positions[loc2] = (x2, y2)

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

        # Track the current location of each box
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and args[0].startswith("box"):
                box, location = args
                current_locations[box] = location

        total_cost = 0  # Initialize action cost counter

        for box, goal_location in self.goal_locations.items():
            # Skip if the box is already at the goal
            if box not in current_locations:
                continue
            current_location = current_locations[box]

            if current_location == goal_location:
                continue  # No action needed

            # Calculate Manhattan distance between current and goal locations
            current_x, current_y = self.location_positions[current_location]
            goal_x, goal_y = self.location_positions[goal_location]
            distance = abs(current_x - goal_x) + abs(current_y - goal_y)

            # Add 1 for the robot to move next to the box to push it
            total_cost += distance + 1

        return total_cost
