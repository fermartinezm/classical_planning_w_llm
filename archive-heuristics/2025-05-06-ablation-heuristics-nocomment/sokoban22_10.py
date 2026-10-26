import math
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def parse_location(location_str):
    """
    Parses a location string like 'loc_row_col' into (row, col) tuple.
    Assumes the format is always 'loc_row_col' where row and col are integers.
    """
    parts = location_str.split('_')
    # Basic validation for expected format
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers if necessary,
            # but assuming valid PDDL input based on examples.
            pass
    # Return None or raise error for unexpected formats
    return None # Or raise ValueError(f"Unexpected location format: {location_str}")


def manhattan_distance(loc1_str, loc2_str):
    """
    Calculates Manhattan distance between two location strings.
    Returns infinity if parsing fails for either location.
    """
    coords1 = parse_location(loc1_str)
    coords2 = parse_location(loc2_str)

    if coords1 is None or coords2 is None:
        # Should not happen with valid PDDL, but handle defensively
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing two components
        for each box that is not yet at its goal location:
        1. The Manhattan distance from the box's current location to its goal location.
           This estimates the minimum number of pushes required for that box, ignoring
           obstacles and robot position.
        2. The Manhattan distance from the robot's current location to the box's
           current location. This estimates the robot's effort to reach the vicinity
           of the box to be able to push it.

        The total heuristic value is the sum of these combined costs over all boxes
        that need to be moved.

    Assumptions:
        - Location names follow the format 'loc_row_col' where row and col
          are integers representing grid coordinates, allowing Manhattan distance
          calculation.
        - The goal state is defined solely by the target locations of the boxes.
        - The robot is always present in the state.
        - The heuristic is non-admissible and designed for greedy best-first search.
          It does not guarantee finding the optimal solution but aims to guide
          the search efficiently. It ignores obstacles (other boxes, walls)
          when calculating distances, which is a common relaxation.

    Heuristic Initialization:
        - The constructor receives the planning task object.
        - It extracts the goal facts from `task.goals`.
        - It stores the target location for each box specified in the goal
          in a dictionary `self.goal_locations`, mapping box names to location strings.
        - Static facts (like 'adjacent') are not explicitly used for distance
          calculation in this Manhattan-based heuristic, but the assumption
          of a grid structure is derived from them and the location naming.

    Step-By-Step Thinking for Computing Heuristic:
        1. Access the current state from the provided `node`.
        2. Initialize variables to store the robot's current location and the
           current locations of all relevant boxes (those listed in the goals).
           Iterate through the facts in the current state:
           - If a fact is `(at-robot ?l)`, store `?l` as the robot's location.
           - If a fact is `(at ?b ?l)` and `?b` is a box listed in `self.goal_locations`,
             store `?l` as the current location for box `?b`.
        3. Initialize the total heuristic value `total_cost` to 0.
        4. Initialize a boolean flag `all_boxes_at_goal` to True.
        5. Iterate through each box and its goal location stored in `self.goal_locations`:
           a. Get the box's current location from the information gathered in step 2.
           b. Check if the box's current location is different from its goal location.
           c. If the box is not at its goal location:
              i. Set `all_boxes_at_goal` to False.
              ii. Calculate the Manhattan distance between the box's current location
                  and its goal location. Add this distance to `total_cost`.
              iii. Calculate the Manhattan distance between the robot's current location
                   and the box's current location. Add this distance to `total_cost`.
        6. After iterating through all goal boxes, check the `all_boxes_at_goal` flag.
        7. If `all_boxes_at_goal` is True, the state is a goal state, so return 0.
        8. Otherwise, return the calculated `total_cost`.
    """
    def __init__(self, task):
        self.goals = task.goals
        # Extract goal locations for each box from the goal facts
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box location)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
        # Note: Static facts (like adjacent) are available in task.static
        # but are not explicitly used in this Manhattan distance heuristic.

    def __call__(self, node):
        state = node.state

        # Find current robot and box locations in the current state
        robot_location = None
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # Only track locations for boxes that are part of the goal
                 if obj in self.goal_locations:
                     current_box_locations[obj] = loc

        total_cost = 0
        all_boxes_at_goal = True

        # Calculate heuristic contribution for each box that needs to reach a goal
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # Ensure the box's current location was found in the state
            # (It should always be present in a valid state)
            if current_location is not None:
                if current_location != goal_location:
                    all_boxes_at_goal = False

                    # Component 1: Distance for the box to reach its goal
                    box_dist = manhattan_distance(current_location, goal_location)
                    total_cost += box_dist

                    # Component 2: Distance for the robot to reach the box
                    # The robot must be located in a valid state
                    if robot_location:
                         robot_dist = manhattan_distance(robot_location, current_location)
                         total_cost += robot_dist
                    # else: # Defensive: if robot location wasn't found (shouldn't happen)
                    #     return float('inf') # Or some large penalty

        # If all boxes are at their goal locations, the heuristic is 0
        if all_boxes_at_goal:
            return 0

        # Otherwise, return the calculated total cost
        return total_cost

