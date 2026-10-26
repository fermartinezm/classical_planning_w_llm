# from fnmatch import fnmatch # Not needed with direct parsing
from heuristics.heuristic_base import Heuristic
import math # For infinity

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def get_coords(location_str):
    """
    Parses a location string like 'loc_R_C' into (row, col) tuple.

    Assumes location names are in the format 'loc_R_C' where R and C are integers.
    """
    try:
        parts = location_str.split('_')
        # parts will be ['loc', 'R', 'C']
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # Handle unexpected location format gracefully
        # Return a value that will result in a large distance, effectively penalizing
        # states with malformed location names if they somehow occur.
        return (math.inf, math.inf)


def manhattan_distance(loc1_str, loc2_str):
    """
    Calculates Manhattan distance between two location strings.

    Assumes location strings can be parsed by get_coords.
    Returns infinity if parsing fails for either location.
    """
    coords1 = get_coords(loc1_str)
    coords2 = get_coords(loc2_str)

    if coords1[0] == math.inf or coords2[0] == math.inf:
        return math.inf

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Estimates the cost to reach the goal state.

    Summary:
    This heuristic estimates the cost to solve a Sokoban state by combining
    the effort required to move the boxes and the effort required for the robot
    to reach a box. It sums the Manhattan distances from each box's current
    location to its goal location. Additionally, it adds the Manhattan distance
    from the robot's current location to the closest box that is not yet at its goal.
    This heuristic is non-admissible as it ignores obstacles and complex
    interactions between objects and the robot's movement during pushes.

    Assumptions:
    - Location names follow the format 'loc_R_C' where R and C are integers,
      allowing the calculation of Manhattan distances as a proxy for grid distance.
    - The grid structure implied by 'loc_R_C' is a reasonable approximation of
      the actual connectivity defined by the 'adjacent' predicates.
    - There is a one-to-one mapping between boxes and goal locations specified
      in the task's goal facts.
    - All boxes mentioned in the goal facts are present in the state facts.
    - The robot is always present in the state facts.

    Heuristic Initialization:
    In the __init__ method, the heuristic processes the task's goal facts
    to create a dictionary (self.box_goal_locations) mapping each box name
    (e.g., 'box1') to its target goal location string (e.g., 'loc_2_4').
    This pre-computation avoids repeatedly parsing goals during the search.

    Step-By-Step Thinking for Computing Heuristic:
    1.  The heuristic function (__call__) takes a search node representing a state.
    2.  It first finds the current location of the robot by iterating through the state facts and looking for the '(at-robot ?l)' predicate.
    3.  It then finds the current location of each box by iterating through the state facts and looking for the '(at ?b ?l)' predicate where ?b is a box.
    4.  It initializes a running total for the heuristic value (total_box_distance) and a variable to track the minimum distance from the robot to any misplaced box (min_robot_to_box_distance). It also counts misplaced boxes.
    5.  It iterates through the box-goal location mapping established during initialization.
    6.  For each box, it compares its current location in the state to its goal location.
    7.  If the box is not at its goal location:
        a.  It calculates the Manhattan distance between the box's current location and its goal location using the parsed row/column coordinates. This distance is added to total_box_distance.
        b.  It calculates the Manhattan distance between the robot's current location and the current location of this misplaced box.
        c.  It updates min_robot_to_box_distance if the distance calculated in the previous step is smaller.
    8.  After iterating through all boxes, if no boxes were found to be misplaced (misplaced_boxes_count is 0), the state is the goal state, and the heuristic returns 0.
    9.  If there were misplaced boxes, the final heuristic value is calculated as total_box_distance + min_robot_to_box_distance. This combines the estimated effort for the boxes to reach their goals with the estimated initial effort for the robot to reach a box it needs to push.
    10. The function includes basic error handling for parsing location names, returning infinity if parsing fails, which signals an invalid or problematic state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by extracting goal locations for boxes.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Heuristic Initialization:
        # Extract goal locations for each box from the task goals.
        # This mapping is stored in self.box_goal_locations.
        self.box_goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at box_name goal_location)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                 box_name = parts[1]
                 goal_location = parts[2]
                 self.box_goal_locations[box_name] = goal_location

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            An integer representing the estimated cost to reach the goal.
            Returns infinity if the state is invalid or cannot be parsed.
        """
        state = node.state

        # Step-By-Step Thinking for Computing Heuristic:
        # 1. Find the robot's current location from the state facts.
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
                break

        # If robot location is not found, the state is likely invalid.
        if robot_location is None:
             return math.inf

        # 2. Find the current location of each box from the state facts.
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box_name = parts[1]
                location = parts[2]
                current_box_locations[box_name] = location

        total_box_distance = 0
        min_robot_to_box_distance = math.inf
        misplaced_boxes_count = 0

        # 3. Iterate through each box that has a goal location defined in the task.
        for box_name, goal_location in self.box_goal_locations.items():
            # 4. For each box, check if its current location matches its goal location.
            current_location = current_box_locations.get(box_name)

            # If a box from the goal is not in the current state, something is wrong.
            if current_location is None:
                 return math.inf

            # 5. If the box is *not* at its goal location (it's misplaced):
            if current_location != goal_location:
                misplaced_boxes_count += 1
                # a. Calculate the Manhattan distance between the box's current location
                #    and its goal location using the parsed row/column coordinates.
                #    This distance is added to total_box_distance.
                #    This estimates the minimum number of pushes required for this box
                #    if obstacles and robot position constraints are ignored.
                dist_box_to_goal = manhattan_distance(current_location, goal_location)
                if dist_box_to_goal == math.inf: return math.inf # Propagate parsing errors
                total_box_distance += dist_box_to_goal

                # b. Calculate the Manhattan distance between the robot's current location
                #    and this misplaced box's current location.
                dist_robot_to_this_box = manhattan_distance(robot_location, current_location)
                if dist_robot_to_this_box == math.inf: return math.inf # Propagate parsing errors

                # c. Update the minimum robot-to-box distance found so far with this value.
                min_robot_to_box_distance = min(min_robot_to_box_distance, dist_robot_to_this_box)

        # 6. After iterating through all boxes, if no boxes were found to be misplaced
        #    (misplaced_boxes_count is 0), the state is the goal state, and the
        #    heuristic returns 0.
        if misplaced_boxes_count == 0:
            return 0

        # 7. If there were misplaced boxes, add the minimum robot-to-box distance
        #    (calculated in step 5c) to the total heuristic. This estimates the
        #    cost for the robot to reach *a* misplaced box to begin pushing.
        #    Handle case where min_robot_to_box_distance is still inf (e.g., no boxes found?)
        if min_robot_to_box_distance == math.inf:
             # This implies misplaced_boxes_count > 0, but min_robot_to_box_distance
             # remained inf, likely due to parsing errors or unexpected state.
             return math.inf

        # 8. Return the final total heuristic value.
        return total_box_distance + min_robot_to_box_distance
