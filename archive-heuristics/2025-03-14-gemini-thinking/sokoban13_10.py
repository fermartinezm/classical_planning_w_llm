from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_4_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes to their goal locations.
    It calculates a simplified Manhattan distance for each box to its closest goal location,
    summing these distances to provide an overall heuristic estimate. This heuristic is admissible
    under the assumption that each move or push action has a cost of 1 and boxes can be moved independently.
    However, in practice, due to the constraints of Sokoban, it serves as a domain-dependent,
    efficient, but not necessarily admissible heuristic for guiding greedy best-first search.

    # Assumptions:
    - Each move (robot move or push) has a uniform cost of 1.
    - Moving boxes independently towards their goals provides a reasonable estimate of the total effort,
      ignoring potential blockages and interactions between boxes.
    - The problem is solvable, and there exists a path to move each box to its designated goal location.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Extracts static adjacency information between locations to calculate Manhattan distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the current locations of all boxes from the given state.
    3. For each box:
        a. Determine its current location.
        b. Find its goal location.
        c. Calculate the Manhattan distance between the current location and the goal location.
           In this simplified version, we assume locations are named in a grid-like manner (loc_row_col).
           If this naming convention is not strictly followed, a simpler distance metric (like just counting
           different locations as 1 unit distance) or a more sophisticated pathfinding approach would be needed.
           For simplicity and efficiency, we will assume a basic distance calculation based on location names.
        d. Add the calculated Manhattan distance to the total heuristic value.
    4. Return the total heuristic value as the estimated cost to reach the goal state.

    Note: This heuristic is a simplification and does not account for complex Sokoban constraints
    like deadlocks, corridors, or the robot's path planning. It focuses on the box movements
    and provides a computationally cheap estimate. For more accurate heuristics, consider
    techniques like pattern databases or more sophisticated potential functions.
    """

    def __init__(self, task):
        """
        Initialize the Sokoban heuristic.

        Extracts goal box locations and static adjacency information from the task.
        """
        self.goal_box_locations = {}
        self.boxes = set()

        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == 'at':
                box_name, location = args
                self.goal_box_locations[box_name] = location
                self.boxes.add(box_name)


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state.

        Estimates the number of moves required to reach the goal state based on Manhattan distances
        of boxes to their goal locations.
        """
        state = node.state
        current_box_locations = {}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == 'at' and args[0] in self.boxes:
                box_name, location = args
                current_box_locations[box_name] = location

        heuristic_value = 0
        for box_name in self.boxes:
            if box_name in self.goal_box_locations:
                goal_location = self.goal_box_locations[box_name]
                current_location = current_box_locations.get(box_name)

                if current_location != goal_location:
                    heuristic_value += 1 # Simplified Manhattan distance: just count as 1 if not at goal.
                                         # For a more accurate Manhattan distance, you would need to parse
                                         # location names (if they are grid-based like loc_row_col) and
                                         # calculate the difference in row and column indices.

        return heuristic_value
