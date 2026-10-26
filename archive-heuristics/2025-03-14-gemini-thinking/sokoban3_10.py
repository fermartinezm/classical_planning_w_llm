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


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of moves required to solve a Sokoban puzzle by calculating the sum of Manhattan distances
    for each box from its current location to its goal location. It assumes that each box needs to be moved independently to its goal,
    ignoring potential blockages by other boxes or walls, and robot movement constraints.

    # Assumptions:
    - The heuristic assumes a grid-based world where locations can be represented by coordinates.
    - It assumes that the cost of moving a box is proportional to the Manhattan distance.
    - It ignores the 'clear' predicate and potential blockages, focusing solely on the distance.
    - It assumes each box has a unique goal location as specified in the goal condition.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Parses location names to extract coordinate information (assuming location names are in the format 'loc_row_col').

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract goal box locations from the task definition and store them in `goal_box_locations`.
    3. For each goal condition in the task that is of the form '(at box ?location)':
        - Identify the box and its goal location.
        - Store this information in `goal_box_locations` dictionary where keys are box names and values are goal location names.
    4. For each location mentioned in the static facts (specifically in 'adjacent' predicates):
        - Parse the location name (e.g., 'loc_1_2') to extract row and column numbers.
        - Store these coordinates in `location_coordinates` dictionary, keyed by location name.
    5. In the heuristic function (__call__):
        - Initialize the heuristic estimate `h` to 0.
        - For each box in `goal_box_locations`:
            - Find the current location of the box in the current state by checking for facts of the form '(at box ?location)'.
            - Retrieve the goal location for this box from `goal_box_locations`.
            - Extract the coordinates of the current location and the goal location from `location_coordinates`.
            - Calculate the Manhattan distance between the current and goal location coordinates: `|row1 - row2| + |col1 - col2|`.
            - Add this Manhattan distance to the heuristic estimate `h`.
        - Return the total heuristic estimate `h`.

    This heuristic provides a lower bound on the number of moves in an ideal scenario where boxes can be moved directly to their goals without any obstacles.
    It is efficiently computable as it primarily involves parsing strings and simple arithmetic operations.
    """

    def __init__(self, task):
        """
        Initialize the sokoban heuristic by extracting goal box locations and parsing location coordinates.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_box_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                predicate, box_name, goal_location = parts
                self.goal_box_locations[box_name] = goal_location

        self.location_coordinates = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations.add(loc1)
                locations.add(loc2)

        for loc_name in locations:
            parts = loc_name.split('_')
            if len(parts) == 3 and parts[0] == 'loc':
                try:
                    row = int(parts[1])
                    col = int(parts[2])
                    self.location_coordinates[loc_name] = (row, col)
                except ValueError:
                    pass # Ignore locations that do not follow loc_row_col format


    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is the sum of Manhattan distances of each box to its goal location.
        """
        state = node.state
        h = 0
        for box_name, goal_location in self.goal_box_locations.items():
            current_location = None
            for fact in state:
                if match(fact, "at", box_name, "*"):
                    current_location = get_parts(fact)[2]
                    break

            if current_location and goal_location in self.location_coordinates and current_location in self.location_coordinates:
                goal_coord = self.location_coordinates[goal_location]
                current_coord = self.location_coordinates[current_location]
                manhattan_distance = abs(current_coord[0] - goal_coord[0]) + abs(current_coord[1] - goal_coord[1])
                h += manhattan_distance
        return h
