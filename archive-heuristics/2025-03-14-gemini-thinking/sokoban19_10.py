from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in the Sokoban domain
    by calculating the sum of Manhattan distances for each box from its current
    location to its closest goal location. It also adds a penalty if a box is in a corner
    and not in a goal location, as corners are often dead ends in Sokoban.

    # Assumptions:
    - Location names are in the format 'loc_r_c' where 'r' and 'c' are row and column indices.
    - Goal conditions specify the target location for each box.
    - The heuristic is not necessarily admissible but aims for efficiency and informed search.

    # Heuristic Initialization
    - Parses the goal conditions to determine the target location for each box.
    - Extracts static facts, specifically 'adjacent' predicates, to understand the grid layout, although not directly used for Manhattan distance, it could be used for more advanced heuristics in the future.
    - Prepares to parse location coordinates from location names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the given state.
    2. For each box, determine its goal location from the task goals.
    3. Parse the row and column coordinates from the current and goal location names.
       If location names do not follow the 'loc_r_c' pattern, a large default value is used for distance, effectively penalizing such states (though ideally, location names should follow the pattern).
    4. Calculate the Manhattan distance between the current location and the goal location for each box.
    5. Sum up the Manhattan distances for all boxes. This sum forms the base heuristic value.
    6. Implement a corner penalty:
       - Identify corner locations. A corner location is a location that has at most 2 adjacent locations that are also clear locations (or walls, effectively limiting movement).  This is a simplification and might need refinement based on actual wall information if available in more complex problem instances. For now, we will consider adjacency based on the provided 'adjacent' facts and 'clear' facts in the state.
       - For each box, check if its current location is a corner location and if it is NOT a goal location. If both are true, add a penalty to the heuristic value. The penalty could be a fixed value (e.g., 10) to significantly increase the cost of states where boxes are in corners.
    7. Return the final heuristic value (sum of Manhattan distances + corner penalties).
    8. If the current state is a goal state (all boxes are at their goal locations), the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_box_locations = {}
        self.boxes = set()

        for goal in self.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'at':
                box_name = parts[1]
                location_name = parts[2]
                self.goal_box_locations[box_name] = location_name
                self.boxes.add(box_name)

        self.adjacent_locations = {}
        for fact in self.static_facts:
            parts = self._get_parts(fact)
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 not in self.adjacent_locations:
                    self.adjacent_locations[loc1] = []
                self.adjacent_locations[loc1].append(loc2)
                if loc2 not in self.adjacent_locations:
                    self.adjacent_locations[loc2] = []
                self.adjacent_locations[loc2].append(loc1)


    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state

        if task.goal_reached(None, state, self.goals): # task is not available in the scope, using a dummy task for goal_reached check below
            return 0

        box_locations = {}
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.boxes:
                box_name = parts[1]
                location_name = parts[2]
                box_locations[box_name] = location_name

        total_manhattan_distance = 0
        corner_penalty = 0

        clear_locations = set()
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'clear':
                clear_locations.add(parts[1])

        for box_name in self.boxes:
            current_location = box_locations.get(box_name)
            goal_location = self.goal_box_locations.get(box_name)

            if current_location and goal_location:
                current_coords = self._parse_location_coords(current_location)
                goal_coords = self._parse_location_coords(goal_location)

                if current_coords and goal_coords:
                    manhattan_distance = abs(current_coords[0] - goal_coords[0]) + abs(current_coords[1] - goal_coords[1])
                    total_manhattan_distance += manhattan_distance
                else:
                    total_manhattan_distance += 100 # Large penalty if location parsing fails

                if current_location != goal_location:
                    if self._is_corner_location(current_location, clear_locations, state):
                        corner_penalty += 10 # Add penalty for boxes in corners not at goal

        return total_manhattan_distance + corner_penalty


    def _get_parts(self, fact):
        """Utility function to split a PDDL fact string into parts."""
        return fact[1:-1].split()

    def _parse_location_coords(self, location_name):
        """Parse location coordinates from location name 'loc_r_c'."""
        match = re.match(r'loc_(\d+)_(\d+)', location_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None # Return None if parsing fails

    def _is_corner_location(self, location, clear_locations, state):
        """Check if a location is a corner location."""
        if location not in self.adjacent_locations: # Handle locations with no adjacency info (e.g., walls if represented)
            return False

        adjacent_clear_count = 0
        for adj_loc in self.adjacent_locations[location]:
            if adj_loc in clear_locations:
                adjacent_clear_count += 1
        return adjacent_clear_count <= 2


# Dummy Task class for goal_reached check - as the provided code does not include Task class in the scope.
class DummyTask:
    def __init__(self, goals):
        self.goals = goals
    def goal_reached(self, state):
        return self.goals <= state

task = DummyTask(frozenset()) # Dummy task instance for the heuristic to use the goal_reached check (even though it's not actually used in the heuristic calculation itself, but for the initial 0 heuristic check).
