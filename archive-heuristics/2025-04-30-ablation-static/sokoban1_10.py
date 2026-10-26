# from fnmatch import fnmatch # Not used in this specific heuristic logic
from heuristics.heuristic_base import Heuristic
# No other specific modules like math are needed for abs().

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the Manhattan distances
    of all boxes to their respective goal locations and adding the Manhattan distance
    from the robot's current location to the nearest box that is not yet at its goal.
    This is a non-admissible heuristic designed to guide a greedy best-first search.

    # Assumptions
    - The goal state is defined by the locations of the boxes.
    - The grid structure and connectivity are defined by the 'adjacent' predicates.
    - Locations are named in the format 'loc_row_col', allowing for Manhattan distance calculation.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic ignores obstacles (walls, other boxes) and the specific side the robot
      needs to be on to push a box.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Parses all relevant location names (from initial state, goals, and adjacent facts)
      to extract their row and column coordinates, storing them in a dictionary
      for efficient distance calculation. This mapping is based on the assumption
      that location names follow the 'loc_row_col' format.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task definition (`self.goal_locations`).
    2. In the current state, find the current location of the robot and each box.
    3. Initialize the total heuristic value (`total_heuristic`) to 0.
    4. Initialize a list (`boxes_to_move_locations`) to store the current locations of boxes
       that are not yet at their goal.
    5. Iterate through each box that has a defined goal location:
       a. Find the box's current location in the state.
       b. If the box is found and is not currently at its goal location:
          i. Calculate the Manhattan distance between the box's current location and its goal location using the pre-parsed coordinates. This distance serves as a simple estimate of the minimum number of 'push' actions required for this box.
          ii. Add this Manhattan distance to `total_heuristic`.
          iii. Add the box's current location name to the `boxes_to_move_locations` list.
    6. After processing all boxes, if the robot's location was found and there are boxes
       in the `boxes_to_move_locations` list:
       a. Calculate the Manhattan distance from the robot's current location to each location
          in the `boxes_to_move_locations` list.
       b. Find the minimum of these distances.
       c. Add this minimum distance to `total_heuristic`. This estimates the cost for the robot
          to reach the vicinity of the nearest box it needs to interact with.
    7. Return the final `total_heuristic` value. This value will be 0 if and only if all
       boxes are already at their goal locations (assuming the task goal only specifies
       box locations, as seen in the examples).

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and location coordinates."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Example goal: (at box1 loc_2_4)
            parts = self._get_parts(goal)
            if parts and parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                location_name = parts[2]
                self.goal_locations[box_name] = location_name

        # Extract coordinates for all locations mentioned in the problem
        self.location_coords = {}
        # Locations appear in initial state (robot, boxes, clear), goals (boxes), and adjacent facts.
        # Parsing adjacent facts usually covers all grid locations.
        # Also parse locations from initial state and goals just in case.
        all_relevant_facts = set(task.initial_state) | set(task.goals) | set(self.static_facts)

        for fact in all_relevant_facts:
             parts = self._get_parts(fact)
             if parts:
                 if parts[0] == 'adjacent' and len(parts) == 4:
                     loc1 = parts[1]
                     loc2 = parts[2]
                     if loc1 not in self.location_coords:
                         self.location_coords[loc1] = self._parse_location(loc1)
                     if loc2 not in self.location_coords:
                         self.location_coords[loc2] = self._parse_location(loc2)
                 elif parts[0] in ['at-robot', 'at', 'clear'] and len(parts) >= 2:
                     # Handle facts like (at-robot loc_X_Y), (at box1 loc_X_Y), (clear loc_X_Y)
                     # The location is typically the last argument
                     loc = parts[-1]
                     if loc not in self.location_coords:
                          self.location_coords[loc] = self._parse_location(loc)


    def _get_parts(self, fact):
        """Helper to extract parts from a PDDL fact string."""
        # Remove parentheses and split by space
        if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
             # Handle unexpected input format gracefully
             return []
        return fact[1:-1].split()

    def _parse_location(self, loc_name):
        """Helper to parse loc_X_Y string into (X, Y) tuple."""
        # Example: 'loc_6_4' -> ('loc', '6', '4') -> (6, 4)
        parts = loc_name.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                # Handle unexpected format if necessary
                # print(f"Warning: Could not parse location string {loc_name} into integers.")
                return None
        # else:
             # print(f"Warning: Unexpected location string format {loc_name}")
        return None


    def _manhattan_distance(self, loc1_name, loc2_name):
        """Calculate Manhattan distance between two location names."""
        coords1 = self.location_coords.get(loc1_name)
        coords2 = self.location_coords.get(loc2_name)

        if coords1 is None or coords2 is None:
            # This indicates a location name was not parsed correctly during init
            # print(f"Error: Could not find coordinates for {loc1_name} or {loc2_name}")
            return float('inf') # Return infinity for unknown locations

        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        total_box_goal_distance = 0
        current_box_locations = {}
        robot_location = None

        # Find current locations of robot and boxes
        for fact in state:
            parts = self._get_parts(fact)
            if parts:
                if parts[0] == 'at-robot' and len(parts) == 2:
                    robot_location = parts[1]
                elif parts[0] == 'at' and len(parts) == 3 and parts[1] in self.goal_locations:
                     box_name = parts[1]
                     location_name = parts[2]
                     current_box_locations[box_name] = location_name

        boxes_to_move_locations = []
        for box_name, goal_loc_name in self.goal_locations.items():
            current_loc_name = current_box_locations.get(box_name)

            # If a box with a goal is not found in the current state's 'at' facts,
            # it implies an invalid state or the box is inside something (not applicable in Sokoban).
            # In Sokoban, boxes are always 'at' a location.
            if current_loc_name is None:
                 # This indicates an issue with state representation or parsing
                 # print(f"Warning: Box {box_name} with goal {goal_loc_name} not found in state.")
                 continue # Skip this box, or handle as an error state

            if current_loc_name != goal_loc_name:
                # Distance for the box to reach its goal
                box_goal_distance = self._manhattan_distance(current_loc_name, goal_loc_name)
                total_box_goal_distance += box_goal_distance
                boxes_to_move_locations.append(current_loc_name)

        # Add distance for the robot to reach the nearest box that needs moving
        # This is a simplified estimate of the robot's contribution to the cost.
        robot_distance_cost = 0
        if robot_location and boxes_to_move_locations:
            min_robot_to_box_dist = float('inf')
            for box_loc in boxes_to_move_locations:
                 dist = self._manhattan_distance(robot_location, box_loc)
                 min_robot_to_box_dist = min(min_robot_to_box_dist, dist)
            if min_robot_to_box_dist != float('inf'):
                 robot_distance_cost = min_robot_to_box_dist

        # The heuristic is the sum of box-goal distances plus the cost for the robot
        # to reach the nearest box that needs moving.
        # This heuristic is 0 if and only if all boxes are at their goal locations,
        # which corresponds to the goal state definition in the provided examples.
        return total_box_goal_distance + robot_distance_cost
