import re

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing the Manhattan
    distances of each box to its goal location and adding the minimum
    Manhattan distance from the robot to any box that is not yet at its goal.
    This heuristic is not admissible but aims to guide a greedy best-first
    search efficiently.

    Assumptions:
    - Location names follow the format 'loc_row_col' where row and col are integers.
    - The goal state specifies a unique target location for each box.
    - The grid derived from location names provides a reasonable approximation
      of traversable distance, ignoring dynamic obstacles (other boxes, robot)
      and static obstacles (walls implied by lack of adjacency).

    Heuristic Initialization:
    The constructor processes the static information from the task:
    1. Extracts the goal location for each box from the task's goal facts.
    2. Parses all unique location names found in static, initial state, and
       goal facts to create a mapping from location name strings to (row, col)
       integer tuples. This mapping is used for calculating Manhattan distances.

    Step-By-Step Thinking for Computing Heuristic:
    The __call__ method computes the heuristic value for a given state:
    1. Identify the robot's current location from the state facts.
    2. Identify the current location of each box from the state facts.
    3. Initialize the total heuristic value (h) to 0.
    4. Initialize the minimum distance from the robot to an off-goal box
       (min_robot_to_box_dist) to infinity.
    5. Iterate through each box that has a specified goal location:
       a. Get the box's current location and its goal location.
       b. If the box exists in the current state and is not at its goal location:
          i. Calculate the Manhattan distance between the box's current location
             and its goal location using the pre-computed coordinates. Add this
             distance to h. This represents the estimated pushes needed for this box.
          ii. Calculate the Manhattan distance between the robot's current location
              and the box's current location. Update min_robot_to_box_dist with
              the minimum distance found so far across all off-goal boxes.
    6. If there was at least one box not at its goal location, add
       min_robot_to_box_dist to h. This accounts for the robot needing to reach
       a box to start pushing.
    7. Return the final value of h.
    """

    def __init__(self, task):
        # Extract box goals
        self.box_goals = {}
        for fact in task.goals:
            # Goal facts are like '(at box1 loc_2_4)'
            match = re.match(r"\(at (\w+) (\w+)\)", fact)
            if match:
                box_name = match.group(1)
                loc_name = match.group(2)
                self.box_goals[box_name] = loc_name

        # Extract all location names and their coordinates
        self.location_coords = {}
        all_locations = set()

        # Locations from goals (at facts)
        for fact in task.goals:
             match = re.match(r"\(at (\w+) (\w+)\)", fact)
             if match:
                 all_locations.add(match.group(2)) # Add location

        # Locations from initial state (at-robot and at facts)
        for fact in task.initial_state:
             match_robot = re.match(r"\(at-robot (\w+)\)", fact)
             if match_robot:
                 all_locations.add(match_robot.group(1)) # Add robot location
             match_at = re.match(r"\(at (\w+) (\w+)\)", fact)
             if match_at:
                 all_locations.add(match_at.group(2)) # Add box location

        # Locations from static facts (adjacent)
        for fact in task.static:
            match = re.match(r"\(adjacent (\w+) (\w+) (\w+)\)", fact)
            if match:
                all_locations.add(match.group(1))
                all_locations.add(match.group(2))

        # Parse coordinates for all found locations
        for loc_name in all_locations:
            coords = self._parse_location(loc_name)
            if coords:
                self.location_coords[loc_name] = coords
            # else: Warning already handled in _parse_location

    def _parse_location(self, loc_string):
        """Parses a location string 'loc_row_col' into a (row, col) tuple."""
        parts = loc_string.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                return None
        return None

    def _manhattan_distance(self, loc1_str, loc2_str):
        """Calculates Manhattan distance between two location strings."""
        coords1 = self.location_coords.get(loc1_str)
        coords2 = self.location_coords.get(loc2_str)

        if coords1 is None or coords2 is None:
            return float('inf')

        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @param task: The planning task object (contains goals, static info etc.).
                     Note: self already stores precomputed info from task.
        @return: The estimated cost (heuristic value).
        """
        robot_location = None
        current_box_locations = {}

        # Extract dynamic facts from the state
        for fact in state:
            if fact.startswith('(at-robot '):
                match = re.match(r"\(at-robot (\w+)\)", fact)
                if match:
                    robot_location = match.group(1)
            elif fact.startswith('(at '):
                match = re.match(r"\(at (\w+) (\w+)\)", fact)
                if match:
                    box_name = match.group(1)
                    loc_name = match.group(2)
                    current_box_locations[box_name] = loc_name

        # Should always have a robot location in a valid state
        if robot_location is None:
             return float('inf') # Invalid state

        h = 0
        min_robot_to_box_dist = float('inf')
        off_goal_boxes_exist = False

        # Calculate box-to-goal distances and find closest off-goal box
        for box_name, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box_name)

            # Ensure the box exists in the current state and is not at its goal
            if current_loc is not None and current_loc != goal_loc:
                off_goal_boxes_exist = True

                # Add box-to-goal distance
                box_dist = self._manhattan_distance(current_loc, goal_loc)
                h += box_dist

                # Find robot distance to this box
                robot_dist = self._manhattan_distance(robot_location, current_loc)
                min_robot_to_box_dist = min(min_robot_to_box_dist, robot_dist)

        # Add robot-to-closest-off-goal-box distance if there are off-goal boxes
        if off_goal_boxes_exist:
             # The robot needs to reach *a* box to push it.
             # The cost is at least the distance to the closest one.
             h += min_robot_to_box_dist
        # else: h remains 0 if all boxes are at goals (goal state)

        return h
