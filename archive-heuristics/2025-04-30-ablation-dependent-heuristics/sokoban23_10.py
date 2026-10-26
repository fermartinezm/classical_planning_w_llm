import re
import math

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach a goal state by summing two components:
    1.  The sum of Manhattan distances for each box to its specific goal location.
        This estimates the minimum number of pushes required for each box independently.
    2.  The minimum Manhattan distance from the robot's current location to any box
        that is not yet at its goal location. This estimates the robot's effort
        to reach a box to start pushing.

    Assumptions:
    - The problem instance uses location names in the format 'loc_R_C' where R and C
      are integers representing row and column.
    - The goal state specifies a unique target location for each box using facts
      like '(at boxN loc_R_C)'.
    - The grid structure implied by 'loc_R_C' and 'adjacent' facts is a standard
      grid where adjacent locations are one step away horizontally or vertically.
      Manhattan distance is used as a proxy for path distance.
    - Deadlocks (situations where a box cannot be moved further) are not explicitly
      detected or penalized. This might lead to the heuristic underestimating
      the cost in deadlock situations, but it keeps the heuristic computation fast.

    Heuristic Initialization:
    The constructor processes the task's goal state to identify the target location
    for each box. It stores this mapping from box object to its goal location string.
    It also stores the mapping from location string ('loc_R_C') to integer coordinates (R, C).

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state. If yes, the heuristic is 0.
    2.  Initialize the total heuristic value to 0.
    3.  Identify the robot's current location string from the state facts.
    4.  Identify the current location string of each box from the state facts.
    5.  Initialize a variable `min_robot_to_box_dist` to infinity.
    6.  Initialize a counter `boxes_not_at_goal_count` to 0.
    7.  Iterate through each box that has a defined goal location (based on `self.box_goals`):
        a.  Get the box's name (`box_name`), its current location string (`box_loc_str`),
            and its goal location string (`goal_loc_str`).
        b.  If the box is already at its goal location (`box_loc_str == goal_loc_str`),
            continue to the next box.
        c.  Increment `boxes_not_at_goal_count`.
        d.  Parse `box_loc_str` and `goal_loc_str` into coordinates `(r1, c1)` and `(r2, c2)`
            using the pre-calculated `location_coords` map. Handle cases where locations
            are not found in the map (e.g., return infinity).
        e.  Calculate the Manhattan distance: `dist = abs(r1 - r2) + abs(c1 - c2)`.
        f.  Add this distance to the `total_box_goal_distance`.
        g.  Parse the robot's current location string (`robot_loc_str`) into coordinates `(rr, rc)`.
            Handle cases where the robot location is not found (e.g., return infinity).
        h.  Calculate the Manhattan distance from the robot to the current box location:
            `robot_to_box_dist = abs(rr - r1) + abs(rc - c1)`.
        i.  Update `min_robot_to_box_dist = min(min_robot_to_box_dist, robot_to_box_dist)`.
    8.  After iterating through all boxes, if `boxes_not_at_goal_count > 0`:
        a.  The total heuristic is `total_box_goal_distance + min_robot_to_box_dist`.
    9.  If `boxes_not_at_goal_count == 0` (all boxes are at goals), the total heuristic is 0.
        (This case is covered by the initial goal check).
    10. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating necessary information
        from the task definition.

        @param task: The planning task object.
        """
        self.task = task
        self.box_goals = {}
        self.location_coords = {} # Map location string to (row, col) tuple

        # Extract goal locations for each box
        # Assuming goals are of the form (at boxN loc_R_C)
        for goal_fact in task.goals:
            match = re.match(r"\(at (\S+) (\S+)\)", goal_fact)
            if match:
                box_name = match.group(1)
                loc_name = match.group(2)
                self.box_goals[box_name] = loc_name

        # Collect all unique location names from initial state, goals, and static facts
        all_locations = set()
        # Locations from initial state
        for fact in task.initial_state:
             parts = fact.split()
             if len(parts) > 1:
                 if fact.startswith('(at '):
                     if len(parts) > 2:
                         all_locations.add(parts[2][:-1])
                 elif fact.startswith('(at-robot '):
                      all_locations.add(parts[1][:-1])
                 elif fact.startswith('(clear '):
                      all_locations.add(parts[1][:-1])


        # Locations from goals
        for goal_fact in task.goals:
             parts = goal_fact.split()
             if goal_fact.startswith('(at '):
                 if len(parts) > 2:
                     all_locations.add(parts[2][:-1])

        # Locations from static facts (adjacent)
        for static_fact in task.static:
             parts = static_fact.split()
             if static_fact.startswith('(adjacent '):
                 if len(parts) > 2:
                     all_locations.add(parts[1])
                 if len(parts) > 3:
                     all_locations.add(parts[2][:-1]) # Remove ')'


        # Parse coordinates for each location name
        for loc_name in all_locations:
            coords = self._parse_location(loc_name)
            if coords is not None:
                self.location_coords[loc_name] = coords
            # else: location name is not in loc_R_C format, ignore or handle differently if needed

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal. Returns infinity if
                 essential location information is missing.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        total_box_goal_distance = 0
        robot_loc_str = None
        box_locations = {} # box_name -> loc_str

        # Find robot and box locations in the current state
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc_str = fact.split()[1][:-1] # Remove ')'
            elif fact.startswith('(at '):
                match = re.match(r"\(at (\S+) (\S+)\)", fact)
                if match:
                    box_name = match.group(1)
                    loc_name = match.group(2)
                    box_locations[box_name] = loc_name

        # Ensure robot location is found and has coordinates
        if robot_loc_str is None:
             # Robot location not found in state - invalid state?
             return float('inf')
        robot_coords = self.location_coords.get(robot_loc_str)
        if robot_coords is None:
             # Robot location string found but coordinates not parsed during init
             # (e.g., not in loc_R_C format or not seen in init/goals/static)
             return float('inf')


        min_robot_to_box_dist = float('inf')
        boxes_not_at_goal_count = 0

        # Calculate box-to-goal distances and find closest box to robot
        # Iterate through the boxes we know goals for
        for box_name, goal_loc_str in self.box_goals.items():
            box_loc_str = box_locations.get(box_name)

            # Ensure box location is found in state
            if box_loc_str is None:
                 # Box with a goal is not present in the current state - invalid state?
                 return float('inf')

            # Check if box is already at goal
            if box_loc_str == goal_loc_str:
                continue # This box is done

            boxes_not_at_goal_count += 1

            # Calculate box-to-goal distance (Manhattan)
            box_coords = self.location_coords.get(box_loc_str)
            goal_coords = self.location_coords.get(goal_loc_str)

            # Ensure both locations have coordinates
            if box_coords is None or goal_coords is None:
                 # Location string found but coordinates not parsed during init
                 return float('inf')

            box_goal_dist = abs(box_coords[0] - goal_coords[0]) + abs(box_coords[1] - goal_coords[1])
            total_box_goal_distance += box_goal_dist

            # Calculate robot-to-box distance (Manhattan)
            robot_to_box_dist = abs(robot_coords[0] - box_coords[0]) + abs(robot_coords[1] - box_coords[1])
            min_robot_to_box_dist = min(min_robot_to_box_dist, robot_to_box_dist)

        # Add robot cost if there are boxes not at goal
        if boxes_not_at_goal_count > 0:
            # The robot needs to reach a position near a box to push it.
            # The minimum distance to any box needing a push is a reasonable
            # estimate for the robot's contribution to the remaining cost.
            total_heuristic = total_box_goal_distance + min_robot_to_box_dist
        else:
            # All boxes are at goal (this case should be caught by the initial check)
            total_heuristic = 0

        return total_heuristic

    def _parse_location(self, loc_str):
        """Helper to parse 'loc_R_C' into (R, C) tuple."""
        if not isinstance(loc_str, str):
            return None
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                # PDDL locations are typically 1-indexed. Keep them as (R, C)
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                return None # Malformed location string (R or C not integer)
        return None # Not in expected 'loc_R_C' format
