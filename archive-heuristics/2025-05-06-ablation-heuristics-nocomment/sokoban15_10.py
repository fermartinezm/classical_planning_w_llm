# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Splits a PDDL fact string into predicate and arguments."""
    # Assumes fact is like '(predicate arg1 arg2)'
    return fact[1:-1].split()

# Helper function to parse location names
def parse_location(loc_name):
    """Parses location names like 'loc_X_Y' into (X, Y) coordinates."""
    parts = loc_name.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            # Assuming loc_X_Y means row X, column Y
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Should not happen with standard PDDL generation
            return None
    return None

# Helper function for Manhattan distance
def manhattan_distance(loc1_name, loc2_name, location_coords):
    """Calculates Manhattan distance between two locations using pre-parsed coordinates."""
    coords1 = location_coords.get(loc1_name)
    coords2 = location_coords.get(loc2_name)
    if coords1 is None or coords2 is None:
        # This indicates an issue if locations are expected to be in location_coords
        # For robustness, return infinity or handle error.
        # Assuming all relevant locations are in location_coords.
        return float('inf')
    x1, y1 = coords1
    x2, y2 = coords2
    return abs(x1 - x2) + abs(y1 - y2)

# Need to import Heuristic base class and math
from heuristics.heuristic_base import Heuristic
import math # For float('inf')


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs for each box that is not yet at its goal location.
        For each misplaced box, the estimated cost is the sum of:
        1. The Manhattan distance between the box's current location and its goal location
           (representing the minimum number of pushes required if the path were clear).
        2. The minimum Manhattan distance between the robot's current location and
           any location from which the robot could push the box one step directly
           towards its goal location. This estimates the robot's effort to get into
           the necessary pushing position for the first step. If no such valid
           pushing location exists in the grid for any direction towards the goal,
           a large penalty (infinity) is assigned for this box.
        The total heuristic is the sum of these costs for all misplaced boxes.

    Assumptions:
        - Location names follow the format 'loc_X_Y' where X and Y are integers
          representing grid coordinates (row and column).
        - The goal is defined by a set of '(at box location)' facts.
        - The state representation includes facts for robot location ('at-robot')
          and box locations ('at').
        - The grid structure is defined by 'adjacent' facts, and all relevant
          locations are part of this grid.
        - The heuristic uses Manhattan distance, ignoring obstacles (other boxes,
          walls/non-clear cells) for both box and robot movement estimates, which
          makes it non-admissible but fast.
        - The heuristic does not explicitly detect or penalize complex deadlock states
          beyond the simple case where no valid pushing location towards the goal
          exists in the grid.

    Heuristic Initialization:
        The constructor processes the task's goals and static facts.
        - It extracts the goal location for each box from the task's goals and stores
          this mapping in `self.goal_locations`.
        - It parses all location names found in the static 'adjacent' facts (which
          define the grid structure) and stores their grid coordinates (row, col)
          in `self.location_coords`.
        - It creates an inverse mapping from coordinates (row, col) back to location
          names in `self.coords_location`. These mappings are used for Manhattan
          distance calculations and finding required robot locations.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state from the provided node.
        2. Initialize the total heuristic value `h` to 0.
        3. Identify the robot's current location by finding the fact `(at-robot ?l)` in the state.
        4. Create a dictionary mapping each box to its current location by finding facts
           `(at ?b ?l)` in the state for all relevant boxes (those with a goal location).
        5. Iterate through each box and its corresponding goal location stored in
           `self.goal_locations`.
        6. For the current box (`box_name`) and its goal location (`goal_loc_name`):
           a. Check if the goal fact `(at box_name goal_loc_name)` is present in the current state.
           b. If the goal fact is NOT present (meaning the box is misplaced):
              i. Find the box's current location (`current_loc_name`) from the dictionary created in step 4.
              ii. Calculate the Manhattan distance (`box_dist`) between `current_loc_name`
                  and `goal_loc_name` using `self.location_coords`. This estimates the
                  minimum pushes needed.
              iii. Get the coordinates `(bx, by)` for `current_loc_name` and `(gx, gy)`
                   for `goal_loc_name` from `self.location_coords`.
              iv. Initialize `min_robot_dist` to infinity.
              v. If the box's current row `bx` is different from the goal row `gx`:
                 # Vertical push towards goal
                 # Push Down (gx > bx): Robot needs to be at (bx - 1, by)
                 # Push Up (gx < bx): Robot needs to be at (bx + 1, by)
                 req_rx, req_ry = (bx - 1, by) if gx > bx else (bx + 1, by)
                 loc_req_v = self.coords_location.get((req_rx, req_ry))

                 if loc_req_v: # Check if this coordinate exists in the grid
                     robot_dist_v = manhattan_distance(robot_location_name, loc_req_v, self.location_coords)
                     min_robot_dist = min(min_robot_dist, robot_dist_v)

              vi. If the box's current column `by` is different from the goal column `gy`:
                 # Horizontal push towards goal
                 # Push Right (gy > by): Robot needs to be at (bx, by - 1)
                 # Push Left (gy < by): Robot needs to be at (bx, by + 1)
                 req_rx, req_ry = (bx, by - 1) if gy > by else (bx, by + 1)
                 loc_req_h = self.coords_location.get((req_rx, req_ry))

                 if loc_req_h: # Check if this coordinate exists in the grid
                     robot_dist_h = manhattan_distance(robot_location_name, loc_req_h, self.location_coords)
                     min_robot_dist = min(min_robot_dist, robot_dist_h)

              vii. If `min_robot_dist == float('inf')`:
                   # No valid push location towards goal found in the grid.
                   # This box might be stuck or requires complex setup.
                   # Assign a large penalty.
                   cost_for_box = float('inf')
              else:
                   # 6bviii. Cost is box distance + minimum robot distance to a push location
                   cost_for_box = box_dist + min_robot_dist

              # 6bix. Add to total heuristic
              h += cost_for_box

        # 7. Return total heuristic
        return h
