# The provided helper functions and class structure are assumed to be available
# in the environment where this heuristic is used.
# Specifically, Heuristic base class and the task representation.

from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
from math import inf

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    return fact[1:-1].split()

# Helper function to match fact parts with patterns
def match(fact, *args):
    """Checks if the parts of a fact match the given patterns."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse location string 'loc_R_C' into (R, C)
def parse_location(location_str):
    """Parses a location string like 'loc_R_C' into a (row, col) tuple."""
    try:
        # Expecting format like 'loc_1_1'
        parts = location_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
             return int(parts[1]), int(parts[2])
        else:
             # Handle unexpected format
             return None, None
    except ValueError:
        # Handle cases where R or C are not integers
        return None, None
    except Exception:
        # Catch any other unexpected errors during parsing
        return None, None


# Helper function for BFS
def bfs(start_loc, target_locs, traversable_locations, adj_list):
    """
    Performs BFS to find the shortest distance from start_loc to any target_loc
    within the traversable_locations.
    Returns inf if no target is reachable.
    """
    # If start is one of the targets, distance is 0
    if start_loc in target_locs:
        return 0

    # The BFS explores locations reachable *from* start_loc.
    # The start_loc itself doesn't need to be in traversable_locations
    # for the initial state, but all subsequent steps must land in traversable_locations.
    # Assuming start_loc is always a valid location in the grid.

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Check adjacent locations
        # Use adj_list.get(current_loc, []) to handle locations with no outgoing edges
        for neighbor_loc in adj_list.get(current_loc, []):
            # Robot can only move to locations that are clear (traversable)
            if neighbor_loc in traversable_locations and neighbor_loc not in visited:
                if neighbor_loc in target_locs:
                    return dist + 1 # Found a target

                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return inf # No target reachable

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of Manhattan distances for each box from its current location to its goal location.
       This component estimates the minimum number of push actions required for all boxes.
    2. The shortest path distance for the robot from its current location to any *clear* location
       adjacent to any box that is not yet at its goal. This component estimates the robot's
       effort to get into a position to perform the next push action on some box. The robot
       can only move through clear locations.

    Assumptions:
    - Location names follow the format 'loc_R_C' where R and C are integers.
    - The grid connectivity is defined by 'adjacent' predicates in the static facts.
    - The goal state is defined by '(at ?b ?l)' predicates for all boxes.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - Assumes the grid defined by 'adjacent' facts is consistent and covers all relevant locations.

    Heuristic Initialization:
    - Parses the goal facts to create a mapping from each box to its target goal location.
    - Parses the static 'adjacent' facts to build an adjacency list representation of the grid,
      mapping each location to its adjacent locations. Also builds a reverse adjacency list
      to quickly find locations adjacent *to* a given location. Collects all unique location names.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract the robot's current location from the state.
    2. Extract the current location of each box from the state.
    3. Identify all locations that are currently occupied by the robot or boxes.
    4. Determine the set of clear locations (all locations minus occupied locations).
    5. Initialize the total heuristic value `h` to 0.
    6. Create a set `potential_robot_target_locations` to store locations adjacent to unsolved boxes.
    7. Iterate through each box defined in the goals:
       a. Get the box's current location from the state and its goal location (from initialization).
       b. If the box is not at its goal location:
          i. Parse the current and goal locations into (row, col) coordinates.
          ii. Calculate the Manhattan distance between the current and goal coordinates.
          iii. Add this Manhattan distance to `h`. This estimates the pushes needed for this box.
          iv. Find locations adjacent *to* the box's current location using the reverse adjacency list. Add these adjacent locations to the `potential_robot_target_locations` set. These are potential spots for the robot to be *before* pushing the box.
    8. Identify the set of actual robot target locations for BFS: these are the locations from `potential_robot_target_locations` that are also currently clear.
    9. If the set of actual robot target locations is empty:
       a. If `h > 0` (meaning there are unsolved boxes), it implies the robot cannot reach any clear location adjacent to an unsolved box. This state is likely unsolvable. Return infinity.
       b. If `h == 0` (meaning all boxes are at goals), the heuristic is 0. Return `h`.
    10. If the set of actual robot target locations is not empty:
        a. Calculate the shortest path distance for the robot from its current location to any location in the set of actual robot target locations. This BFS considers only moves to locations that are currently clear.
        b. If a path is found, add the shortest distance to `h`. This estimates the robot's cost to reach a position from which it can push an unsolved box.
        c. If no path is found (distance is infinity), it suggests the robot cannot reach any *clear* location adjacent to an unsolved box via clear cells. Return infinity.
    11. Return the final calculated value of `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goals: map box to goal location
        self.box_goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at boxN loc_R_C)
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3: # Ensure correct number of parts
                    _, box, location = parts
                    self.box_goal_locations[box] = location


        # 2. Parse static facts: build adjacency list and reverse adjacency list for grid graph
        self.adj_list = {}
        self.rev_adj_list = {}
        self.all_locations = set()
        for fact in static_facts:
            # Adjacent facts are (adjacent loc1 loc2 direction)
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 4: # Ensure correct number of parts
                    _, loc1, loc2, _ = parts

                    if loc1 not in self.adj_list:
                        self.adj_list[loc1] = []
                    self.adj_list[loc1].append(loc2)

                    if loc2 not in self.rev_adj_list:
                        self.rev_adj_list[loc2] = []
                    self.rev_adj_list[loc2].append(loc1)

                    self.all_locations.add(loc1)
                    self.all_locations.add(loc2)


    def __call__(self, node):
        state = node.state

        # 1. Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, robot_location = parts
                    break
        if robot_location is None:
             # Should not happen in a valid Sokoban state, but handle defensively
             return inf # Robot location unknown

        # 2. Find current box locations and occupied locations
        current_box_locations = {}
        occupied_locations = {robot_location}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, box, location = parts
                    current_box_locations[box] = location
                    occupied_locations.add(location)


        # 3. Determine clear locations
        # A location is clear if it's not occupied by the robot or a box.
        clear_locations = {loc for loc in self.all_locations if loc not in occupied_locations}

        # 4. Calculate box-goal distances and identify potential robot target locations (adjacent to unsolved boxes)
        h = 0
        potential_robot_target_locations = set()

        # Iterate through boxes defined in goals to ensure we consider all relevant boxes
        for box, goal_location in self.box_goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box defined in the goal is not found in the current state,
            # something is wrong or the state is unreachable/invalid.
            if current_location is None:
                 return inf

            if current_location != goal_location:
                # Calculate Manhattan distance for box
                curr_row, curr_col = parse_location(current_location)
                goal_row, goal_col = parse_location(goal_location)

                if curr_row is not None and goal_row is not None:
                    box_dist = abs(curr_row - goal_row) + abs(curr_col - goal_col)
                    h += box_dist
                else:
                    # Handle parsing error - indicates invalid location format
                    return inf

                # Add locations adjacent *to* the box's current location as potential robot targets.
                # These are potential spots for the robot to be *before* pushing the box.
                # We use rev_adj_list to find locations L' such that L' -> current_location is an edge.
                if current_location in self.rev_adj_list:
                    potential_robot_target_locations.update(self.rev_adj_list[current_location])


        # 5. Calculate robot distance to closest *clear* target location
        # The robot needs to reach a location adjacent to an unsolved box *that is clear*.
        # The target locations for BFS are the subset of potential targets that are actually clear.
        actual_robot_target_locations = potential_robot_target_locations & clear_locations

        if not actual_robot_target_locations:
            # This happens if all boxes are at goals (h=0), or if unsolved boxes exist (h>0)
            # but none of the locations adjacent to them are clear or reachable by the robot.
            # If h > 0, it's likely a deadlock/unsolvable state.
            if h > 0:
                 return inf
            else: # h == 0, means all boxes are at goals
                 pass # h is already 0, which is correct.

        else:
            # Perform BFS from robot location to find distance to any actual_robot_target_location
            # Robot can only move to clear locations.
            min_robot_dist = bfs(robot_location, actual_robot_target_locations, clear_locations, self.adj_list)

            if min_robot_dist == inf:
                # Robot cannot reach any clear location adjacent to an unsolved box via clear cells
                return inf # State is likely a deadlock or unsolvable

            h += min_robot_dist

        return h
