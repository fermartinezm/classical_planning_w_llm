# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the framework
from fnmatch import fnmatch
import math
from collections import deque

# Define a dummy Heuristic base class if not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location_name(loc_name):
    """Parses 'loc_row_col' into (row, col) tuple."""
    try:
        parts = loc_name.split('_')
        # Assuming format loc_row_col
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # Handle unexpected location name format - log or raise?
        # For a heuristic, maybe return None or a default invalid coordinate
        # print(f"Warning: Could not parse location name {loc_name}")
        return None

def manhattan_distance(loc1, loc2, location_coords):
    """Calculates Manhattan distance between two locations."""
    coords1 = location_coords.get(loc1)
    coords2 = location_coords.get(loc2)
    if coords1 is None or coords2 is None:
        # This indicates an issue with location parsing or mapping
        return float('inf') # Should ideally not happen in valid problems
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

def opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen for valid directions

def robot_bfs_distance(start_loc, goal_locs, obstacle_locations, adjacency_list):
    """
    Performs BFS to find the shortest path distance from start_loc to any goal in goal_locs,
    avoiding obstacle_locations.
    Returns float('inf') if no goal is reachable.
    """
    if not goal_locs:
        return float('inf') # No valid push positions to reach

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    # Convert goal_locs list/set to a set for efficient lookup
    goal_locs_set = set(goal_locs)

    while queue:
        current_loc, dist = queue.popleft() # Use deque for efficient popleft

        if current_loc in goal_locs_set:
            return dist

        # Neighbors are locations adjacent to current_loc
        neighbors_info = adjacency_list.get(current_loc, [])
        for next_loc, _ in neighbors_info: # We only need the location name
            if next_loc not in visited and next_loc not in obstacle_locations:
                visited.add(next_loc)
                queue.append((next_loc, dist + 1))

    return float('inf') # Goal not reachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum, over all boxes not at their goal,
    of (Manhattan distance from box to goal + minimum robot distance
    to a position from which the box can be pushed towards the goal).
    Includes a penalty for boxes that cannot be pushed towards their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Builds graph representation and location coordinate mapping.
        """
        # Store goal locations for each box.
        self.goal_locations = {} # Map box name to its goal location name
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build graph representation and location coordinate mapping from static facts.
        self.location_coords = {} # Map location name to (row, col)
        self.adjacency_list = {} # Map location name to list of (neighbor_loc, direction_to_neighbor)
        self.reverse_adjacency_list = {} # Map location name to list of (neighbor_loc, direction_from_neighbor)

        # Collect all locations mentioned in static facts and parse coordinates
        all_locations_set = set()
        for fact in task.static:
             parts = get_parts(fact)
             if not parts: continue

             if parts[0] == "adjacent":
                 # Fact is (adjacent loc1 loc2 dir)
                 if len(parts) == 4:
                     _, loc1, loc2, dir = parts
                     all_locations_set.add(loc1)
                     all_locations_set.add(loc2)

                     # Build adjacency lists
                     if loc1 not in self.adjacency_list: self.adjacency_list[loc1] = []
                     if loc2 not in self.adjacency_list: self.adjacency_list[loc2] = []
                     if loc1 not in self.reverse_adjacency_list: self.reverse_adjacency_list[loc1] = []
                     if loc2 not in self.reverse_adjacency_list: self.reverse_adjacency_list[loc2] = []

                     self.adjacency_list[loc1].append((loc2, dir))
                     self.adjacency_list[loc2].append((loc1, opposite_direction(dir)))

                     # reverse_adjacency_list helps find the location *behind* a box
                     # If loc1 -> loc2 is 'dir', then loc1 is behind loc2 relative to 'dir' push
                     # So, loc1 is a neighbor of loc2, and the direction *from* loc1 *to* loc2 is 'dir'
                     self.reverse_adjacency_list[loc2].append((loc1, dir))
                     self.reverse_adjacency_list[loc1].append((loc2, opposite_direction(dir)))

        # Also collect locations from initial state and goals to ensure all are mapped
        # (Though typically adjacent facts cover the relevant grid)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ["at-robot", "at", "clear"]:
                 if len(parts) > 1:
                     all_locations_set.add(parts[-1]) # Location is usually the last argument

        for goal in task.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "at":
                 if len(parts) > 2:
                     all_locations_set.add(parts[2]) # Location is the third argument

        # Parse coordinates for all collected locations
        for loc in all_locations_set:
            coords = parse_location_name(loc)
            if coords is not None:
                self.location_coords[loc] = coords

        # Store the set of all known locations for obstacle calculation
        self.all_locations = set(self.location_coords.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current state information
        robot_location = None
        box_locations = {} # Map box name to current location name
        clear_locations = set() # Set of location names that are clear

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at-robot":
                if len(parts) > 1: robot_location = parts[1]
            elif predicate == "at":
                if len(parts) > 2:
                    box, location = parts[1], parts[2]
                    box_locations[box] = location
            elif predicate == "clear":
                if len(parts) > 1: clear_locations.add(parts[1])

        total_heuristic_cost = 0

        # Calculate obstacles for robot movement BFS
        # Obstacles are locations that are NOT clear AND are NOT the robot's current location
        # The robot's current location is traversable by the robot.
        obstacle_locations = self.all_locations - clear_locations
        if robot_location in obstacle_locations:
             obstacle_locations.remove(robot_location)


        for box, goal_loc in self.goal_locations.items():
            current_loc_b = box_locations.get(box) # Get box's current location

            if current_loc_b is None:
                 # Box not found in state facts? Should not happen in valid states.
                 # print(f"Warning: Box {box} not found in state.")
                 continue # Skip this box

            if current_loc_b == goal_loc:
                # Box is already at its goal, contributes 0 to heuristic
                continue

            # Box is not at its goal
            dist_b_g = manhattan_distance(current_loc_b, goal_loc, self.location_coords)

            # Find valid push positions for this box towards its goal
            push_positions = self._find_push_positions(
                current_loc_b, goal_loc, clear_locations
            )

            if not push_positions:
                # Box is in a position where it cannot be pushed towards the goal
                # (e.g., blocked by wall/other box, or target cell not clear, or no move reduces distance)
                # Assign a penalty. This state is likely bad.
                # Penalty = distance + a large constant.
                box_heuristic_cost = dist_b_g + 1000 # Arbitrary large penalty
            else:
                # Calculate minimum robot distance to any of the valid push positions
                # Need robot_location for BFS start
                if robot_location is None:
                    # Robot location not found in state? Should not happen.
                    # print("Warning: Robot location not found in state.")
                    min_robot_dist = math.inf # Cannot move robot
                else:
                    min_robot_dist = robot_bfs_distance(
                        robot_location, push_positions, obstacle_locations, self.adjacency_list
                    )

                if min_robot_dist == math.inf:
                    # Robot cannot reach any valid push position for this box
                    box_heuristic_cost = dist_b_g + 1000 # Penalty
                else:
                    # Heuristic for this box:
                    # Minimum pushes needed (Manhattan distance) +
                    # Minimum robot moves to get into position for the *first* push.
                    box_heuristic_cost = dist_b_g + min_robot_dist

            total_heuristic_cost += box_heuristic_cost

        return total_heuristic_cost

    def _find_push_positions(self, loc_b, loc_g, clear_locations):
        """
        Finds locations adjacent to loc_b from which the robot can push
        the box towards loc_g into a clear cell.
        Returns a list of location names.
        """
        push_positions = set()
        coords_b = self.location_coords.get(loc_b)
        coords_g = self.location_coords.get(loc_g)

        if coords_b is None or coords_g is None:
             # Should not happen if locations are parsed correctly
             return []

        (br, bc) = coords_b
        (gr, gc) = coords_g

        dist_b_g = abs(br - gr) + abs(bc - gc)

        # Iterate through potential next locations for the box (neighbors of loc_b)
        neighbors_of_b = self.adjacency_list.get(loc_b, [])
        for loc_b_next, dir_b_to_next in neighbors_of_b:
            # Check if the target location for the box is clear
            if loc_b_next in clear_locations:
                # Check if moving to loc_b_next reduces Manhattan distance to goal
                coords_b_next = self.location_coords.get(loc_b_next)
                if coords_b_next is not None:
                    (nr, nc) = coords_b_next
                    dist_b_next_g = abs(nr - gr) + abs(nc - gc)

                    # Strict inequality for non-admissibility and progress guarantee
                    if dist_b_next_g < dist_b_g:
                        # This is a valid step towards the goal.
                        # Find the robot position required for this push.
                        # The robot must be adjacent to loc_b in the opposite direction.
                        dir_push_to_b = opposite_direction(dir_b_to_next)
                        if dir_push_to_b:
                            # Find the location adjacent to loc_b in dir_push_to_b
                            potential_push_locs = self.reverse_adjacency_list.get(loc_b, [])
                            for loc_push, dir_neighbor_to_loc_b in potential_push_locs:
                                if dir_neighbor_to_loc_b == dir_push_to_b:
                                    push_positions.add(loc_push)
                                    break # Found the unique location behind loc_b

        return list(push_positions) # Return as list for robot_bfs_distance goal_locs

