from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re # To parse location names

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_location(loc_name):
    """Parses a location string like 'loc_R_C' into a (row, col) tuple."""
    # This regex assumes location names are strictly in the format 'loc_R_C'
    # where R and C are digits.
    match = re.match(r'loc_(\d+)_(\d+)', loc_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # If location names can be different, this needs to be adjusted.
    # Based on the provided examples, 'loc_R_C' seems to be the standard.
    # Returning None indicates failure to parse this specific format.
    return None

def manhattan_distance(loc1_name, loc2_name, loc_map):
    """Calculates the Manhattan distance between two locations using precomputed coordinates."""
    coord1 = loc_map.get(loc1_name)
    coord2 = loc_map.get(loc2_name)

    # If coordinates for either location are not found (e.g., due to parsing failure),
    # return infinity to indicate an unreachable or invalid location for distance calculation.
    if coord1 is None or coord2 is None:
        return float('inf')

    r1, c1 = coord1
    r2, c2 = coord2
    return abs(r1 - r2) + abs(c1 - c2)

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated cost for each box to reach its goal location and
    the estimated cost for the robot to get into a position adjacent to an un-goaled box.
    It uses Manhattan distance as a simple grid distance approximation.

    # Assumptions
    - The grid structure is defined by 'adjacent' facts.
    - Location names follow the format 'loc_R_C' allowing Manhattan distance calculation.
    - The primary cost involves moving boxes to their goal locations.
    - The robot must be adjacent to a box to push it.
    - The heuristic is non-admissible and does not guarantee finding optimal solutions,
      nor does it explicitly detect dead-end states where boxes are irrevocably stuck.
    - All relevant locations in the problem instance (initial state, goals, adjacent facts)
      are expected to follow the 'loc_R_C' naming convention for coordinate parsing.

    # Heuristic Initialization
    - Extract goal locations for each box from the task goals.
    - Build a mapping from location names to (row, col) coordinates by parsing location strings found in static facts (adjacent) and goals. This map is used for Manhattan distance calculations.
    - Build an adjacency map from 'adjacent' facts to quickly find neighbors of a location. This is used to find locations adjacent to boxes.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state facts.
    2. Calculate the 'box component' of the heuristic:
       - Initialize `total_box_distance = 0`.
       - Identify the set of boxes that are not currently at their goal location (`ungoaled_boxes`).
       - For each box in `ungoaled_boxes`:
         - Find its current location and its goal location (from initialization).
         - If both locations were successfully parsed into coordinates during initialization:
           - Add the Manhattan distance between the current box location and its goal location to `total_box_distance`. This provides a lower bound on the grid steps the box needs to traverse.
         - If parsing failed for either location, the state might be malformed; return infinity.
    3. Calculate the 'robot component' of the heuristic:
       - If `ungoaled_boxes` is empty (meaning all boxes are at their goals), the robot component is 0. The total heuristic is 0.
       - If `ungoaled_boxes` is not empty:
         - Identify the set of all locations that are adjacent to *any* box in `ungoaled_boxes`. These are potential target locations for the robot to initiate a push. Use the precomputed adjacency map.
         - If no adjacent locations are found for any un-goaled box (e.g., due to missing adjacency facts or unparseable locations), the state might be unsolvable; return infinity.
         - Calculate the minimum Manhattan distance from the robot's current location to any location in this set of adjacent locations. This estimates the minimum travel cost for the robot to get near a box that needs attention.
         - Initialize `min_robot_distance_to_adjacent = infinity`.
         - For each location adjacent to an un-goaled box (ensuring it was successfully parsed into coordinates):
           - Calculate the Manhattan distance from the robot's current location to this adjacent location.
           - Update `min_robot_distance_to_adjacent` with the minimum distance found so far.
         - The robot component is `min_robot_distance_to_adjacent`.
    4. The total heuristic value is the sum of `total_box_distance` and the robot component (`min_robot_distance_to_adjacent`).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        all_locations_set = set() # Collect all location names encountered

        for goal in self.goals:
            parts = get_parts(goal)
            # Goal format is typically (at boxX locY)
            if parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # Assuming anything with a goal location is a box
                self.goal_locations[obj] = location
                all_locations_set.add(location)

        # Build adjacency map and collect all location names from static facts
        self.adjacent_map = {} # {loc1: {dir: loc2, ...}, ...}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.adjacent_map:
                    self.adjacent_map[loc1] = {}
                self.adjacent_map[loc1][direction] = loc2
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)

        # Create mapping from location name to (row, col) coordinates
        self.location_coords = {}
        for loc_name in all_locations_set:
            coords = parse_location(loc_name)
            if coords is not None:
                self.location_coords[loc_name] = coords
            # Locations that cannot be parsed will not be in self.location_coords.
            # manhattan_distance handles this by returning inf.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
                break

        # If robot location isn't found or cannot be parsed, the state is likely invalid.
        # Return infinity to prune this branch.
        if robot_location is None or robot_location not in self.location_coords:
             return float('inf')

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate for an object that has a goal (i.e., a box)
            if parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_locations:
                box, location = parts[1], parts[2]
                current_box_locations[box] = location

        total_box_distance = 0
        ungoaled_boxes = []
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If a box defined in goals is not found in the current state,
            # or its current location cannot be parsed, treat as highly undesirable.
            if current_loc is None or current_loc not in self.location_coords:
                 return float('inf') # Should not happen in valid states

            if current_loc != goal_loc:
                # Add Manhattan distance for this box
                # manhattan_distance handles cases where goal_loc is not in location_coords
                dist = manhattan_distance(current_loc, goal_loc, self.location_coords)
                if dist == float('inf'): return float('inf') # Propagate infinity if goal is unparseable
                total_box_distance += dist
                ungoaled_boxes.append(box)

        # If all boxes are at their goals, the heuristic is 0.
        if not ungoaled_boxes:
            return 0

        # Calculate robot distance component
        set_of_adjacent_to_ungoaled_boxes = set()
        for box in ungoaled_boxes:
            box_loc = current_box_locations[box]
            # Find all neighbors of the box's current location using the adjacency map
            if box_loc in self.adjacent_map:
                 for neighbor_loc in self.adjacent_map[box_loc].values():
                     # Only consider neighbors that were successfully parsed into coordinates
                     if neighbor_loc in self.location_coords:
                        set_of_adjacent_to_ungoaled_boxes.add(neighbor_loc)

        # If there are ungoaled boxes but no adjacent locations could be found/parsed
        # (e.g., grid definition issues, or box is in a location not in adjacent map),
        # return infinity as the state might be unsolvable or malformed.
        if not set_of_adjacent_to_ungoaled_boxes:
             # This case should ideally not be reached in a valid problem instance
             # where ungoaled_boxes is not empty and the grid is well-defined.
             return float('inf')


        # Calculate minimum distance from robot to any location adjacent to an ungoaled box
        min_robot_distance_to_adjacent = float('inf')
        for adj_loc in set_of_adjacent_to_ungoaled_boxes:
             # We already filtered for locations in self.location_coords when building the set
             dist = manhattan_distance(robot_location, adj_loc, self.location_coords)
             min_robot_distance_to_adjacent = min(min_robot_distance_to_adjacent, dist)

        # If min_robot_distance_to_adjacent is still infinity, it means robot_location
        # was not in location_coords, which was already checked at the start.
        # Or, all adjacent locations in the set had parsing issues, which was filtered.
        # So, if we reach here and ungoaled_boxes is not empty, min_robot_distance_to_adjacent
        # should be a finite non-negative number.

        # The heuristic is the sum of box distances and the minimum robot distance to get adjacent to a box.
        # This is a non-admissible estimate.
        return total_box_distance + min_robot_distance_to_adjacent
