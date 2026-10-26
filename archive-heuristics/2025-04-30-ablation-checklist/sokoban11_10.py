import math
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions adapted from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_name):
    """Parses a location string like 'loc_row_col' into a tuple (row, col)."""
    try:
        parts = loc_name.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            return (int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        pass # Handle potential parsing errors gracefully
    # Return None or raise an error for invalid format
    return None # Or raise ValueError(f"Invalid location format: {loc_name}")

def manhattan_distance(loc1_name, loc2_name):
    """Calculates the Manhattan distance between two locations."""
    coord1 = parse_location(loc1_name)
    coord2 = parse_location(loc2_name)
    if coord1 is None or coord2 is None:
        # Cannot calculate distance for invalid locations
        return float('inf')
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def opposite_dir(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen with valid directions

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of Manhattan distances for each box to its goal location.
    2. The minimum BFS distance for the robot to reach a location adjacent to
       any box that is not yet at its goal.

    # Assumptions
    - The grid structure is defined by 'adjacent' facts.
    - Locations are named in the format 'loc_row_col'.
    - The cost of moving the robot is 1.
    - The cost of pushing a box one step (which includes robot movement) is estimated.
      The heuristic approximates this by summing box distances and robot distance to a pushable spot.

    # Heuristic Initialization
    - Extract goal locations for each box.
    - Build an adjacency map representing the grid from 'adjacent' facts.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Determine which boxes are not yet at their goal locations.
    4. Calculate the first component (h1): Sum of Manhattan distances for each
       box not at its goal to its respective goal location.
    5. Identify all locations currently occupied by the robot or boxes. These
       are obstacles for the robot's free movement.
    6. Perform a Breadth-First Search (BFS) starting from the robot's current
       location to find the shortest path distance to all reachable *clear* locations.
    7. Identify potential "pushing positions": For each box not at its goal,
       any location adjacent to that box's current location is a potential
       position for the robot to be in *before* pushing the box.
    8. Calculate the second component (h2): Find the minimum BFS distance from
       the robot's current location to any of the potential pushing positions
       identified in step 7. If no such position is reachable, this component
       should be a large value (indicating a likely dead end or difficult state).
    9. The total heuristic value is h1 + h2.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - The grid structure (adjacency map) from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build adjacency map: {loc1: {direction: loc2, ...}, ...}
        self.adj = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                if loc1 not in self.adj:
                    self.adj[loc1] = {}
                self.adj[loc1][direction] = loc2
                # Add the reverse adjacency as well
                opp_dir = opposite_dir(direction)
                if opp_dir:
                    if loc2 not in self.adj:
                        self.adj[loc2] = {}
                    self.adj[loc2][opp_dir] = loc1

    def robot_bfs(self, start_loc, occupied_locations):
        """
        Performs BFS from start_loc to find distances to all reachable clear locations.
        Occupied locations are treated as obstacles.
        Returns a dictionary {location: distance}.
        """
        q = deque([(start_loc, 0)])
        visited = {start_loc}
        distances = {start_loc: 0}

        while q:
            current_loc, current_dist = q.popleft()

            # Robot can only move to adjacent locations that are not occupied
            for neighbor_loc in self.adj.get(current_loc, {}).values():
                if neighbor_loc not in occupied_locations and neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    distances[neighbor_loc] = current_dist + 1
                    q.append((neighbor_loc, current_dist + 1))

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Robot location unknown

        # 2. Find box locations and occupied locations
        box_locations = {}
        occupied_locations = {robot_loc} # Robot occupies its spot
        for fact in state:
            if match(fact, "at", "*", "*"):
                box, loc = get_parts(fact)
                box_locations[box] = loc
                occupied_locations.add(loc) # Boxes occupy their spots

        # 3. Identify boxes not at goal
        boxes_to_move = {
            box: loc for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        }

        # If all boxes are at their goals, the heuristic is 0
        if not boxes_to_move:
            return 0

        # 4. Calculate h1: Sum of box-to-goal Manhattan distances
        h1 = 0
        for box, current_box_loc in boxes_to_move.items():
            goal_box_loc = self.goal_locations[box]
            h1 += manhattan_distance(current_box_loc, goal_box_loc)

        # 5. & 6. Perform BFS for robot movement
        # The BFS should consider locations occupied by *other* objects (boxes, robot)
        # as blocked. However, the robot *can* move onto a location previously occupied
        # by a box *if* it pushes the box out of the way.
        # A simpler, non-admissible approach for BFS is to treat *all* currently
        # occupied locations (except the robot's start location itself) as obstacles.
        # This is an approximation.
        robot_obstacles = set(occupied_locations) - {robot_loc}
        robot_distances = self.robot_bfs(robot_loc, robot_obstacles)

        # 7. Identify potential pushing positions
        # A potential pushing position is any location adjacent to a box that needs moving.
        potential_push_positions = set()
        for box, box_loc in boxes_to_move.items():
             # Get all locations adjacent to the box's current location
             for adj_loc in self.adj.get(box_loc, {}).values():
                 potential_push_positions.add(adj_loc)

        # 8. Calculate h2: Minimum robot distance to a potential pushing position
        min_robot_dist_to_push_pos = float('inf')
        for push_pos in potential_push_positions:
            if push_pos in robot_distances:
                min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_distances[push_pos])

        # If no potential pushing position is reachable, return a large value
        if min_robot_dist_to_push_pos == float('inf'):
            return float('inf') # Or a large constant like 1000

        # 9. Total heuristic value
        # The robot needs to reach a pushing position (cost h2), and then perform
        # pushes (estimated by h1). Each push also involves the robot moving
        # into the box's previous spot. A simple sum h1 + h2 is a reasonable
        # non-admissible estimate. h1 estimates the box movement cost, h2 estimates
        # the robot's cost to get *to* the action area.
        return h1 + min_robot_dist_to_push_pos

