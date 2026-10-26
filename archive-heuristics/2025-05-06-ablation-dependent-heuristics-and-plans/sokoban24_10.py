# Assume Heuristic base class is imported from heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math # Used for float('inf')

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Handle facts that might not be in (pred arg1 arg2) format, though less common in init/goal
    return fact_str.split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    try:
        parts = loc_str.split('_')
        # Ensure there are exactly 3 parts and the last two are digits
        if len(parts) == 3 and parts[0] == 'loc' and parts[1].isdigit() and parts[2].isdigit():
             return (int(parts[1]), int(parts[2]))
        # print(f"Warning: Unexpected location format: {loc_str}") # Debugging
        return None # Indicate parsing failure
    except (ValueError, IndexError):
        # print(f"Warning: Could not parse location string: {loc_str}") # Debugging
        return None # Indicate parsing failure


def md(p1, p2):
    """Calculates Manhattan distance between two (row, col) tuples."""
    if p1 is None or p2 is None:
        return float('inf') # Cannot calculate distance if parsing failed
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def opposite_dir(dir_str):
    """Returns the opposite direction string."""
    if dir_str == 'up': return 'down'
    if dir_str == 'down': return 'up'
    if dir_str == 'left': return 'right'
    if dir_str == 'right': return 'left'
    return None # Should not happen with valid directions

def bfs(start_loc, graph, allowed_nodes):
    """
    Performs BFS on the graph starting from start_loc, only traversing edges
    where the destination node is in allowed_nodes.
    Returns a dictionary mapping reachable nodes to their distance from start_loc.
    """
    # Assume start_loc is a valid node in the graph keys.
    # allowed_nodes should include start_loc if it's a valid place for the robot to be.

    queue = deque([start_loc])
    distances = {start_loc: 0}
    visited = {start_loc}

    while queue:
        current_loc = queue.popleft()
        current_dist = distances[current_loc]

        # Iterate through neighbors using the full adjacency graph
        for direction, neighbor_loc in graph.get(current_loc, {}).items():
            # Only consider neighbors that are allowed (clear or robot's start) and not visited
            if neighbor_loc in allowed_nodes and neighbor_loc not in visited:
                visited.add(neighbor_loc)
                distances[neighbor_loc] = current_dist + 1
                queue.append(neighbor_loc)

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing
    contributions for each box that is not yet at its goal location.
    The contribution for a single box is estimated as the minimum cost
    to move that box one step closer to its goal, plus the remaining
    Manhattan distance to the goal (representing subsequent pushes).
    The cost to move a box one step closer includes the robot's travel
    cost to the required push position and the push action itself.

    # Assumptions
    - The problem grid is represented by 'loc_R_C' strings where R and C are integers.
    - Locations are connected as defined by 'adjacent' facts.
    - All boxes have a specific goal location defined in the task goals.
    - The heuristic assumes it's always possible to eventually move a box
      towards its goal if it's not already there, although it checks for
      immediate pushability from the current robot position and clear squares.
      If a box cannot be pushed towards its goal *at all* from the current
      state (robot cannot reach push position, or target square is blocked
      for all goal-reducing directions), the heuristic returns a large value.

    # Heuristic Initialization
    - Parses the task goals to store the target location for each box.
    - Builds an adjacency graph of all locations based on 'adjacent' facts
      to facilitate pathfinding (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Identify all locations that are currently clear.
    3. Determine which boxes are not yet at their goal locations.
    4. If all boxes are at their goals, the heuristic is 0.
    5. For each box not at its goal:
       a. Calculate the Manhattan distance between the box's current location
          and its goal location. This is a lower bound on the number of pushes needed.
       b. Identify potential "push actions" that would move the box one step
          closer to its goal. A push is possible from `loc_b` to `loc_b_next`
          in direction `dir` if `loc_b_next` is adjacent to `loc_b` in `dir`,
          `loc_b_next` is closer to the goal than `loc_b`, and the location
          `loc_r_push` behind `loc_b` (adjacent in the opposite direction of `dir`)
          exists.
       c. From the potential push actions identified in 5b, filter those where
          the target location (`loc_b_next`) is currently clear.
       d. For each valid push action (from 5c), calculate the cost for the robot
          to reach the required push location (`loc_r_push`) from its current
          position. This is done using BFS on the graph of *currently clear*
          locations (plus the robot's own location).
       e. Find the minimum cost among all valid push actions for this box.
          The cost of a push action is `robot_travel_cost + 1` (for the push).
       f. If no valid push action towards the goal is possible for this box
          (either no push location exists, target is blocked, or robot cannot
           reach any push location), the state is likely problematic. Return
           a large penalty value for the total heuristic.
       g. If at least one valid push action is possible, the heuristic
          contribution for this box is `min_cost_one_push + (manhattan_distance - 1)`.
          This accounts for the cost of the first necessary push and assumes
          subsequent pushes cost at least 1 each.
    6. The total heuristic value is the sum of the contributions for all
       boxes not at their goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the adjacency graph."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at box loc)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Collect all locations mentioned in init, goal, and static adjacent facts.
        all_locations_set = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["at-robot", "at", "clear"] and len(parts) >= 2:
                 # The location is usually the last part
                 if len(parts) > 1:
                    all_locations_set.add(parts[-1])
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == "at" and len(parts) == 3:
                 all_locations_set.add(parts[2])
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "adjacent" and len(parts) == 4:
                 all_locations_set.add(parts[1])
                 all_locations_set.add(parts[2])

        # Build the full adjacency graph ensuring all relevant locations are keys
        self.adj_graph = {loc: {} for loc in all_locations_set}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Ensure locations are in our set before adding to graph
                if loc1 in self.adj_graph and loc2 in self.adj_graph:
                    self.adj_graph[loc1][direction] = loc2
                    rev_dir = opposite_dir(direction)
                    if rev_dir: # opposite_dir should not return None for valid directions
                         self.adj_graph[loc2][rev_dir] = loc1
                # else: print(f"Warning: Adjacent fact mentions unknown location: {fact}") # Debugging


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current locations and clear locations
        robot_loc = None
        box_locations = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                box_locations[box] = location
            elif parts[0] == "clear" and len(parts) == 2:
                clear_locations.add(parts[1])

        # Ensure robot location is known and is a valid node in the graph
        if robot_loc is None or robot_loc not in self.adj_graph:
             # print(f"Error: Robot location {robot_loc} not found or invalid.") # Debugging
             return float('inf') # Cannot proceed without valid robot location

        # 3. Determine boxes not at goal
        boxes_to_move = [
            box for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        ]

        # 4. If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        # 5. Calculate robot distances using BFS on clear locations + robot location
        # The robot can start BFS from its current location. It can move into clear locations.
        # The set of nodes the robot can be *on* during the path includes clear locations
        # and the starting location (even if the starting location isn't currently clear,
        # which happens right after pushing a box into a clear spot).
        robot_allowed_locations = clear_locations | {robot_loc}
        robot_distances = bfs(robot_loc, self.adj_graph, robot_allowed_locations)

        total_h = 0
        large_penalty = 1000000 # Penalty for boxes that cannot be moved towards goal

        # 5. For each box not at its goal
        for box in boxes_to_move:
            loc_b = box_locations.get(box) # Use .get for safety
            goal_b = self.goal_locations.get(box) # Use .get for safety

            # Ensure box location and goal location are valid and in the graph
            if loc_b is None or goal_b is None or loc_b not in self.adj_graph or goal_b not in self.adj_graph:
                 # print(f"Warning: Invalid location data for box {box} ({loc_b}) or goal ({goal_b}). Returning penalty.") # Debugging
                 return large_penalty # Indicate an unhandleable state

            parsed_loc_b = parse_location(loc_b)
            parsed_goal_b = parse_location(goal_b)

            # Handle parsing errors
            if parsed_loc_b is None or parsed_goal_b is None:
                 # print(f"Warning: Could not parse location for box {box} ({loc_b}) or its goal ({goal_b}). Returning penalty.") # Debugging
                 return large_penalty # Indicate an unhandleable state


            box_md = md(parsed_loc_b, parsed_goal_b)

            # If box is somehow at goal but in boxes_to_move list (shouldn't happen with correct logic)
            if box_md == 0:
                 continue

            min_cost_one_push = float('inf')

            # 5b, 5c, 5d: Find valid push actions and calculate robot cost
            # Iterate through possible push directions from the box's perspective
            for dir in ['up', 'down', 'left', 'right']:
                loc_b_next = self.adj_graph.get(loc_b, {}).get(dir)

                # Check if pushing in this direction is possible and moves towards goal
                if loc_b_next is not None:
                    parsed_loc_b_next = parse_location(loc_b_next)
                    # Ensure the next location is valid, parsing was successful, and it's in the graph
                    if parsed_loc_b_next is not None and loc_b_next in self.adj_graph and md(parsed_loc_b_next, parsed_goal_b) < box_md:
                        # This push moves the box closer to the goal
                        # Find the required robot location behind the box
                        rev_dir = opposite_dir(dir)
                        loc_r_push = self.adj_graph.get(loc_b, {}).get(rev_dir)

                        # Check if the required robot location exists, is in the graph, and the target square is clear
                        if loc_r_push is not None and loc_r_push in self.adj_graph and loc_b_next in clear_locations:
                            # Check if the robot can reach the required push location
                            if loc_r_push in robot_distances:
                                cost_one_push = robot_distances[loc_r_push] + 1 # Robot travel + push action
                                min_cost_one_push = min(min_cost_one_push, cost_one_push)

            # 5f: Check if any valid push towards goal is possible
            if min_cost_one_push == float('inf'):
                # No push towards the goal is currently possible for this box.
                # This state is likely a dead end or requires complex detours.
                # Return a large penalty to discourage the search from exploring this path.
                # print(f"Box {box} at {loc_b} cannot be pushed towards goal {goal_b} immediately. Returning penalty.") # Debugging
                return large_penalty

            # 5g: Add contribution for this box
            # Cost is cost of first push + remaining pushes (estimated by box_md - 1)
            total_h += min_cost_one_push + (box_md - 1)

        # 6. Return total heuristic value
        return total_h
