# Need to import Heuristic from the correct path
# Assuming the structure is 'heuristics.heuristic_base'
# If running as a standalone file for testing or without the planner framework,
# define a dummy Heuristic base class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the real one is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_location(loc_str):
    """Parses a location string like 'loc_row_col' into a tuple (row, col)."""
    # Assuming format is always loc_row_col based on examples
    parts = loc_str.split('_')
    # Add error handling for unexpected formats if necessary, but assuming valid input per problem description.
    return (int(parts[1]), int(parts[2]))


def manhattan_distance(loc1_str, loc2_str):
    """Calculates Manhattan distance between two location strings."""
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)

def get_adjacent_locations(adj_list, loc):
    """Returns a list of locations adjacent to loc based on the adjacency list."""
    # adj_list maps loc -> {direction -> neighbor_loc}
    return list(adj_list.get(loc, {}).values())

def get_direction(loc1_str, loc2_str, adj_list):
     """Finds the direction from loc1 to loc2 if they are adjacent."""
     # Iterate through directions from loc1 to find loc2
     for direction, neighbor in adj_list.get(loc1_str, {}).items():
         if neighbor == loc2_str:
             return direction
     return None # Not adjacent

def get_pushing_locations_refined(box_loc_str, goal_loc_str, adj_list, clear_locations):
    """
    Returns a set of locations from which the robot can push the box
    at box_loc_str towards goal_loc_str, provided the location is clear.
    A location P is a valid pushing location if pushing the box from P
    moves it one step strictly closer (in Manhattan distance) to the goal,
    and P is clear in the current state.
    """
    potential_push_locs = set()
    current_md = manhattan_distance(box_loc_str, goal_loc_str)

    # Iterate through locations adjacent to the box (these are potential robot positions)
    for push_from_loc in get_adjacent_locations(adj_list, box_loc_str):
        # Find the direction from push_from_loc to box_loc_str
        direction_to_box = get_direction(push_from_loc, box_loc_str, adj_list)
        if direction_to_box:
            # The box will move in the direction_to_box if pushed from push_from_loc
            push_direction = direction_to_box
            # Find the location the box would move to
            box_prime_loc = adj_list.get(box_loc_str, {}).get(push_direction)

            if box_prime_loc:
                # Check if this push moves the box strictly closer to the goal
                if manhattan_distance(box_prime_loc, goal_loc_str) < current_md:
                    # Check if the potential pushing location is clear
                    if push_from_loc in clear_locations:
                        potential_push_locs.add(push_from_loc)

    return potential_push_locs


def bfs_distance_with_inf(start_loc, target_locs, traversable_locations, adj_list):
    """
    Performs BFS to find the minimum distance from start_loc to any location in target_locs,
    only moving through traversable_locations.
    Returns infinity if target_locs is empty or no target is reachable.
    """
    if not target_locs:
        return float('inf') # No valid target locations

    if start_loc in target_locs:
        return 0

    queue = deque([(start_loc, 0)])
    visited = {start_loc}
    traversable_set = set(traversable_locations)

    # The robot's start location must be considered traversable for the BFS start node
    # even if it's not marked 'clear' (e.g., robot is standing there).
    # We ensure start_loc is in traversable_locations before calling this function.
    if start_loc not in traversable_set:
         # This case indicates an issue with traversable_locations logic or state representation.
         # For robustness, return inf if robot starts in an untraversable spot (shouldn't happen in valid states).
         return float('inf')


    while queue:
        current_loc, dist = queue.popleft()

        # Get neighbors from the pre-built adjacency list
        for neighbor in get_adjacent_locations(adj_list, current_loc):
            # Check if the neighbor location is traversable in the current state
            if neighbor in traversable_set:
                if neighbor in target_locs:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Target not reachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to reach a goal state
    by summing the Manhattan distances of each misplaced box to its goal location
    and adding the minimum number of robot moves required to reach a position
    from which it can push any of the misplaced boxes towards its goal.

    # Assumptions
    - Each box has a specific goal location defined in the task goals.
    - The grid structure is implied by the 'loc_row_col' naming convention and 'adjacent' facts.
    - The heuristic considers the minimum pushes for boxes (Manhattan distance)
      and the robot's cost to get into a useful pushing position (BFS distance
      on clear cells), but ignores complex interactions like multiple boxes
      blocking each other or deadlocks where no useful pushing position is reachable.
    - Assumes unit cost for all actions (move and push).

    # Heuristic Initialization
    - Extract the goal location for each box from the task's goal conditions.
    - Build an adjacency list representing the grid connections from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Find the current location of the robot from the state.
    3. Find the current location of each box that has a goal from the state.
    4. Identify the set of locations that are currently 'clear' from the state.
    5. Identify the set of 'misplaced' boxes (those not at their goal location).
    6. If there are no misplaced boxes, the state is a goal state, return 0.
    7. Calculate the sum of Manhattan distances for each misplaced box to its goal location. Add this to the total cost. This estimates the minimum number of pushes required for all boxes independently.
    8. Determine the set of potential target locations for the robot: these are locations adjacent to any misplaced box, from which pushing the box would move it strictly closer (in Manhattan distance) to its goal, and which are currently 'clear'.
    9. Define the set of traversable locations for the robot's movement BFS: this includes all locations marked 'clear' in the state, plus the robot's current location.
    10. Calculate the minimum number of moves the robot needs to reach any of the potential target pushing locations using BFS on the traversable locations.
    11. If no target pushing location is reachable by the robot (BFS returns infinity), the state might be unsolvable or require complex unblocking; return infinity.
    12. Add the calculated robot distance to the total heuristic cost.
    13. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box
        and building the adjacency list from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        self.static_facts = task.static # Static facts

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build adjacency list from static facts
        self.adj_list = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                if l1 not in self.adj_list:
                    self.adj_list[l1] = {}
                self.adj_list[l1][direction] = l2
                # The PDDL defines adjacency in both directions explicitly,
                # so we don't need to add the reverse here.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location, box locations, and clear locations from the current state
        robot_loc = None
        box_locations = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1] in self.goal_locations:
                # Only track boxes that have a goal defined
                box, loc = parts[1], parts[2]
                box_locations[box] = loc
            elif parts[0] == "clear":
                 clear_locations.add(parts[1])

        # Identify misplaced boxes (boxes that have a goal but are not at that goal)
        misplaced_boxes = {
            box for box in self.goal_locations
            if box in box_locations and box_locations[box] != self.goal_locations[box]
        }

        # If all boxes are at their goals, the state is a goal state
        if not misplaced_boxes:
            return 0

        # Calculate sum of Manhattan distances for misplaced boxes
        # This is a lower bound on the number of pushes required
        sum_md_boxes = sum(
            manhattan_distance(box_locations[box], self.goal_locations[box])
            for box in misplaced_boxes
        )

        # Determine potential target locations for the robot's BFS.
        # These are clear locations adjacent to any misplaced box, from which
        # pushing the box moves it strictly closer to its goal.
        target_robot_locations = set()
        for box in misplaced_boxes:
            box_loc = box_locations[box]
            goal_loc = self.goal_locations[box]
            target_robot_locations.update(
                get_pushing_locations_refined(box_loc, goal_loc, self.adj_list, clear_locations)
            )

        # If there are no valid pushing locations reachable for the robot
        # (e.g., all sides of the box are blocked or pushing doesn't help),
        # the state might be unsolvable or require complex unblocking.
        # Returning infinity guides the search away from such states.
        if not target_robot_locations:
             return float('inf')

        # Define traversable locations for the robot's BFS.
        # The robot can move to any location marked 'clear'.
        # Its current location is also traversable (it can move *from* there).
        traversable_locations = clear_locations | {robot_loc}

        # Calculate the minimum number of moves the robot needs to reach any
        # of the potential target pushing locations using BFS.
        robot_dist = bfs_distance_with_inf(
            robot_loc,
            target_robot_locations,
            traversable_locations,
            self.adj_list
        )

        # If the robot cannot reach any valid pushing location, return infinity.
        if robot_dist == float('inf'):
             return float('inf')

        # The total heuristic cost is the sum of the minimum pushes needed for
        # all boxes (ignoring robot moves between pushes) plus the cost for
        # the robot to get into a position to start pushing *any* box.
        total_cost = sum_md_boxes + robot_dist

        return total_cost
