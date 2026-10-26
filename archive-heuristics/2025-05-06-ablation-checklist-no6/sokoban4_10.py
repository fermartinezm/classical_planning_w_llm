# Add necessary imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs(start_loc, end_loc, graph, occupied_locations):
    """
    Performs Breadth-First Search to find the shortest path distance.
    Args:
        start_loc (str): The starting location.
        end_loc (str): The target location.
        graph (dict): Adjacency list representation of the grid.
        occupied_locations (set): Locations that cannot be traversed.
    Returns:
        int: The shortest distance, or float('inf') if unreachable.
    """
    if start_loc == end_loc:
        return 0
    # Check if start or end locations are valid nodes in the graph
    if start_loc not in graph or end_loc not in graph:
        return float('inf')

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end_loc:
            return dist

        # Check neighbors
        # current_loc is guaranteed to be in graph here because it came from the queue
        # and the start_loc check was done.
        for neighbor in graph[current_loc]:
            # A location is traversable if it's not in the occupied_locations set
            # Note: The target location (end_loc) *can* be in occupied_locations,
            # but intermediate steps cannot.
            is_occupied = neighbor in occupied_locations and neighbor != end_loc

            if neighbor not in visited and not is_occupied:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Target unreachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing:
    1. The minimum number of pushes required for each box to reach its goal
       (calculated as shortest path distance on the grid, ignoring other objects).
    2. The minimum number of moves required for the robot to reach the nearest
       box that is not yet at its goal (calculated as shortest path distance
       on the grid, avoiding other boxes).

    # Heuristic Initialization
    - Builds a graph representation of the grid from `adjacent` facts.
    - Stores goal locations for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot.
    2. Identify the current location of each box relevant to the goal.
    3. Calculate the total cost for boxes:
       For each box not at its goal:
       a. Calculate the shortest path distance for the box from its current
          location to its goal location on the grid, ignoring all obstacles.
          This is the minimum number of pushes required for this box.
       b. Add this distance to the total box cost.
    4. Calculate the robot cost:
       Find the box that is not at its goal and is closest to the robot
       (shortest path distance on the grid, avoiding other boxes).
       Calculate the shortest path distance for the robot from its current
       location to this nearest box location, avoiding other boxes.
       Add this robot distance to the total robot cost.
    5. The total heuristic is the sum of the total box cost and the total robot cost.
    6. If any required distance calculation returns infinity (unreachable),
       the heuristic should return infinity or a very large number, unless
       the state is the goal state (heuristic should be 0).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (`adjacent` relationships).
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically like '(at box1 loc_X)'
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Build the grid graph from adjacent facts.
        # The graph maps location names (str) to a list of adjacent location names (str).
        self.graph = {}
        for fact in static_facts:
            # Adjacent facts are like '(adjacent loc1 loc2 direction)'
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                _, loc1, loc2, _ = parts
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Assuming adjacency is bidirectional based on example files
                if loc2 not in self.graph[loc1]:
                    self.graph[loc1].append(loc2)
                if loc1 not in self.graph[loc2]:
                     self.graph[loc2].append(loc1)

        # Get all possible locations from the graph keys
        self.all_locations = set(self.graph.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
                break
        if robot_loc is None:
             # Robot location must be known in a valid state
             return float('inf')

        # Find current box locations for boxes relevant to the goal
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # Only track boxes we care about
                    current_box_locations[obj] = loc

        # Identify locations occupied by boxes relevant to the goal
        occupied_by_boxes = set(current_box_locations.values())

        # Calculate total box movement cost and identify boxes needing move
        total_box_push_cost = 0
        boxes_needing_move_locs = []
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)
            if current_loc is None:
                 # A box from the goal is not present in the state facts. Unsolvable.
                 return float('inf')

            if current_loc != goal_loc:
                # Box needs to move. Calculate box-to-goal distance (pushes).
                # Box movement ignores other objects as obstacles in this simple heuristic.
                # The BFS for the box path should not consider any locations as occupied.
                box_dist = bfs(current_loc, goal_loc, self.graph, set())
                if box_dist == float('inf'):
                    # Box cannot reach its goal even without obstacles
                    return float('inf') # Unsolvable state
                total_box_push_cost += box_dist
                boxes_needing_move_locs.append(current_loc)

        # If total_box_push_cost is 0, all relevant boxes are at their goals.
        # The initial goal check should have caught this as a goal state.
        # If we are here, the state is not a goal state but all relevant boxes are in place.
        # This is unexpected based on typical Sokoban goals. Return infinity as a safe fallback.
        # Or, perhaps the goal includes something else?
        # Given the problem description and examples, this branch implies
        # an issue with the state/goal representation or an unsolvable state
        # where the goal involves something other than box positions that
        # cannot be achieved. Returning infinity seems appropriate.
        # However, if total_box_push_cost is 0, the initial check should pass.
        # Let's assume this else block is effectively unreachable for solvable states.
        # If it *is* reached, it means self.goals <= state is False, but
        # all boxes in self.goal_locations are at their target locations.
        # This implies the goal has other conditions. This heuristic doesn't
        # handle other goal conditions, so it should return infinity if
        # the state isn't the goal state despite boxes being in place.
        if total_box_push_cost == 0:
             # This implies all boxes are at their goals, but the state is not the goal state.
             # This heuristic only considers box positions for the goal.
             # If the goal has other conditions, this heuristic cannot guide towards them.
             # Return infinity as it's likely unsolvable under this heuristic's scope.
             # However, the initial `if self.goals <= state:` check handles the true goal.
             # If we are here, `self.goals <= state` is False. If `total_box_push_cost == 0`,
             # it means all box-at-goal conditions *are* met, but other goal conditions are not.
             # This heuristic doesn't help with those. Returning infinity is correct.
             return float('inf')


        # Calculate robot movement cost to the nearest box needing a move.
        min_robot_dist_to_box = float('inf')
        # boxes_needing_move_locs is guaranteed to be non-empty here because total_box_push_cost > 0
        # The robot needs to reach a location near a box that needs moving.
        # Calculate the distance from the robot to the location of each box
        # that needs moving. The robot cannot move onto squares occupied by
        # *other* boxes.
        robot_obstacles = occupied_by_boxes - {robot_loc} # Robot's own square is not an obstacle for itself

        for box_loc in boxes_needing_move_locs:
             # Calculate distance from robot_loc to box_loc, avoiding other boxes
             # Note: The target box_loc *is* occupied by the box, so the robot
             # cannot end *on* that square. The BFS finds the path *to* that square.
             # This is an approximation of the cost to get *adjacent* to the box.
             dist = bfs(robot_loc, box_loc, self.graph, robot_obstacles)
             min_robot_dist_to_box = min(min_robot_dist_to_box, dist)

        if min_robot_dist_to_box == float('inf'):
             # Robot cannot reach any box that needs moving
             return float('inf') # Unsolvable state

        # The total heuristic is the sum of the box pushes and the robot's
        # effort to get to the nearest box.
        total_cost = total_box_push_cost + min_robot_dist_to_box

        return total_cost
