# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts (similar to Logistics example)
from fnmatch import fnmatch
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
from collections import deque

def bfs_distance(graph, start, end):
    """
    Computes the shortest path distance between two locations in the graph.
    Graph is an adjacency list: {location: [adjacent_locations]}
    Returns distance or float('inf') if no path exists.
    """
    if start == end:
        return 0
    # Ensure start and end are in the graph, otherwise distance is infinite
    if start not in graph or end not in graph:
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        # Check if current_loc is in graph before accessing neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # No path found

# Import the base Heuristic class
from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its goal location,
    and adding the shortest path distance from the robot to the nearest box
    that still needs to be moved.

    # Assumptions
    - Each box has a unique goal location specified in the problem goal.
    - The grid defined by 'adjacent' facts represents the traversable locations
      and connections.
    - The heuristic ignores obstacles (other boxes, walls) on the path and
      the specific pushing mechanics (robot needs to be behind the box, target
      location must be clear), making it non-admissible but potentially efficient.
    - Assumes the graph is connected enough for relevant paths to exist in solvable problems.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds a graph (adjacency list) of all locations based on the 'adjacent'
      facts provided in the static information. This graph represents the
      possible movements between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Initialize total heuristic cost to 0.
    3.  Parse the current state to find the location of the robot and each box.
    4.  Initialize `min_robot_dist_to_box` to infinity and `has_boxes_to_move` to False.
    5.  For each box that has a goal location:
        a.  Get its current location from the state. If a box with a goal is not found
            in the state (unexpected in valid states), return infinity.
        b.  Look up its corresponding goal location from the pre-computed goal mapping.
        c.  Compute the shortest path distance between the box's current location
            and its goal location using BFS on the location graph. This distance
            represents the minimum number of 'pushes' needed for this box
            if there were no obstacles and the robot was always in position.
        d.  If the distance is infinity (box cannot reach goal), return infinity
            for the entire state.
        e.  Add this distance to the total heuristic cost.
        f.  If the box is not yet at its goal location:
            i.  Set `has_boxes_to_move` to True.
            ii. Compute the shortest path distance from the robot's current location
                to the box's current location using BFS.
            iii. Update `min_robot_dist_to_box` with the minimum distance found so far.
    6.  Check if robot can reach any box that needs moving. If `has_boxes_to_move` is True
        and `min_robot_dist_to_box` is still infinity (robot cannot reach any box that needs moving),
        return infinity.
    7.  If `has_boxes_to_move` is True, add `min_robot_dist_to_box` to the total heuristic cost.
        This accounts for the robot needing to reach a box to start pushing.
    8.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the location graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        self.location_graph = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent":
                loc1, loc2, direction = args
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                # Add bidirectional edges for movement
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1) # Assuming adjacency is symmetric

        # Remove duplicates from adjacency lists (optional but clean)
        for loc in self.location_graph:
             self.location_graph[loc] = list(set(self.location_graph[loc]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        # 2. Initialize total heuristic cost
        total_cost = 0

        # 3. Parse current state for robot and box locations
        robot_location = None
        current_box_locations = {}
        for fact in state:
             predicate, *args = get_parts(fact)
             if predicate == "at-robot":
                  robot_location = args[0]
             elif predicate == "at" and args[0] in self.goal_locations:
                  box, location = args
                  current_box_locations[box] = location

        # Ensure robot location is found (should always be the case in valid states)
        if robot_location is None:
             # This indicates an invalid state representation
             return float('inf')

        # 4. Initialize robot distance tracking
        min_robot_dist_to_box = float('inf')
        has_boxes_to_move = False

        # 5. Process each box with a goal
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)
            if current_location is None:
                 # A box with a goal is not located in the state facts.
                 # This indicates an invalid or unsolvable state from here.
                 return float('inf')

            # 5c. Compute shortest path distance for the box
            dist_box_to_goal = bfs_distance(self.location_graph, current_location, goal_location)

            # 5d. If box cannot reach goal, state is unsolvable
            if dist_box_to_goal == float('inf'):
                 return float('inf')

            # 5e. Add box distance to total cost
            total_cost += dist_box_to_goal

            # 5f. If box is not at its goal, consider robot distance
            if current_location != goal_location:
                 has_boxes_to_move = True
                 # 5f.ii. Compute robot distance to this box
                 dist_robot_to_box = bfs_distance(self.location_graph, robot_location, current_location)
                 # 5f.iii. Update minimum robot distance
                 min_robot_dist_to_box = min(min_robot_dist_to_box, dist_robot_to_box)

        # 6. Check if robot can reach any box that needs moving. If `has_boxes_to_move` is True
        # and `min_robot_dist_to_box` is still infinity (robot cannot reach any box that needs moving),
        # return infinity.
        if has_boxes_to_move and min_robot_dist_to_box == float('inf'):
             return float('inf')

        # 7. If `has_boxes_to_move` is True, add `min_robot_dist_to_box` to the total heuristic cost.
        # This accounts for the robot needing to reach a box to start pushing.
        if has_boxes_to_move:
             total_cost += min_robot_dist_to_box

        # 8. Return the total heuristic cost
        return total_cost
