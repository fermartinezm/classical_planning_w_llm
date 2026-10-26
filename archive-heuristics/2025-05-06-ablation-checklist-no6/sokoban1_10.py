from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base class is not found
    # This allows the code structure to be checked, but it won't run
    # in a real planning environment without the actual base class.
    print("Warning: Heuristic base class not found. Using dummy class.")
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of args for matching
    if len(parts) < len(args):
         return False
    # Use zip to compare parts and args up to the length of args
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost based on the sum of shortest path distances for each box
    to its goal, plus the shortest path distance for the robot to reach a
    position from which it can push the "most critical" box (e.g., the one
    requiring the robot's nearest intervention).

    # Heuristic Initialization
    - Builds the graph of locations based on 'adjacent' facts.
    - Precomputes all-pairs shortest path distances between locations using BFS.
    - Extracts goal locations for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and each box.
    2. For each box not at its goal:
       a. Calculate the shortest path distance from the box's current location
          to its goal location using the precomputed distances. Sum these distances.
          This represents the minimum number of pushes required for all boxes.
       b. Find a location adjacent to the box's current location that lies on
          a shortest path towards the goal.
       c. Determine the required robot position to push the box from its current
          location to that next location (this position is adjacent to the box's
          current location, in the opposite direction of the push).
       d. Calculate the shortest path distance from the robot's current location
          to this required pushing position. Keep track of the minimum such distance
          across all boxes that need pushing.
    3. The heuristic value is the sum of box-to-goal distances plus the minimum
       robot-to-push-position distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, precomputing
        distances, and extracting goal locations.
        """
        # Call the base class constructor
        super().__init__(task)

        # Build the adjacency list graph from static facts
        self.adjacency_list = {}
        self.locations = set()
        direction_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.adjacency_list:
                    self.adjacency_list[loc1] = {}
                # Store neighbor and the direction *to* that neighbor
                self.adjacency_list[loc1][direction] = loc2

                # Add reverse adjacency
                if loc2 not in self.adjacency_list:
                    self.adjacency_list[loc2] = {}
                opposite_dir = direction_map[direction]
                 # Store neighbor and the direction *to* that neighbor
                self.adjacency_list[loc2][opposite_dir] = loc1


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

        # Store goal locations for each box
        self.goal_locations = {}
        # self.goals is a frozenset of goal facts
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            # Assuming goals are always (at box location)
            if parts[0] == "at":
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Map directions to their opposites for finding push positions
        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to compute shortest distances
        to all reachable nodes. Stores results in self.distances.
        """
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_loc, dist = q.popleft()

            # Check all neighbors
            if current_loc in self.adjacency_list:
                # Iterate through directions to get neighbors
                for direction, neighbor_loc in self.adjacency_list[current_loc].items():
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        self.distances[(start_node, neighbor_loc)] = dist + 1
                        q.append((neighbor_loc, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract robot and box locations from the current state
        robot_loc = None
        box_locations = {} # {box_name: location_name}
        # clear_locations = set() # Not used in this heuristic

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at-robot":
                robot_loc = parts[1]
            elif predicate == "at":
                # Check if it's a box that is a goal object
                if parts[1] in self.goal_locations:
                    box, loc = parts[1], parts[2]
                    box_locations[box] = loc
            # elif predicate == "clear":
            #     clear_locations.add(parts[1])

        # If robot_loc is None, something is wrong with the state representation or domain.
        # Should not happen in valid states.
        if robot_loc is None:
             return float('inf') # Indicate an invalid or unreachable state

        total_box_distance = 0
        min_robot_to_push_pos_distance = float('inf')
        boxes_need_moving = False

        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # If a goal box is not found in the current state, it's an issue.
            # This could happen if a box was somehow removed from the state facts.
            # Assuming valid state representation where all goal boxes are present.
            if current_loc is None:
                 return float('inf') # Indicate an invalid or unreachable state

            if current_loc != goal_loc:
                boxes_need_moving = True
                # Add box-to-goal distance
                # Handle cases where goal is unreachable from current box location
                if (current_loc, goal_loc) not in self.distances:
                    return float('inf') # Box cannot reach goal

                dist_b_goal = self.distances[(current_loc, goal_loc)]
                total_box_distance += dist_b_goal

                # Find potential push positions for this box
                # A push position is adjacent to current_loc, allowing a push towards goal_loc
                # This means the neighbor is on a shortest path from current_loc to goal_loc.
                # The robot must be at the location adjacent to current_loc in the *opposite*
                # direction of the box's intended movement.

                if current_loc in self.adjacency_list:
                    # Iterate through neighbors of the current box location
                    for direction_to_neighbor, neighbor_loc in self.adjacency_list[current_loc].items():
                        # Check if this neighbor_loc is on a shortest path from current_loc to goal_loc
                        # This is true if dist(current_loc, goal_loc) = 1 + dist(neighbor_loc, goal_loc)
                        if (neighbor_loc, goal_loc) in self.distances and \
                           self.distances[(neighbor_loc, goal_loc)] == dist_b_goal - 1:

                            # neighbor_loc is a valid next step for the box towards the goal.
                            # The direction the box will move is 'direction_to_neighbor'.
                            # The robot needs to be at the location adjacent to current_loc
                            # in the opposite direction of 'direction_to_neighbor'.
                            required_robot_dir = self.opposite_direction.get(direction_to_neighbor)

                            if required_robot_dir and current_loc in self.adjacency_list and \
                               required_robot_dir in self.adjacency_list[current_loc]:

                                push_pos = self.adjacency_list[current_loc][required_robot_dir]

                                # Calculate robot distance to this push position
                                if (robot_loc, push_pos) in self.distances:
                                    dist_robot_to_this_push_pos = self.distances[(robot_loc, push_pos)]
                                    min_robot_to_push_pos_distance = min(min_robot_to_push_pos_distance, dist_robot_to_this_push_pos)
                                else:
                                     # Robot cannot reach this specific push position.
                                     # This push might be impossible from the current robot location.
                                     # We continue searching for other valid push positions for this box or other boxes.
                                     pass # min_robot_to_push_pos_distance remains unchanged

        # If no boxes need moving, the heuristic is 0.
        if not boxes_need_moving:
            return 0

        # If there are boxes to move, but the robot cannot reach *any* valid
        # push position for *any* of them (min_robot_to_push_pos_distance is still inf).
        # This indicates the state is likely unsolvable or a dead end.
        if min_robot_to_push_pos_distance == float('inf'):
             return float('inf')

        # The heuristic is the sum of box-to-goal distances (minimum pushes)
        # plus the cost for the robot to get to the first push position.
        # This ignores robot movement between pushes or between boxes after the first push.
        # It's a simple, non-admissible estimate.
        return total_box_distance + min_robot_to_push_pos_distance
