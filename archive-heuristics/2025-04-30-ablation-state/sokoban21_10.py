from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen in this domain

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It considers the shortest path distance for each
    misplaced box to its goal and the shortest path distance for the robot
    to reach a position from which it can push any misplaced box towards its goal.

    # Assumptions
    - The grid structure and adjacency are defined by the 'adjacent' facts.
    - Each box has a specific goal location defined in the task goals.
    - The cost of moving the robot and pushing a box is 1 action.
    - The heuristic simplifies the robot's movement cost by only considering
      the cost to reach the *first* useful push position, ignoring subsequent
      robot repositioning costs between pushes.
    - It detects simple deadlocks where a box cannot be pushed towards its goal
      or the robot cannot reach any valid push position.
    - Robot movement cost calculation ignores dynamic obstacles (other boxes or target box locations)
      along the path, only checking the final target location for occupancy.

    # Heuristic Initialization
    - Parses 'adjacent' facts to build a graph of locations and their adjacencies
      including directions.
    - Computes all-pairs shortest path distances between all locations using BFS.
    - Parses goal conditions to map each box to its goal location.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Identify all locations that are currently clear.
    3. Initialize the sum of box-to-goal distances (`total_box_distance`) to 0.
    4. Initialize the minimum robot-to-push-position distance (`min_robot_push_distance`) to infinity.
    5. Identify all misplaced boxes (boxes not at their goal location).
    6. If there are no misplaced boxes, the goal is reached, return 0.
    7. For each misplaced box:
       a. Get its current location (`box_loc`) and its goal location (`goal_loc`).
       b. Calculate the shortest path distance from `box_loc` to `goal_loc` using the precomputed distances. If unreachable, the state is likely unsolvable, return infinity.
       c. Add this distance to `total_box_distance`. This represents the minimum number of pushes needed for this box.
       d. Find potential "valid push positions" for this box: A location `p` is a valid push position if it's adjacent to `box_loc` in the direction of a useful push (`d_push`), the location `next_box_loc` where the box would move if pushed from `p` is clear in the current state, and `next_box_loc` is strictly closer to `goal_loc` than `box_loc`. The required robot position `p` is such that `(adjacent p box_loc d_push)`.
       e. Add all such valid push positions `p` to a set of `all_valid_push_positions`.
    8. If `total_box_distance > 0` but `all_valid_push_positions` is empty, it indicates a potential deadlock where no box can be pushed towards its goal from any reachable adjacent location, or the target location is blocked. Return infinity.
    9. If `all_valid_push_positions` is not empty, calculate the minimum shortest path distance from the robot's current location to any location in `all_valid_push_positions`. Ensure the target push position is not occupied by another box. If the robot cannot reach any valid push position, return infinity.
    10. The heuristic value is `total_box_distance + min_robot_push_distance`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency maps and list of all locations
        self.adj_map = {} # { loc1: { dir: loc2, ... }, ... }
        self.rev_adj_map = {} # { loc2: { opposite_dir: loc1, ... }, ... }
        self.bfs_graph = {} # { loc: [neighbor1, neighbor2, ...], ... }
        all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                all_locations.add(l1)
                all_locations.add(l2)

                if l1 not in self.adj_map:
                    self.adj_map[l1] = {}
                self.adj_map[l1][direction] = l2

                if l2 not in self.rev_adj_map:
                    self.rev_adj_map[l2] = {}
                self.rev_adj_map[l2][opposite_direction(direction)] = l1

                # Build undirected graph for robot movement BFS
                if l1 not in self.bfs_graph:
                    self.bfs_graph[l1] = []
                self.bfs_graph[l1].append(l2)

                if l2 not in self.bfs_graph:
                    self.bfs_graph[l2] = []
                self.bfs_graph[l2].append(l1)

        self.all_locations = list(all_locations) # Convert to list

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: math.inf for node in self.all_locations}
        if start_node not in distances: # Handle case where start_node might not be in the graph (e.g., isolated location)
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            curr_node = queue.popleft()

            if curr_node in self.bfs_graph:
                for neighbor in self.bfs_graph[curr_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[curr_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current locations of robot and boxes
        robot_loc = None
        box_locations = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] in self.goal_locations: # Only track boxes that have a goal
                 box_locations[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                 clear_locations.add(parts[1])

        # Check if robot_loc is valid
        if robot_loc is None or robot_loc not in self.distances:
             return math.inf # Robot in an unknown or isolated location

        total_box_distance = 0
        all_valid_push_positions = set()

        misplaced_boxes = {box for box, goal_loc in self.goal_locations.items() if box_locations.get(box) != goal_loc}

        if not misplaced_boxes:
            return 0 # Goal reached

        for box in misplaced_boxes:
            box_loc = box_locations.get(box)
            goal_loc = self.goal_locations[box]

            if box_loc is None or box_loc not in self.distances or goal_loc not in self.distances.get(box_loc, {}):
                 # Box location unknown or goal location unknown/unreachable from box location
                 return math.inf

            # Distance for the box to reach its goal (minimum pushes)
            box_dist = self.distances[box_loc][goal_loc]

            if box_dist == math.inf:
                return math.inf # Box cannot reach its goal location

            total_box_distance += box_dist

            # Find valid push positions for this box
            # A valid push moves the box from box_loc to next_box_loc in direction d_push
            # such that next_box_loc is clear and closer to goal_loc.
            # The required robot position is p such that (adjacent p box_loc d_push).
            if box_loc in self.adj_map: # Check if box_loc has outgoing push directions
                for d_push, next_box_loc in self.adj_map[box_loc].items():
                    # Check if pushing towards goal (strictly reduces distance)
                    if next_box_loc in self.distances and goal_loc in self.distances.get(next_box_loc, {}):
                         if self.distances[next_box_loc][goal_loc] < box_dist:
                            # This is a useful push direction (d_push) moving box from box_loc to next_box_loc
                            # The required robot position is p such that (adjacent p box_loc d_push)
                            # We find p using the reverse adjacency map: rev_adj_map[box_loc][d_push]
                            if box_loc in self.rev_adj_map and d_push in self.rev_adj_map[box_loc]:
                                 required_robot_pos = self.rev_adj_map[box_loc][d_push]

                                 # Check if the target location for the box (next_box_loc) is clear in the current state
                                 if next_box_loc in clear_locations:
                                     all_valid_push_positions.add(required_robot_pos)

        # If there are misplaced boxes but no valid push positions, it's a deadlock
        if total_box_distance > 0 and not all_valid_push_positions:
             return math.inf

        # Calculate minimum robot distance to any valid push position
        min_robot_push_distance = math.inf
        if all_valid_push_positions:
            # Robot cannot move to a location occupied by a box
            occupied_by_box = set(box_locations.values())

            for p in all_valid_push_positions:
                if p != robot_loc and p not in occupied_by_box:
                     if p in self.distances[robot_loc]:
                         min_robot_push_distance = min(min_robot_push_distance, self.distances[robot_loc][p])

        # If there are misplaced boxes but the robot cannot reach any valid push position
        if total_box_distance > 0 and min_robot_push_distance == math.inf:
             return math.inf

        # If total_box_distance is 0, goal is reached.
        if total_box_distance == 0:
            return 0

        # Otherwise, return the sum.
        return total_box_distance + min_robot_push_distance
