from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

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

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    number of pushes required for each box to reach its goal location, plus
    the minimum number of robot movements required to reach a position from
    which it can make the first push for any box that needs moving.

    # Assumptions
    - The grid is represented by adjacent facts.
    - All locations mentioned in adjacent facts are reachable from each other
      (or relevant locations are connected).
    - The heuristic ignores potential blockages by other boxes or the robot itself
      when calculating distances and push positions.
    - The heuristic ignores deadlocks (states where a box cannot be moved further).
    - Each box has a unique goal location specified in the problem file.

    # Heuristic Initialization
    - Build a graph of locations based on `adjacent` facts.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Extract the goal location for each box from the task's goal conditions.
    - Define the mapping for opposite directions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify the goal location for each box (precomputed during initialization).
    3. Initialize the total heuristic cost to 0.
    4. For each box that is not currently at its goal location:
       a. Calculate the shortest path distance from the box's current location
          to its goal location using the precomputed distances. This distance
          represents the minimum number of pushes required for this box.
       b. Add this distance to the total heuristic cost.
       c. Keep track of these boxes that need to be moved.
    5. If there are boxes that need to be moved:
       a. Find the minimum distance from the robot's current location to a
          valid "push position" for any of these boxes.
       b. A valid "push position" for a box at `box_l` aiming for `goal_l`
          is a location `p` adjacent to `box_l` such that `p` is on the
          opposite side of `box_l` from a neighbor `next_l` that lies on
          a shortest path from `box_l` to `goal_l`.
       c. Find such a `next_l` (a neighbor of `box_l` with distance to `goal_l`
          one less than `box_l`'s distance to `goal_l`).
       d. Determine the direction from `box_l` to `next_l`.
       e. Find the location `push_pos` adjacent to `box_l` in the opposite
          of that direction.
       f. Calculate the shortest path distance from the robot's current location
          to this `push_pos`.
       g. Find the minimum such distance over all boxes that need moving.
       h. Add this minimum robot-to-push-position distance to the total heuristic cost.
       i. If no reachable push position is found for any box that needs moving,
          the state is likely unsolvable, and the heuristic should return infinity.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = collections.defaultdict(dict)
        self.opposite_dir = {'down': 'up', 'up': 'down', 'left': 'right', 'right': 'left'}
        locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "adjacent", "*", "*", "*"):
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.graph[l1][l2] = direction
                # Assuming adjacency is symmetric, add the reverse edge
                if direction in self.opposite_dir:
                     self.graph[l2][l1] = self.opposite_dir[direction]
                locations.add(l1)
                locations.add(l2)

        # Precompute all-pairs shortest path distances using BFS
        self.distances = {}
        all_locations = list(locations) # Use the set of locations found from adjacent facts

        for start_loc in all_locations:
            self.distances[start_loc] = {}
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while queue:
                current_loc, dist = queue.popleft()
                if current_loc in self.graph: # Ensure current_loc is a valid node in the graph
                    for neighbor in self.graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_curr_l = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_curr_l = get_parts(fact)[1]
                break

        if robot_curr_l is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             # Returning infinity indicates this state is likely invalid or unsolvable
             return float('inf')

        # Find box locations
        box_curr_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                box, location = get_parts(fact)[1], get_parts(fact)[2]
                if box in self.goal_locations: # Only consider boxes that have a goal
                    box_curr_locations[box] = location

        total_cost = 0
        boxes_to_move = [] # Store (box_name, box_l, goal_l) for boxes not at goal

        for box, goal_l in self.goal_locations.items():
            box_l = box_curr_locations.get(box) # Get current location

            if box_l is None:
                 # Box not found in state facts - should not happen in valid states
                 return float('inf') # Indicates an invalid state

            if box_l == goal_l:
                continue # Box is already at its goal

            # Ensure locations are in our precomputed distances graph
            if box_l not in self.distances or goal_l not in self.distances.get(box_l, {}):
                 # Goal is unreachable from current box location
                 return float('inf') # Indicates a likely unsolvable state

            box_dist = self.distances[box_l][goal_l]
            total_cost += box_dist # Add minimum pushes for this box

            boxes_to_move.append((box, box_l, goal_l))

        # If all boxes are at their goals, the heuristic is 0.
        if not boxes_to_move:
            return 0

        # Add cost for the robot to get into position for the first push
        min_robot_to_push_pos_dist = float('inf')

        # Check if robot_curr_l is a valid location in our graph before calculating distances from it
        robot_location_known = robot_curr_l in self.distances

        for box, box_l, goal_l in boxes_to_move:
            # Find a neighbor `next_l` of `box_l` on a shortest path to `goal_l`
            next_loc_on_shortest_path = None
            push_dir = None

            # Check if box_l is in the graph keys before iterating neighbors
            if box_l in self.graph:
                for next_l, dir in self.graph[box_l].items():
                    # Ensure next_l is in precomputed distances from box_l
                    # Check if next_l is on a shortest path to goal_l
                    if next_l in self.distances and goal_l in self.distances.get(next_l, {}):
                       if self.distances[next_l][goal_l] == self.distances[box_l][goal_l] - 1:
                           next_loc_on_shortest_path = next_l
                           push_dir = dir
                           break # Found one shortest path step

            if next_loc_on_shortest_path is not None and push_dir in self.opposite_dir:
                required_robot_dir = self.opposite_dir[push_dir]
                # Find the location `push_pos` adjacent to `box_l` in `required_robot_dir`
                push_pos = None
                 # Check if box_l is in the graph keys before iterating neighbors
                if box_l in self.graph:
                    for neighbor, dir in self.graph[box_l].items():
                        if dir == required_robot_dir:
                            push_pos = neighbor
                            break

                if push_pos is not None and robot_location_known:
                     # Ensure push_pos is in precomputed distances from robot location
                     if push_pos in self.distances.get(robot_curr_l, {}):
                         # Add distance from robot to this potential push position
                         min_robot_to_push_pos_dist = min(min_robot_to_push_pos_dist, self.distances[robot_curr_l][push_pos])
                     # else: Robot cannot reach this specific push_pos, ignore it for the minimum calculation

        # If the robot cannot reach any valid push position for any box that needs moving,
        # the state is likely unsolvable.
        if min_robot_to_push_pos_dist == float('inf'):
             # This happens if either robot_location_known is False, or no reachable push_pos was found for any box.
             # If robot_location_known is False, total_cost is already inf from the check at the start.
             # If robot_location_known is True but no reachable push_pos, total_cost might be finite (sum of box distances).
             # In this case, the state is unsolvable by pushing boxes.
             return float('inf')

        # Add the minimum robot movement cost
        total_cost += min_robot_to_push_pos_dist

        return total_cost
