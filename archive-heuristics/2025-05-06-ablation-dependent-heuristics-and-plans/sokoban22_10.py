from fnmatch import fnmatch
import collections
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts gracefully
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Or raise an error, depending on expected input quality
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

opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    costs for each box that is not yet at its goal location. The estimated cost
    for a single box is the sum of the shortest path distance for the robot to
    reach a position from which it can push the box towards its goal, and the
    shortest path distance for the box itself to reach the goal location.

    # Assumptions
    - The environment is a grid implicitly defined by the 'adjacent' predicates.
    - The heuristic assumes that boxes can be moved independently towards their
      goals, ignoring potential blocking by other boxes or complex robot
      maneuvering around boxes.
    - The cost of moving the robot one step and pushing a box one step is
      considered equal (1 action). A push action involves the robot moving
      into the box's previous location, effectively moving both robot and box.
      The heuristic counts robot moves and box pushes separately and sums them.
    - Deadlocks (boxes in unresolvable positions) are detected if a path for a
      misplaced box to its goal or a required robot push location does not exist
      on the static location graph, resulting in an infinite heuristic value.

    # Heuristic Initialization
    - The adjacency graph of locations is built from the 'adjacent' static facts.
      This graph represents possible movements for the robot and pushes for boxes
      on an empty grid.
    - The goal location for each box is extracted from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot by finding the fact `(at-robot ?l)`.
    2. Identify the current location of each box that is relevant to the goal
       by finding facts `(at ?b ?l)` where `?b` is a box from the goal.
    3. Initialize the total heuristic cost to 0.
    4. For each box specified in the goal conditions (`self.box_goals`):
       a. Determine the box's current location and its goal location.
       b. If the box is already at its goal location, its cost contribution is 0.
          Continue to the next box.
       c. If the box is not at its goal:
          i. Calculate the shortest path distance from the box's current location
             to its goal location using BFS on the static location graph (`self.adj_list`).
             This distance (`box_dist`) represents the minimum number of 'push'
             actions required for the box to reach the goal if the path were clear.
             Also, determine the direction of the first step (`first_step_dir_from_box`)
             on this shortest path from the box's perspective.
          ii. If no path exists from the box to the goal (`box_dist == float('inf')`),
              the state is likely a deadlock or unsolvable from this point based on
              the static graph. Return infinity as the heuristic value.
          iii. Determine the required location for the robot to push the box
               in the direction of the first step towards the goal. The robot must
               be adjacent to the box, on the side opposite the direction the box
               needs to move. Find this required robot location (`required_robot_push_location`)
               by looking for a location adjacent to the box's current location
               in the direction `opposite_dir[first_step_dir_from_box]`.
          iv. If the `required_robot_push_location` does not exist (e.g., trying to push
              from a boundary where no adjacent location exists in the required direction),
              the state is likely a deadlock. Return infinity.
          v. Calculate the shortest path distance from the robot's current location
              to this `required_robot_push_location` using BFS on the static location graph.
              This distance (`robot_dist_to_push`) is the estimated number of 'move'
              actions for the robot to get into position for the first push.
          vi. If the required pushing location is unreachable by the robot
              (`robot_dist_to_push == float('inf')`), the state is likely a deadlock.
              Return infinity.
          vii. The estimated cost for this box is the sum of the robot's travel distance
               to the initial push position (`robot_dist_to_push`) and the box's
               minimum travel distance (`box_dist`). Add this cost to the total heuristic.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and extracting
        box goal locations.
        """
        self.goals = task.goals
        self.static = task.static

        # Build the adjacency list graph from static facts
        self.adj_list = collections.defaultdict(list)
        self.locations = set() # Keep track of all locations mentioned in adjacent facts
        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.adj_list[loc1].append((loc2, direction))
                # Add the reverse connection
                if direction in opposite_dir:
                     self.adj_list[loc2].append((loc1, opposite_dir[direction]))
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Store goal locations for each box
        self.box_goals = {}
        # Assuming task.goals is a frozenset of goal facts
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if match(goal_fact, "at", "*", "*"):
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

    def bfs(self, start_loc, end_loc):
        """
        Performs BFS on the static location graph to find the shortest distance
        and the first step direction from start_loc to end_loc.

        Args:
            start_loc: The starting location.
            end_loc: The target location.

        Returns:
            (distance, first_step_direction) or (float('inf'), None) if no path.
        """
        # Check if locations exist in our graph (derived from adjacent facts)
        if start_loc not in self.locations or end_loc not in self.locations:
             return (float('inf'), None)

        queue = collections.deque([(start_loc, 0, None)]) # (current_loc, distance, first_step_dir)
        visited = {start_loc: (0, None)} # {location: (distance, first_step_dir)}

        while queue:
            curr_loc, dist, path_first_dir = queue.popleft()

            if curr_loc == end_loc:
                return (dist, path_first_dir)

            # Check if curr_loc has any adjacent locations defined
            if curr_loc not in self.adj_list:
                 continue # Should not happen if curr_loc is in self.locations

            for neighbor, direction in self.adj_list[curr_loc]:
                if neighbor not in visited:
                    new_dist = dist + 1
                    # The first step direction is the direction taken from the start_loc
                    # If we are one step away from start_loc (dist == 0), the first_dir is the direction taken to get to neighbor.
                    # Otherwise, it's the first_dir inherited from the parent (path_first_dir).
                    new_path_first_dir = direction if dist == 0 else path_first_dir
                    visited[neighbor] = (new_dist, new_path_first_dir)
                    queue.append((neighbor, new_dist, new_path_first_dir))

        return (float('inf'), None) # No path found

    def find_push_location(self, box_loc, push_direction_from_box):
        """
        Finds the location adjacent to box_loc in the specified direction.
        This is the location the robot must be in to push the box.

        Args:
            box_loc: The location of the box.
            push_direction_from_box: The direction the robot must be relative to the box
                                     (e.g., 'left' if pushing 'right').

        Returns:
            The required robot location (L_push) or None if no such location exists.
        """
        if box_loc not in self.adj_list:
            return None
        for neighbor, direction in self.adj_list[box_loc]:
            if direction == push_direction_from_box:
                return neighbor
        return None


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the current location of the robot.
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        if robot_location is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Robot location unknown or invalid state

        # 2. Identify the current location of each box.
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                # Only track locations for objects that are boxes in the goal
                if obj_name in self.box_goals:
                    current_box_locations[obj_name] = get_parts(fact)[2]

        # 3. Initialize the total heuristic cost to 0.
        total_heuristic = 0

        # 4. For each box specified in the goal conditions:
        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box)

            # If a box from the goal is not found in the state, it's an invalid state.
            if current_location is None:
                 return float('inf') # Box from goal is not present in the state

            # b. If the box is already at its goal location, its cost contribution is 0.
            if current_location == goal_location:
                continue # Box is home, cost is 0 for this box

            # c. If the box is not at its goal:
            # i. Calculate the shortest path distance from the box's current location
            #    to its goal location using BFS.
            box_dist, first_step_dir_from_box = self.bfs(current_location, goal_location)

            # ii. If no path exists from the box to the goal, the state is likely a deadlock.
            if box_dist == float('inf'):
                return float('inf') # Box cannot reach its goal on the static graph

            # iii. Determine the required location for the robot to push the box
            #      in the direction of the first step towards the goal.
            if first_step_dir_from_box is None or first_step_dir_from_box not in opposite_dir:
                 # This implies the box is at the goal (handled above) or the BFS failed strangely.
                 # If box_dist is finite but first_step_dir_from_box is None, it means dist is 0.
                 # But we already handled current_location == goal_location.
                 # So this case implies an issue with BFS or graph structure for dist > 0.
                 # Treat as unreachable for safety.
                 return float('inf')

            required_push_direction_from_box = opposite_dir[first_step_dir_from_box]
            required_robot_push_location = self.find_push_location(current_location, required_push_direction_from_box)

            # iv. If the `required_robot_push_location` does not exist, the state is likely a deadlock.
            if required_robot_push_location is None:
                 # Cannot find a location adjacent to the box in the required pushing direction.
                 # This means the box cannot be pushed in the required direction.
                 return float('inf')

            # v. Calculate the shortest path distance from the robot's current location
            #     to this required pushing location using BFS.
            robot_dist_to_push, _ = self.bfs(robot_location, required_robot_push_location)

            # vi. If the required pushing location is unreachable by the robot,
            #    the state is likely a deadlock.
            if robot_dist_to_push == float('inf'):
                 return float('inf') # Robot cannot reach the push position on the static graph

            # vii. The estimated cost for this box is the sum of the robot's travel distance
            #     to the initial push position (`robot_dist_to_push`) and the box's
            #     minimum travel distance (`box_dist`). Add this cost to the total heuristic.
            # Each push action costs 1. Robot movement costs 1 per step.
            # The cost is robot_dist_to_push steps + box_dist pushes.
            total_heuristic += robot_dist_to_push + box_dist

        # 5. Return the total heuristic cost.
        return total_heuristic
