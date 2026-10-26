from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions (can be outside the class or inside)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components:
    1.  The sum of the minimum number of pushes required for each misplaced box
        to reach its goal location, calculated as the shortest path distance
        on the grid graph ignoring dynamic obstacles.
    2.  The minimum number of robot moves required to reach a position from
        which it can perform the *first* push for any of the misplaced boxes.
        This distance is calculated on the grid graph, treating locations
        occupied by other boxes as obstacles for the robot.

    # Assumptions
    - The grid structure is defined by 'adjacent' facts.
    - Robot can move into 'clear' locations or its current location.
    - Robot cannot move into locations occupied by boxes.
    - Boxes can only be pushed into 'clear' locations (this is implicitly handled
      by the push action preconditions, but the heuristic uses a simplified
      box distance ignoring dynamic obstacles).
    - The goal specifies the final location for each box.
    - The PDDL push action corresponds to standard Sokoban push: robot at L0,
      box at L1, push to L2, robot ends at L1, box at L2.

    # Heuristic Initialization
    - Build the adjacency list representation of the grid graph from 'adjacent' facts.
    - Store the goal locations for each box from the task's goal conditions.
    - Store static facts to determine required robot push positions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state to find the robot's location and the location of each box.
    2.  Identify which boxes are not yet at their goal locations.
    3.  If all boxes are at their goals, the heuristic is 0.
    4.  Initialize the total heuristic `total_h` to 0 and the minimum robot distance
        to a push position `min_robot_dist_to_push_pos` to infinity.
    5.  Identify locations occupied by boxes, as these are obstacles for robot movement.
    6.  For each box that is not at its goal:
        a.  Compute the shortest path distance (number of pushes) from the box's
            current location to its goal location using BFS on the full grid graph
            (ignoring dynamic obstacles like other boxes or the robot). Add this
            distance to `total_h`. If the goal is unreachable for the box, the
            state is likely unsolvable, return infinity.
        b.  Find the first step location (`box_next_loc`) on a shortest path for
            the box from its current location towards its goal. This requires
            computing distances from the goal location backwards.
        c.  Determine the required robot position (`required_robot_pos`) adjacent
            to the box's current location (`box_loc`) such that pushing from
            `required_robot_pos` moves the box from `box_loc` to `box_next_loc`.
            This position is adjacent to `box_loc` in the direction opposite
            of `box_loc` to `box_next_loc`.
        d.  Compute the shortest path distance from the robot's current location
            to this `required_robot_pos` using BFS, treating locations occupied
            by other boxes as obstacles. Update `min_robot_dist_to_push_pos`
            with the minimum distance found across all misplaced boxes.
    7.  If there were misplaced boxes, add `min_robot_dist_to_push_pos` to `total_h`.
        If the robot cannot reach any required push position, return infinity.
    8.  Return the final `total_h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The grid adjacency list from static facts.
        - Goal locations for each box.
        - Static facts for determining push positions.
        """
        self.goals = task.goals
        self.static_facts = task.static # Store static facts

        # Build the adjacency list for the grid graph
        self.adjacency_list = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2 = parts[1], parts[2]
                if loc1 not in self.adjacency_list:
                    self.adjacency_list[loc1] = []
                self.adjacency_list[loc1].append(loc2)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and args[0].startswith('box'):
                box, location = args[0], args[1]
                self.goal_locations[box] = location

        # Define a large value for unreachable locations
        self.UNREACHABLE = float('inf')

    def bfs_distance_map(self, start_loc, occupied_locations, adjacency_list):
        """
        Performs a Breadth-First Search to find shortest path distances from start_loc
        to all reachable locations.

        Args:
            start_loc (str): The starting location.
            occupied_locations (set): A set of locations that cannot be traversed.
            adjacency_list (dict): The graph adjacency list.

        Returns:
            dict: A dictionary mapping reachable locations to their shortest distance from start_loc.
        """
        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        dist_map = {start_loc: 0}

        while queue:
            current_loc, dist = queue.popleft()

            for neighbor in adjacency_list.get(current_loc, []):
                if neighbor not in visited and neighbor not in occupied_locations:
                    visited.add(neighbor)
                    dist_map[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return dist_map

    def get_required_robot_pos(self, box_loc, box_next_loc):
        """
        Finds the location the robot must be at to push the box from box_loc to box_next_loc.
        Assumes box_loc and box_next_loc are adjacent.
        """
        # Find the direction from box_loc to box_next_loc
        direction = None
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] == 'adjacent' and parts[1] == box_loc and parts[2] == box_next_loc:
                 direction = parts[3]
                 break
        if direction is None:
            # This implies box_loc and box_next_loc are not adjacent according to static facts
            # which shouldn't happen if box_next_loc is the first step on a shortest path.
            return None

        # Find the opposite direction
        opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}.get(direction)
        if opposite_direction is None:
             return None # Should not happen with valid directions

        # Find the location adjacent to box_loc in the opposite direction
        required_pos = None
        # We are looking for a location 'prev_loc' such that (adjacent prev_loc box_loc opposite_direction)
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] == 'adjacent' and parts[2] == box_loc and parts[3] == opposite_direction:
                 required_pos = parts[1] # The 'from' location in the adjacent fact
                 break

        return required_pos


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot and box locations in the current state
        robot_loc = None
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                current_box_locations[box] = loc

        total_h = 0
        min_robot_dist_to_push_pos = self.UNREACHABLE
        box_obstacles_for_robot = set(current_box_locations.values())

        # Calculate sum of box distances and find required robot positions
        required_robot_positions = []
        misplaced_boxes_count = 0

        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)
            if current_loc is None:
                 # Should not happen in a valid Sokoban state where all objects are tracked
                 return self.UNREACHABLE # Indicate invalid state

            if current_loc != goal_loc:
                misplaced_boxes_count += 1

                # Calculate box distance (pushes)
                # BFS from current_loc to goal_loc for box distance (no dynamic obstacles)
                box_dist_map = self.bfs_distance_map(current_loc, set(), self.adjacency_list)
                box_dist = box_dist_map.get(goal_loc, self.UNREACHABLE)

                if box_dist == self.UNREACHABLE:
                    # A box is in a location from which its goal is unreachable
                    return self.UNREACHABLE # This state is likely a dead end

                total_h += box_dist

                # Find the first step on a shortest path for the box
                first_step = None
                # We need the distance map from the GOAL back to find the path FROM current_loc
                # So, BFS from goal_loc to get distances TO current_loc
                dist_from_goal_map = self.bfs_distance_map(goal_loc, set(), self.adjacency_list)
                current_dist_from_goal = dist_from_goal_map.get(current_loc, self.UNREACHABLE)

                if current_dist_from_goal == self.UNREACHABLE:
                     # This should be consistent with the box_dist check above
                     return self.UNREACHABLE

                # Find a neighbor 'n' of current_loc such that dist(n, goal_loc) == dist(current_loc, goal_loc) - 1
                # There might be multiple such neighbors if multiple shortest paths exist.
                # We only need one to determine a potential first push.
                for neighbor in self.adjacency_list.get(current_loc, []):
                     if dist_from_goal_map.get(neighbor, self.UNREACHABLE) == current_dist_from_goal - 1:
                         first_step = neighbor
                         break # Found one first step

                if first_step is None:
                     # This implies current_loc is not connected to goal_loc
                     return self.UNREACHABLE

                # Find the required robot position to make this first push
                required_robot_pos = self.get_required_robot_pos(current_loc, first_step)
                if required_robot_pos is None:
                     # This implies invalid grid structure or logic error
                     return self.UNREACHABLE

                required_robot_positions.append(required_robot_pos)


        # If there are misplaced boxes, calculate robot's distance to the nearest required push position
        if misplaced_boxes_count > 0:
             # Robot distance: BFS from robot_loc avoiding locations occupied by boxes
             robot_dist_map = self.bfs_distance_map(robot_loc, box_obstacles_for_robot, self.adjacency_list)

             for push_pos in required_robot_positions:
                  robot_dist = robot_dist_map.get(push_pos, self.UNREACHABLE)
                  min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist)

             if min_robot_dist_to_push_pos == self.UNREACHABLE:
                  # Robot cannot reach any required push position for any misplaced box
                  return self.UNREACHABLE

             total_h += min_robot_dist_to_push_pos

        return total_h
