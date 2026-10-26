# Import necessary modules
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this is the base class provided by the planner environment

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Helper to split a PDDL fact string into parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

# Helper function to match a fact against a pattern
def match(fact, *args):
    """Helper to match a fact against a pattern."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to get the opposite direction
def get_opposite_direction(direction):
    """Helper to get the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen in this domain

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing two components for each box that is not yet at its goal location:
    1. The shortest path distance for the box from its current location to its goal location. This represents the minimum number of push actions required for the box itself, assuming an unobstructed path and available robot.
    2. The shortest path distance for the robot from its current location to the specific location required to perform the *first* push action for that box towards its goal. This accounts for the robot's movement cost to initiate the box movement.
    The total heuristic is the sum of these costs over all boxes not yet at their goals.

    Assumptions:
    - The locations form a connected graph (or at least, all relevant locations for a problem instance are in the same connected component).
    - The `adjacent` predicates define the movement graph for both the robot (via `move` action) and boxes (via `push` action).
    - The heuristic assumes that moving a box one step requires one push action, and moving the robot one step requires one move action.
    - The heuristic does not consider potential deadlocks (e.g., pushing a box into a corner) or the need to move other boxes or obstacles out of the way. It ignores the `clear` predicate. It is therefore non-admissible.
    - The heuristic assumes that the shortest path for a box is always achievable by the robot.
    - The heuristic only considers the robot's cost to initiate the *first* push for each box, not subsequent repositioning costs between pushes or between different boxes.

    Heuristic Initialization:
    - Parses static facts to build an adjacency list representation of the location graph based on `adjacent` predicates. This graph is used for BFS to find shortest paths for both robot and boxes. The graph is treated as undirected for robot movement (move action) but the adjacency map and push_robot_pos map store directional information crucial for push actions.
    - Stores the goal location for each box by parsing the `task.goals`.
    - Stores a mapping from a box's current location and the desired push direction to the location from which the robot must push (i.e., the location adjacent to the box in the push direction). This is derived from the inverse of the `adjacent` relations used in the push precondition.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Get the current state, robot's location, and current locations of all boxes.
    2.  Initialize total heuristic value `h = 0`.
    3.  For each box specified in the goals:
        a.  Check if the box is already at its goal location. If yes, skip this box as it contributes 0 to the heuristic.
        b.  Find the box's current location in the state. If the box is not found (should not happen in valid states), return infinity.
        c.  Calculate the shortest path distance (`dist_box`) from the box's current location to its goal location using BFS on the location graph. This is the minimum number of pushes needed for the box along the graph edges. If no path exists, return a large value (infinity) as the state is likely unsolvable.
        d.  Add `dist_box` to `h`.
        e.  Find a location `loc_b_next` that is the first step on a shortest path for the box from its current location towards its goal. This is done by finding a neighbor of the current box location that is one step closer to the goal according to the BFS distance from the goal.
        f.  Determine the direction `dir` from the box's current location `loc_b` to `loc_b_next`.
        g.  Determine the required robot location `loc_r` such that `adjacent(loc_r, loc_b, dir)` is true. This is the location "behind" the box relative to the push direction. Use the pre-computed mapping (`self.push_robot_pos`) for this. If no such location exists (e.g., box is against a wall and needs to be pushed away from it, but there's no space behind it), this indicates an issue or unsolvable state via simple pushes; return infinity.
        h.  Calculate the shortest path distance (`dist_robot`) from the robot's current location to `loc_r` using BFS on the location graph. If no path exists, return a large value (infinity).
        i.  Add `dist_robot` to `h`.
    4.  Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts.
        # This graph is treated as undirected for robot movement (move action)
        # and for general distance calculations for boxes.
        self.graph = {}
        # Store mapping from (location, direction) to adjacent location
        self.adj_map = {}
        # Store mapping from (box_location, push_direction) to the required robot location
        # adjacent(?rloc, ?bloc, ?dir) is true implies robot at ?rloc can push box at ?bloc in ?dir
        # So, push_robot_pos[(?bloc, ?dir)] = ?rloc
        self.push_robot_pos = {}

        # Build the graph (bidirectional for robot/distance) and push_robot_pos map
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)

                # Add edge loc1 -> loc2 in direction
                if loc1 not in self.graph:
                    self.graph[loc1] = {}
                self.graph[loc1][direction] = loc2
                self.adj_map[(loc1, direction)] = loc2

                # Add reverse edge loc2 -> loc1 in opposite direction for robot movement
                opp_dir = get_opposite_direction(direction)
                if loc2 not in self.graph:
                     self.graph[loc2] = {}
                self.graph[loc2][opp_dir] = loc1
                self.adj_map[(loc2, opp_dir)] = loc1

                # Store the robot position needed to push from loc2 in direction
                # Robot needs to be at loc1 to push from loc2 in direction
                self.push_robot_pos[(loc2, direction)] = loc1


        # Store goal locations for boxes
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is (at ?b ?l)
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

    def bfs_distance(self, start_loc, end_loc):
        """Calculates shortest path distance between two locations using BFS on the graph."""
        if start_loc == end_loc:
            return 0

        # Ensure start_loc is in the graph, otherwise it's unreachable
        if start_loc not in self.graph:
             return float('inf')

        queue = deque([(start_loc, 0)]) # (location, distance)
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # Check neighbors from the graph
            if current_loc in self.graph:
                for direction, neighbor_loc in self.graph[current_loc].items():
                    if neighbor_loc == end_loc:
                        return dist + 1
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        queue.append((neighbor_loc, dist + 1))

        # If end_loc is not reachable from start_loc
        return float('inf')

    def find_first_step_towards_goal(self, start_loc, goal_loc):
        """
        Finds the location and direction of the first step on a shortest path
        from start_loc to goal_loc. Returns (next_loc, direction) or (None, None) if no path.
        """
        if start_loc == goal_loc:
            return None, None # Already at goal

        # Use BFS from the goal to find distances to all reachable nodes.
        # Then find a neighbor of start_loc that is one step closer to goal_loc.

        # Ensure goal_loc is in the graph
        if goal_loc not in self.graph:
             return None, None

        queue_bfs = deque([goal_loc])
        visited_bfs = {goal_loc}
        distances_from_goal = {goal_loc: 0}

        while queue_bfs:
            current_loc = queue_bfs.popleft()
            current_dist = distances_from_goal[current_loc]

            if current_loc in self.graph:
                # Iterate through neighbors (both directions)
                for direction, neighbor_loc in self.graph[current_loc].items():
                    if neighbor_loc not in visited_bfs:
                        visited_bfs.add(neighbor_loc)
                        distances_from_goal[neighbor_loc] = current_dist + 1
                        queue_bfs.append(neighbor_loc)


        # Now find a neighbor of start_loc that has distance_from_goal = distances_from_goal[start_loc] - 1
        if start_loc not in distances_from_goal:
             # start_loc is not reachable from goal_loc (or vice versa in a connected graph)
             return None, None

        target_dist = distances_from_goal[start_loc] - 1

        if start_loc in self.graph:
             for direction, neighbor_loc in self.graph[start_loc].items():
                  if neighbor_loc in distances_from_goal and distances_from_goal[neighbor_loc] == target_dist:
                       # Found a neighbor that is one step closer. Return it and the direction
                       # from start_loc to neighbor_loc.
                       return neighbor_loc, direction

        # Should find a step if start_loc is not goal_loc and is reachable from goal_loc
        # This might happen if start_loc is a dead end or isolated, but BFS from goal
        # should handle reachability. If start_loc is in distances_from_goal but no
        # neighbor is closer, it implies an issue with graph structure or BFS logic.
        # Given the simple grid structure of Sokoban, this case should ideally not occur
        # for reachable states.
        return None, None # Error case

    def __call__(self, node):
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_loc = get_parts(fact)
                break
        if robot_loc is None:
             # Robot location not found, invalid state
             return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        total_cost = 0

        # Consider each box that needs to reach a goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # Get current location of the box

            if current_loc is None:
                 # Box location not found, invalid state
                 return float('inf')

            # If box is already at goal, cost for this box is 0
            if current_loc == goal_loc:
                continue

            # 1. Cost for the box to reach the goal (minimum pushes)
            # This is the shortest path distance on the location graph
            dist_box = self.bfs_distance(current_loc, goal_loc)
            if dist_box == float('inf'):
                # Goal is unreachable for this box
                return float('inf')
            total_cost += dist_box

            # 2. Cost for the robot to get into position for the first push
            # Find the first step location and direction for the box towards the goal
            # We need the location adjacent to current_loc that is on a shortest path to goal_loc
            first_step_loc, push_direction = self.find_first_step_towards_goal(current_loc, goal_loc)

            if first_step_loc is None:
                 # This implies current_loc is the goal_loc (handled by continue)
                 # or goal_loc is unreachable from current_loc (handled by dist_box check)
                 # or there's an issue finding the step. If dist_box > 0, this is an error.
                 if dist_box > 0:
                      # This case should ideally not be reached if dist_box > 0 and graph is valid
                      # Returning infinity indicates a problem with the state or graph structure
                      return float('inf')
                 else: # dist_box == 0, but current_loc != goal_loc, contradiction
                      # This case should also not be reached
                      continue # Skip this box if somehow it passed the initial check but dist_box is 0


            # Determine the required robot location to push from current_loc to first_step_loc
            # The robot must be at loc_r such that adjacent(loc_r, current_loc, push_direction) is true.
            required_robot_loc = self.push_robot_pos.get((current_loc, push_direction))

            if required_robot_loc is None:
                 # This means there is no location from which the robot can push the box
                 # in the required direction according to the adjacent facts.
                 # This state might be unsolvable via simple pushes. Penalize heavily.
                 return float('inf')

            # Calculate robot distance to the required push location
            dist_robot = self.bfs_distance(robot_loc, required_robot_loc)
            if dist_robot == float('inf'):
                # Robot cannot reach the required position
                return float('inf')
            total_cost += dist_robot

        return total_cost
