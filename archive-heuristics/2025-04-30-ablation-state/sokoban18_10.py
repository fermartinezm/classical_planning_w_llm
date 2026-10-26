import re
from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total cost by summing the minimum number of pushes required for each box to reach its goal location (calculated as shortest path distance on the location graph) and adding the minimum robot movement cost required to get into position to make the first push for any of the boxes that are not yet at their goal.

    # Assumptions
    - The grid structure and adjacency are defined by the `adjacent` facts.
    - Robot movement cost between adjacent locations is 1.
    - Pushing a box costs 1 action and moves the box one step and the robot one step (to the box's previous location).
    - The shortest path distance for a box on the location graph is a reasonable estimate for the minimum number of pushes required for that box, ignoring dynamic obstacles (other boxes, robot) and potential deadlocks caused by static obstacles (walls) that prevent the robot from reaching the push-from position.
    - The robot can reach any location reachable on the graph unless explicitly blocked in the current state (which this heuristic doesn't fully model dynamically).
    - Deadlocks where a box is pushed into an unrecoverable position (e.g., corner) are partially detected if the box's goal is unreachable on the static graph or if the required push-from location is unreachable for the robot on the static graph.

    # Heuristic Initialization
    - Parses `adjacent` facts to build an undirected graph for robot movement and a directed adjacency map for push directions.
    - Identifies all unique locations from the `adjacent` facts.
    - Precomputes all-pairs shortest path distances for robot movement using BFS on the undirected graph, covering all identified locations.
    - Extracts goal locations for each box from the task goals.
    - Precomputes shortest path distances for each box to its goal on the undirected location graph using BFS. During this BFS, it also identifies the required 'push-from' location for the very first step on *a* shortest path.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify the current location of the robot and all boxes from the state facts.
    3. Initialize `total_box_distance = 0` and `min_robot_cost_to_push = infinity`.
    4. Iterate through each box that has a goal location defined in the task.
    5. For each box, if it is not currently at its goal location:
        a. Mark that there are boxes needing movement (`has_boxes_to_move = True`).
        b. Look up the precomputed shortest path distance (`box_dist`) for this box from its current location to its goal location on the location graph. This represents the minimum number of pushes needed for this box in an ideal scenario.
        c. If `box_dist` is infinity, the goal is unreachable for this box on the static graph, indicating an unsolvable state. Return infinity.
        d. Add `box_dist` to `total_box_distance`.
        e. Look up the precomputed 'push-from' location required for the first step of a shortest path for this box.
        f. If a valid 'push-from' location exists:
            i. Calculate the shortest path distance for the robot from its current location to this 'push-from' location using precomputed distances.
            ii. If the robot cannot reach this 'push-from' location, it indicates a potential deadlock or unsolvable state relative to the robot's position. Return infinity.
            iii. Update `min_robot_cost_to_push` with the minimum distance found so far across all boxes needing movement.
        g. If `box_dist > 0` but no valid 'push-from' location exists for the first step, it implies the box cannot be pushed from its current location towards the goal. This indicates a static deadlock. Return infinity.
    8. If `has_boxes_to_move` is true and `min_robot_cost_to_push` is still infinity, it means there are boxes to move but the robot cannot reach a valid push position for *any* of them. Return infinity.
    9. The heuristic value is `total_box_distance + min_robot_cost_to_push`. This sums the estimated pushes for all boxes and adds the minimum robot movement cost to initiate the process for any box. If no boxes need moving, the robot cost part is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts
        to build the location graph and precompute distances.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Build location graph (undirected for robot movement) and directed adjacency (for pushes)
        self.location_graph = defaultdict(set)
        self.directed_adj = defaultdict(dict)
        self.locations = set()
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Assume symmetric adjacency for robot movement
                self.directed_adj[loc1][direction] = loc2
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Precompute robot shortest paths (all-pairs BFS)
        self.robot_distances = {}
        for start_loc in self.locations:
            self.robot_distances[start_loc] = self._bfs_distances(start_loc, self.location_graph, self.locations)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Precompute box shortest paths and first push locations
        # This BFS finds shortest path in terms of locations visited (pushes).
        # We also store the 'push-from' location needed for the very first step.
        self.box_path_info = {} # {(start_loc, goal_loc): (distance, first_step_loc, first_push_from_loc)}
        for start_loc in self.locations:
            for goal_loc in self.locations:
                if start_loc == goal_loc:
                    self.box_path_info[(start_loc, goal_loc)] = (0, None, None)
                    continue

                # BFS to find a shortest path for the box on the location graph
                q = deque([(start_loc, [start_loc])])
                visited = {start_loc}
                found = False
                while q:
                    curr_loc, path = q.popleft()

                    if curr_loc == goal_loc:
                        box_dist = len(path) - 1
                        first_step_loc = path[1] if len(path) > 1 else None
                        first_push_from_loc = None
                        if first_step_loc:
                            # Find direction from curr_loc to first_step_loc
                            dir_to_first_step = None
                            # Iterate through directed neighbors of curr_loc
                            for d, neighbor in self.directed_adj.get(curr_loc, {}).items():
                                if neighbor == first_step_loc:
                                    dir_to_first_step = d
                                    break # Found the direction

                            if dir_to_first_step and dir_to_first_step in self.opposite_dir:
                                # Find location behind curr_loc in opposite direction
                                opp_dir = self.opposite_dir[dir_to_first_step]
                                if opp_dir in self.directed_adj.get(curr_loc, {}):
                                     first_push_from_loc = self.directed_adj[curr_loc][opp_dir]

                        self.box_path_info[(start_loc, goal_loc)] = (box_dist, first_step_loc, first_push_from_loc)
                        found = True
                        break # Found a shortest path (the first one found by BFS)

                    # Explore neighbors for box movement (any adjacent location on the undirected graph)
                    # A box can potentially move to any adjacent location if the robot can get behind it
                    # This BFS finds the shortest path *distance* for the box on the grid.
                    for neighbor in self.location_graph.get(curr_loc, []):
                         if neighbor not in visited:
                             visited.add(neighbor)
                             q.append((neighbor, path + [neighbor]))

                if not found:
                     # Goal is unreachable for the box on the graph
                     self.box_path_info[(start_loc, goal_loc)] = (float('inf'), None, None)


    def _bfs_distances(self, start_node, graph, all_nodes):
        """Performs BFS to find shortest distances from start_node to all reachable nodes."""
        distances = {node: float('inf') for node in all_nodes}
        if start_node not in distances:
             # Start node is not in the set of known locations. Should not happen in valid problems.
             return {} # Indicate error or isolation

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Check if neighbor is a known node before accessing its distance
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_robot_distance(self, loc1, loc2):
        """Gets the precomputed shortest path distance for the robot."""
        # Ensure both locations are in the precomputed distances map
        if loc1 not in self.robot_distances or loc2 not in self.robot_distances.get(loc1, {}):
            # This means loc2 is unreachable from loc1 on the static graph
            return float('inf')
        return self.robot_distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current box locations and robot location
        current_box_locations = {}
        robot_location = None
        # Use match helper for robustness
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj_name, loc_name = get_parts(fact)[1], get_parts(fact)[2]
                 if obj_name in self.goal_locations: # It's a box we need to move
                      current_box_locations[obj_name] = loc_name
            elif match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]

        # If robot location is unknown, state is invalid or unsolvable from here
        if robot_location is None:
            return float('inf')

        total_box_distance = 0
        min_robot_cost_to_push = float('inf')
        has_boxes_to_move = False

        # Iterate through all boxes that have a goal defined
        for box, goal_l in self.goal_locations.items():
            box_l = current_box_locations.get(box)

            # If box is not found in the current state, it's an invalid state
            if box_l is None:
                 return float('inf')

            # If box is not at its goal location
            if box_l != goal_l:
                has_boxes_to_move = True

                # Get precomputed box path info
                box_dist, first_step_loc, first_push_from_loc = self.box_path_info.get((box_l, goal_l), (float('inf'), None, None))

                # If box cannot reach goal on the static graph, it's unsolvable
                if box_dist == float('inf'):
                    return float('inf')

                # Add the estimated number of pushes for this box
                total_box_distance += box_dist

                # Calculate robot cost to get into position for the first push for this box
                # Only consider if a valid first push position was found during precomputation
                if first_push_from_loc:
                    robot_to_push_pos_dist = self.get_robot_distance(robot_location, first_push_from_loc)
                    # If robot cannot reach the required push position for this box, it's unsolvable
                    if robot_to_push_pos_dist == float('inf'):
                         return float('inf')
                    # Update the minimum robot cost needed to start pushing *any* box
                    min_robot_cost_to_push = min(min_robot_cost_to_push, robot_to_push_pos_dist)
                else:
                    # If box_dist > 0 but no valid first_push_from_loc was found,
                    # it means the box cannot be pushed from its current location towards the goal.
                    # This indicates a static deadlock.
                    if box_dist > 0: # This check is technically redundant here but explicit
                         return float('inf')


        # If there are boxes that need to be moved, but the robot cannot reach
        # a push position for *any* of them (min_robot_cost_to_push is still infinity),
        # the state is unsolvable.
        if has_boxes_to_move and min_robot_cost_to_push == float('inf'):
             return float('inf')

        # The heuristic is the sum of estimated pushes for all boxes plus
        # the minimum robot movement cost to get into position for the first push
        # of any box that needs moving. If no boxes need moving, the robot cost part is 0.
        # If has_boxes_to_move is true, min_robot_cost_to_push is finite here.
        return total_box_distance + (min_robot_cost_to_push if has_boxes_to_move else 0)
