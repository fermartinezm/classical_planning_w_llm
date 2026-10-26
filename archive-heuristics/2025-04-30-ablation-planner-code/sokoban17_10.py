from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
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
    return None # Should not happen for valid directions

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on the graph to find shortest distances and predecessors.

    Args:
        graph: Adjacency list representation {node: [(neighbor, direction), ...]}
        start_node: The starting node for the BFS.

    Returns:
        A tuple (distances, predecessors) where:
        distances: {node: distance from start_node}
        predecessors: {node: parent_node in the shortest path tree}
    """
    distances = {node: float('inf') for node in graph}
    predecessors = {node: None for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        curr = queue.popleft()

        # If curr is unreachable, skip its neighbors (distance remains inf)
        if distances[curr] == float('inf'):
            continue

        for neighbor, direction in graph.get(curr, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[curr] + 1
                predecessors[neighbor] = curr
                queue.append(neighbor)

    return distances, predecessors

def get_first_step_info(start_node, end_node, predecessors, graph):
    """
    Finds the first step (neighbor of start_node) on a shortest path to end_node
    and the direction of that step.

    Args:
        start_node: The starting node.
        end_node: The target node.
        predecessors: The predecessors map from BFS starting at start_node.
        graph: The graph used for BFS {node: [(neighbor, direction), ...]}

    Returns:
        A tuple (first_step_node, direction) or (None, None) if no path exists
        or start_node is the same as end_node.
    """
    # If no path exists or start is the same as end
    if predecessors.get(end_node) is None or start_node == end_node:
        return None, None

    curr = end_node
    # Trace back from end_node until we find the node whose predecessor is start_node
    while predecessors.get(curr) != start_node:
        # If we hit None before reaching start_node, something is wrong or path doesn't exist
        if predecessors.get(curr) is None:
             return None, None # Should be caught by initial check, but safety
        curr = predecessors[curr]

    first_step = curr

    # Find the direction from start_node to first_step
    for neighbor, direction in graph.get(start_node, []):
        if neighbor == first_step:
            return first_step, direction

    return None, None # Should not happen if first_step is a valid neighbor


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions (pushes and robot moves)
    required to move each box to its goal location. It sums the minimum number
    of pushes needed for each box (shortest path on the box-push graph) and
    the minimum number of robot moves needed to reach the required position
    for the *first* push of that box.

    # Assumptions
    - The grid structure and possible movements are defined by the `adjacent`
      predicates in the static facts.
    - Each box has a single specific goal location defined in the goal state.
    - The heuristic calculates shortest paths on the static grid, ignoring
      dynamic obstacles (`clear` predicate) except for the robot's current
      position when calculating its initial move cost. This is an optimistic
      assumption that paths can eventually be cleared.
    - The cost of moving a box is the number of pushes. Each push requires
      the robot to be in a specific adjacent location. The heuristic only
      explicitly models the robot's movement cost to reach the position for
      the *first* push for each box, ignoring subsequent robot repositioning
      costs between pushes.
    - The total heuristic is the sum of individual box costs, ignoring
      interactions or coordination between moving multiple boxes.

    # Heuristic Initialization
    - Parses static facts to build the location graph (for robot movement)
      and the box-push graph (for box movement).
    - Builds a mapping from a location and a direction to the location
      behind it (needed to find the robot's required push position).
    - Computes all-pairs shortest paths for robot movement on the location graph.
    - Computes shortest path distances and the information about the first step
      (next location and direction) for box pushes on the box-push graph.
    - Parses goal conditions to map each box to its goal location.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize the total heuristic value to 0.
    3. For each box that is not yet at its goal location:
       a. Find the shortest path (sequence of pushes) from the box's current
          location to its goal location using the precomputed box-push graph
          information. The length of this path is the minimum number of pushes
          required for this box (`dist_b_g`). If the goal is unreachable
          via pushes, the heuristic for this box is infinity.
       b. Determine the location (`r_0`) where the robot must be positioned
          to perform the *first* push action on the shortest path found in step 3a.
          This location is adjacent to the box's current location, in the
          direction opposite to the first step of the box's push path.
       c. Calculate the shortest path distance (`dist_r_r0`) for the robot
          to move from its current location to the required initial push
          location (`r_0`) using the precomputed robot movement shortest paths.
          If `r_0` is unreachable by the robot, the heuristic for this box is infinity.
       d. The heuristic contribution for this box is `dist_b_g + dist_r_r0`.
       e. Add this contribution to the total heuristic value.
    4. Return the total heuristic value. If any box is in an unsolvable state
       (e.g., goal unreachable), the total heuristic will be infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing graph information."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Build location graph and location_behind mapping
        self.location_graph = {} # loc -> [(neighbor, direction), ...]
        self.location_behind = {} # (loc, direction) -> loc_behind
        locations = set()

        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, l1, l2, d = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                if l1 not in self.location_graph:
                    self.location_graph[l1] = []
                # Add edge for robot movement
                self.location_graph[l1].append((l2, d))
                # Store location behind l2 in direction d is l1
                self.location_behind[(l2, d)] = l1

        self.locations = list(locations) # Store list of all locations

        # 2. Build box-push graph
        self.box_push_graph = {} # loc -> [(pushable_neighbor, direction), ...]
        for l1 in self.locations:
            self.box_push_graph[l1] = []
            # Iterate through possible robot moves from l1 to find pushable directions
            # A box at l1 can be pushed to l2 in direction d if robot is at l_prev
            # where (adjacent l_prev l1 d) and (adjacent l1 l2 d)
            # This means l_prev is location_behind[(l1, d)] and l2 is neighbor of l1 in direction d
            for l2, d in self.location_graph.get(l1, []):
                 # Check if there is a location behind l1 in direction d
                 # If location_behind[(l1, d)] exists, it means robot can get behind l1
                 # to push towards l2 in direction d.
                 if self.location_behind.get((l1, d)) is not None:
                    self.box_push_graph[l1].append((l2, d))


        # 3. Compute robot shortest paths on the location graph
        self.robot_shortest_paths = {} # (start_loc, end_loc) -> distance
        for start_loc in self.locations:
            distances, _ = bfs(self.location_graph, start_loc)
            for end_loc in self.locations:
                self.robot_shortest_paths[(start_loc, end_loc)] = distances[end_loc]

        # 4. Compute box push shortest path info on the box-push graph
        self.box_push_shortest_path_info = {} # (start_loc, end_loc) -> {'distance': dist, 'first_step': next_loc, 'direction': dir_to_next}
        for start_loc in self.locations:
            distances, predecessors = bfs(self.box_push_graph, start_loc)
            for end_loc in self.locations:
                dist = distances[end_loc]
                if dist == float('inf') or dist == 0:
                     self.box_push_shortest_path_info[(start_loc, end_loc)] = {'distance': dist, 'first_step': None, 'direction': None}
                else:
                    first_step, direction = get_first_step_info(start_loc, end_loc, predecessors, self.box_push_graph)
                    self.box_push_shortest_path_info[(start_loc, end_loc)] = {'distance': dist, 'first_step': first_step, 'direction': direction}


        # 5. Parse goal conditions
        self.box_goals = {} # box -> goal_loc
        # Goals can be a single fact or a conjunction (and ...)
        if isinstance(self.goals, str): # Single goal fact
             if match(self.goals, "at", "*", "*"):
                 _, box, goal_loc = get_parts(self.goals)
                 self.box_goals[box] = goal_loc
        elif isinstance(self.goals, frozenset): # Conjunction of goal facts
             for goal_fact in self.goals:
                 if match(goal_fact, "at", "*", "*"):
                     _, box, goal_loc = get_parts(goal_fact)
                     self.box_goals[box] = goal_loc
        # Note: Assuming goal only contains 'at' predicates for boxes


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_loc = get_parts(fact)
                break
        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf')

        # Get current box locations
        box_locations = {} # box -> current_loc
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Check if the object is a box by seeing if it's in our goal list
                if obj in self.box_goals:
                     box_locations[obj] = loc

        total_heuristic = 0

        # Calculate heuristic for each box
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)

            # If box is not in state or not at goal
            if current_loc is None:
                 # Box is missing? Should not happen in valid states.
                 return float('inf')

            if current_loc == goal_loc:
                continue # Box is already at its goal

            # Get box push path info from current location to goal location
            push_path_info = self.box_push_shortest_path_info.get((current_loc, goal_loc))

            if push_path_info is None or push_path_info['distance'] == float('inf'):
                # Goal is unreachable for this box via pushes on the static graph
                return float('inf') # Unsolvable state

            dist_b_g = push_path_info['distance']
            first_push_step = push_path_info['first_step']
            dir_to_first_push_step = push_path_info['direction']

            # Find the required robot location for the first push
            # This is the location behind the box's current location
            # in the direction opposite to the first push direction.
            dir_opp = opposite_direction(dir_to_first_push_step)
            required_robot_loc_for_first_push = self.location_behind.get((current_loc, dir_opp))

            # If required_robot_loc_for_first_push is None, it means the box is
            # next to a boundary/wall in the direction the robot needs to be.
            # This should ideally be prevented by the box_push_graph construction,
            # as a push is only possible if the robot can get behind the box.
            # If we reached here, it means BFS on box_push_graph found a path,
            # implying the first step is pushable, which implies required_robot_loc_for_first_push exists.
            # Asserting this for safety during development, but in theory, it should not be None.
            # assert required_robot_loc_for_first_push is not None, f"Internal error: Cannot find location behind {current_loc} in direction {dir_opp} for first push step {first_push_step}"
            if required_robot_loc_for_first_push is None:
                 # This state indicates a logical inconsistency between graph construction and path finding,
                 # or a complex scenario not handled. Treat as unsolvable.
                 return float('inf')


            # Calculate robot movement cost to reach the required push location
            dist_r_r0 = self.robot_shortest_paths.get((robot_loc, required_robot_loc_for_first_push), float('inf'))

            if dist_r_r0 == float('inf'):
                 # Robot cannot reach the required push location on the static graph
                 return float('inf') # Unsolvable state

            # Heuristic contribution for this box: pushes + robot moves to get ready for first push
            h_box = dist_b_g + dist_r_r0

            total_heuristic += h_box

        return total_heuristic
