# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the framework
import math
from collections import deque
from fnmatch import fnmatch

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation for robot distance (undirected graph)
def bfs_robot(graph, start_node):
    """
    Performs BFS on an undirected graph to find shortest distances.
    Graph is {node: set(neighbors)}.
    Returns distances {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    queue = deque([start_node])
    distances[start_node] = 0
    visited = {start_node}

    while queue:
        curr = queue.popleft()

        for neighbor in graph.get(curr, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[curr] + 1
                queue.append(neighbor)

    return distances

# BFS implementation for box path info (directed graph)
def bfs_box_path_info(graph, start_node):
    """
    Performs BFS on a directed graph to find shortest distances,
    the first step location, and the direction of the first step
    from the start_node to any reachable node.
    Graph is {node: {neighbor: direction}}.
    Returns (distances, first_step_loc, first_step_dir).
    """
    distances = {node: float('inf') for node in graph}
    first_step_loc = {node: None for node in graph}
    first_step_dir = {node: None for node in graph}
    queue = deque()
    visited = set()

    # Initialize queue with neighbors of the start_node
    distances[start_node] = 0 # Distance from start to itself is 0
    first_step_loc[start_node] = start_node # First step from start to itself is itself
    first_step_dir[start_node] = None # No direction from start to itself
    visited.add(start_node)

    # Add direct neighbors of start_node to the queue with their info
    for neighbor, direction in graph.get(start_node, {}).items():
        if neighbor not in visited:
            visited.add(neighbor)
            distances[neighbor] = 1
            first_step_loc[neighbor] = neighbor # The neighbor is the first step
            first_step_dir[neighbor] = direction # The direction to the neighbor is the first direction
            queue.append(neighbor) # Add just the neighbor to the queue

    while queue:
        curr = queue.popleft()

        # Propagate first_step_loc and first_step_dir from curr's values
        fs_loc = first_step_loc[curr]
        fs_dir = first_step_dir[curr]

        for neighbor, direction in graph.get(curr, {}).items():
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[curr] + 1
                first_step_loc[neighbor] = fs_loc # Inherit first step from curr
                first_step_dir[neighbor] = fs_dir # Inherit first direction from curr
                queue.append(neighbor)

    return distances, first_step_loc, first_step_dir


# class sokobanHeuristic(Heuristic): # Use this line if inheriting from Heuristic base class
class sokobanHeuristic: # Use this line if Heuristic base class is not provided
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost based on the distance of each box to its goal
    and the robot's distance to the position required for the first push
    towards the goal for each box, plus estimated robot repositioning costs
    for subsequent pushes.

    Heuristic = Sum over misplaced boxes (
        box_distance_to_goal +
        robot_distance_to_first_push_position +
        max(0, box_distance_to_goal - 1) * estimated_robot_reposition_cost
    )
    Where estimated_robot_reposition_cost is 1, based on grid-like movement.
    This simplifies to:
    Heuristic = Sum over misplaced boxes (
        box_distance_to_goal +
        robot_distance_to_first_push_position +
        max(0, box_distance_to_goal - 1)
    )
    If k = box_distance_to_goal:
    Cost = k + robot_dist + max(0, k-1)
    If k=1: Cost = 1 + robot_dist + 0 = 1 + robot_dist
    If k>1: Cost = k + robot_dist + k - 1 = 2k + robot_dist - 1
    This is the formula implemented.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and precomputing
        distances and path information.
        """
        self.goals = task.goals
        static_facts = task.static
        # Include initial state facts to find all locations mentioned
        # Note: task.initial_state might not contain ALL locations in the domain,
        # only those involved in initial facts. We should get all locations from
        # the domain definition or the problem objects list if available,
        # but parsing adjacent facts is a good proxy for connected locations.
        # Let's collect locations from all relevant facts in init and static.
        all_facts_in_init_and_static = task.initial_state | static_facts

        # 1. Extract all locations mentioned in init and static facts
        self.all_locations = set()
        for fact in all_facts_in_init_and_static:
            parts = get_parts(fact)
            if parts[0] in ['at-robot', 'at', 'clear']:
                 if len(parts) > 1: self.all_locations.add(parts[1])
            elif parts[0] == 'adjacent':
                 if len(parts) > 2:
                    self.all_locations.add(parts[1])
                    self.all_locations.add(parts[2])

        # 2. Build graphs from adjacent facts
        # Robot graph (undirected/bidirectional)
        self.robot_graph = {loc: set() for loc in self.all_locations}
        # Box graph (directed) and reverse adjacency for finding push positions
        self.box_graph = {loc: {} for loc in self.all_locations} # {l1: {l2: dir}}
        self.rev_adj_graph = {loc: {} for loc in self.all_locations} # {l2: {l1: dir_from_l1_to_l2}}

        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, l1, l2, direction = get_parts(fact)
                if l1 in self.all_locations and l2 in self.all_locations: # Ensure locations are known
                    self.robot_graph[l1].add(l2)
                    self.robot_graph[l2].add(l1) # Assume robot can move back if adjacent is defined
                    self.box_graph[l1][l2] = direction
                    # Store the direction from l1 to l2 under the key l2 in rev_adj_graph
                    self.rev_adj_graph[l2][l1] = direction

        # 3. Precompute distances and path info using BFS
        self.robot_dist = {}
        for start_loc in self.all_locations:
            self.robot_dist[start_loc] = bfs_robot(self.robot_graph, start_loc)

        self.box_dist = {}
        self.box_first_step = {}
        self.box_first_dir = {}
        for start_loc in self.all_locations:
            distances, first_step_loc, first_step_dir = bfs_box_path_info(self.box_graph, start_loc)
            self.box_dist[start_loc] = distances
            self.box_first_step[start_loc] = first_step_loc
            self.box_first_dir[start_loc] = first_step_dir

        # 4. Store goal locations for boxes and identify all boxes
        self.box_goals = {}
        self.all_boxes = set()
        # Find all box objects mentioned in the initial state
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 box_name, _ = get_parts(fact)
                 # A more robust way would be to parse the :objects section,
                 # but we only have facts. Assume anything starting with 'box'
                 # that appears in an 'at' predicate in the initial state is a box.
                 if box_name.startswith('box'):
                     self.all_boxes.add(box_name)

        # Find goal locations for these boxes
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                box_name, goal_location = get_parts(goal)
                # Only store goals for boxes we identified
                if box_name in self.all_boxes:
                    self.box_goals[box_name] = goal_location
                # Note: If a box exists but has no goal 'at' predicate, it's ignored.
                # If a goal 'at' predicate exists for a non-existent box, it's ignored.

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current robot location
        robot_l = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_l = get_parts(fact)[1]
                break
        # If robot_l is None, the state is likely invalid or terminal (goal reached)
        # Since we already checked for goal reached, this might indicate an issue
        # or an unsolvable state where the robot disappeared.
        if robot_l is None:
             return float('inf') # Should not happen in valid states

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 box_name, loc = get_parts(fact)
                 if box_name in self.all_boxes: # Only track known boxes
                    current_box_locations[box_name] = loc

        total_heuristic = 0

        # Calculate cost for each misplaced box
        for box, goal_l in self.box_goals.items():
            box_l = current_box_locations.get(box)

            # If box_l is None, the box is not 'at' any location. This is an invalid state.
            if box_l is None:
                 return float('inf')

            if box_l == goal_l:
                continue # Box is already at its goal

            # Get box distance to goal (minimum pushes)
            pushes_needed = self.box_dist.get(box_l, {}).get(goal_l, float('inf'))

            if pushes_needed == float('inf'):
                # Box cannot reach its goal from this location
                return float('inf')

            # If pushes_needed is 0, it should have been caught by box_l == goal_l
            if pushes_needed == 0:
                 continue

            # Find the first step location and direction on a shortest path for the box
            first_step_l = self.box_first_step.get(box_l, {}).get(goal_l)
            first_dir = self.box_first_dir.get(box_l, {}).get(goal_l)

            # These should exist if pushes_needed > 0 and finite
            if first_step_l is None or first_dir is None:
                 return float('inf') # Path info missing, indicates problem

            # Find the required robot position for the first push
            # Robot must be at rloc such that (adjacent rloc box_l first_dir) holds.
            # This means rloc is a neighbor of box_l, and the direction FROM rloc TO box_l is first_dir.
            # We look in the reverse adjacency graph: rev_adj_graph[box_l][rloc] == first_dir
            required_robot_pos_for_first_push = None
            for neighbor_of_box_l, dir_from_neighbor_to_box_l in self.rev_adj_graph.get(box_l, {}).items():
                 if dir_from_neighbor_to_box_l == first_dir:
                      required_robot_pos_for_first_push = neighbor_of_box_l
                      break # Found the required position

            if required_robot_pos_for_first_push is None:
                 # This box might be in a position where it cannot be pushed towards its goal
                 # (e.g., against a wall/obstacle on the push side relative to the first step direction).
                 # This is a form of dead end. Return infinity.
                 return float('inf')

            # Get robot distance from its current location to the required push position
            robot_moves_to_push_pos = self.robot_dist.get(robot_l, {}).get(required_robot_pos_for_first_push, float('inf'))

            if robot_moves_to_push_pos == float('inf'):
                 # Robot cannot reach the required push position
                 return float('inf')

            # Calculate cost for this box
            # Cost = pushes_needed + robot_moves_to_push_pos + max(0, pushes_needed - 1)
            # This is equivalent to:
            cost_for_box = pushes_needed + robot_moves_to_push_pos
            if pushes_needed > 1:
                 cost_for_box += (pushes_needed - 1) # Add 1 robot move for each push after the first

            total_heuristic += cost_for_box

        return total_heuristic
