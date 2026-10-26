from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    return fact[1:-1].split()

# BFS function for shortest path on the location graph
def bfs(start, targets, graph, traversable_locations=None):
    """
    Performs BFS to find the shortest distance from start to any target location.

    Args:
        start (str): The starting location.
        targets (set[str]): The set of target locations.
        graph (dict[str, list[str]]): The adjacency list representation of the location graph.
        traversable_locations (set[str], optional): Locations the path *can* go through.
                                                    If None, all locations in the graph are traversable.

    Returns:
        int: The shortest distance, or float('inf') if no target is reachable.
    """
    # If start is not in the graph and not a target, it's isolated/invalid for pathfinding
    # This check assumes all valid locations are either in the graph keys or are targets.
    # Given how the graph is built from adjacent facts, any location mentioned should be a key.
    # Keeping this check for robustness.
    if start not in graph and start not in targets:
         return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc in targets:
            return dist

        # If current_loc is not in graph keys, it has no outgoing edges.
        # Stop exploring from this location.
        if current_loc not in graph:
             continue

        for neighbor in graph[current_loc]:
            is_traversable = (traversable_locations is None) or (neighbor in traversable_locations)
            if is_traversable and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf')


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its goal location, ignoring obstacles for the box itself (relaxed distance).
    2. The shortest path distance for the robot from its current location to
       any location adjacent to any box that is not yet at its goal. The robot
       path considers obstacles (locations not clear).

    # Assumptions
    - The goal is a conjunction of (at box location) predicates.
    - The grid structure is defined by the 'adjacent' predicates, forming a graph.
    - The cost of any action (move or push) is 1.

    # Heuristic Initialization
    - Build the bidirectional location graph from 'adjacent' facts.
    - Extract the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Calculate the first component: Sum of box-to-goal distances.
       - For each box not at its goal:
         - Find its current location and its goal location.
         - Compute the shortest path distance between these two locations on the
           location graph, considering all locations traversable (ignoring obstacles).
         - Add this distance to the total box distance sum.
       - If any box is unreachable from its goal in this relaxed graph, the state
         is likely unsolvable, return infinity.
    5. Calculate the second component: Robot distance to a push position.
       - Identify all locations that are adjacent to any box that is not yet at its goal. These are the potential target locations for the robot.
       - Identify the locations that are currently clear in the state. These are the traversable locations for the robot.
       - Compute the shortest path distance for the robot from its current location
         to any of the target adjacent locations, only traversing through clear locations.
       - If the robot cannot reach any target adjacent location, the state is likely
         unsolvable, return infinity.
    6. The total heuristic value is the sum of the box distance sum and the robot distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        Args:
            task (Task): The planning task object.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the bidirectional location graph from adjacent facts
        # Store as {loc: [neighbor1, neighbor2, ...]}
        self.location_graph = {}

        all_locations_set = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                # Add bidirectional edges
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)

        # Ensure all locations mentioned in adjacent facts are keys in the graph
        for loc in all_locations_set:
             if loc not in self.location_graph:
                 self.location_graph[loc] = []


        # Store goal locations for each box
        self.goal_locations = {}
        # Goals are conjunctions, iterate through them
        for goal in self.goals:
             parts = get_parts(goal)
             # Check if it's an (at box location) predicate
             if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                 box, location = parts[1], parts[2]
                 self.goal_locations[box] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        Args:
            node (Node): The search node containing the current state.

        Returns:
            int or float('inf'): The estimated cost to the goal.
        """
        state = node.state

        # 1. Identify current locations
        current_box_locations = {}
        current_robot_location = None
        current_clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('box'):
                    current_box_locations[obj] = loc
            elif parts[0] == 'at-robot':
                 current_robot_location = parts[1]
            elif parts[0] == 'clear':
                 current_clear_locations.add(parts[1])

        # 2. Identify boxes not at goal
        boxes_to_move = [
            box for box, loc in current_box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        ]

        # 3. If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        # 4. Calculate sum of box-to-goal distances (relaxed)
        box_dist_sum = 0
        for box in boxes_to_move:
            start_loc = current_box_locations[box]
            goal_loc = self.goal_locations[box]
            # BFS for box ignores obstacles, so traversable_locations is None
            dist = bfs(start_loc, {goal_loc}, self.location_graph, traversable_locations=None)
            if dist == float('inf'):
                # Box cannot reach its goal even in relaxed graph
                return float('inf')
            box_dist_sum += dist

        # 5. Calculate robot distance to closest adjacent location of a box to move
        target_adjacent_locations = set()
        for box in boxes_to_move:
            box_loc = current_box_locations[box]
            # Add all neighbors of the box's current location as potential robot targets
            if box_loc in self.location_graph:
                 target_adjacent_locations.update(self.location_graph[box_loc])

        # If there are boxes to move, there should be adjacent locations unless
        # the box is in a completely isolated node, which makes it unsolvable.
        # If target_adjacent_locations is empty, it implies unsolvability.
        if not target_adjacent_locations:
             return float('inf')

        # Robot can only move through locations that are currently clear.
        # The robot's current location is NOT clear, but it is the starting point.
        # The BFS starts from current_robot_location and only traverses to neighbors
        # that are in current_clear_locations.
        robot_dist = bfs(current_robot_location, target_adjacent_locations, self.location_graph, traversable_locations=current_clear_locations)

        if robot_dist == float('inf'):
             # Robot cannot reach any location adjacent to a box that needs moving
             return float('inf')

        # 6. Total heuristic
        return box_dist_sum + robot_dist
