from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed). The number of args
              must match the number of components in the fact.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing
    the shortest path distances for each box to its goal location and adding
    the shortest path distance from the robot to the closest box that is
    not yet at its goal.

    # Assumptions
    - Each box has a unique goal location specified in the task goals.
    - The grid structure is defined by `adjacent` predicates, forming an
      undirected graph where adjacency is symmetric.
    - Shortest path distances on the adjacency graph approximate the minimum
      number of moves/pushes required.
    - The heuristic ignores complex interactions like boxes blocking paths
      or dead-end configurations, and simplifies the robot's movement cost.

    # Heuristic Initialization
    - Build an undirected adjacency graph of locations from the `adjacent` static facts.
    - Pre-calculate shortest path distances between all pairs of locations using BFS.
    - Store the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot (`L_robot`). If not found or invalid, return infinity.
    2. Identify the current location of each box (`box_locations`).
    3. Identify the set of boxes that are not currently at their goal locations (`boxes_to_move`).
    4. If `boxes_to_move` is empty, the state is a goal state, return 0.
    5. Calculate the sum of shortest path distances from the current location of each box in `boxes_to_move` to its corresponding goal location. For each box, if its current location is not in the graph or cannot reach its goal, return infinity. Let this sum be `box_goal_sum_dist`.
    6. Find the box `b_closest` in `boxes_to_to_move` that has the minimum shortest path distance from `L_robot` to its current location (`dist(L_robot, current_location(b))`). If the robot cannot reach any box needing movement, return infinity.
    7. Calculate `robot_closest_box_dist = dist(L_robot, current_location(b_closest))`.
    8. The heuristic value is `box_goal_sum_dist + robot_closest_box_dist`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and storing goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the undirected adjacency graph for distance calculation
        self.graph = defaultdict(set) # loc -> {neighbor_loc}
        self.locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            # Match adjacent facts like (adjacent loc_1_1 loc_1_2 right)
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2 = parts[1], parts[2]
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Adjacency is symmetric
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Pre-calculate all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Match goal facts like (at box1 loc_2_4)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_node):
        """
        Perform BFS starting from start_node to find distances to all reachable nodes.
        Returns a dictionary {location: distance}.
        """
        distances = {node: float('inf') for node in self.locations}
        # Only start BFS if the start_node is actually in the graph
        if start_node not in self.locations:
             return distances # Return distances dictionary with all infinities

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.graph: # Ensure the node has neighbors
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """
        Get the pre-calculated shortest path distance between two locations.
        Returns infinity if locations are not in the graph or not connected.
        """
        # Ensure both locations are in the graph before looking up distance
        if loc1 not in self.locations or loc2 not in self.locations:
             return float('inf')
        # BFS result stores distance from start_node (loc1) to loc2
        return self.distances[loc1].get(loc2, float('inf'))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        # If robot location isn't found or not in graph, return infinity (unsolvable state)
        if robot_location is None or robot_location not in self.locations:
             return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                 if obj in self.goal_locations: # Only track boxes that have goals
                     box_locations[obj] = loc

        boxes_to_move = []
        box_goal_sum_dist = 0

        # Calculate sum of box-to-goal distances and identify boxes needing move
        for box, goal_location in self.goal_locations.items():
            current_box_location = box_locations.get(box)

            # If box is missing or already at goal, skip
            if current_box_location is None or current_box_location == goal_location:
                continue

            # If box location is not in the graph, it's unreachable/invalid
            if current_box_location not in self.locations:
                 return float('inf')

            box_dist = self.get_distance(current_box_location, goal_location)

            # If box cannot reach its goal location on the static graph, return infinity
            if box_dist == float('inf'):
                 return float('inf')

            box_goal_sum_dist += box_dist
            boxes_to_move.append(box)

        # If all boxes are at their goals, the heuristic is 0
        if not boxes_to_move:
            return 0

        # Find the closest box that needs moving
        min_robot_box_dist = float('inf')

        for box in boxes_to_move:
            current_box_location = box_locations[box] # We know it's not None and in locations from previous loop

            robot_dist_to_box = self.get_distance(robot_location, current_box_location)

            # If robot cannot reach this box on the static graph, this state might be unsolvable.
            # Returning infinity here helps prune such states.
            if robot_dist_to_box == float('inf'):
                 return float('inf')

            min_robot_box_dist = min(min_robot_box_dist, robot_dist_to_box)

        # If min_robot_box_dist is still infinity, it means there are boxes to move,
        # but the robot cannot reach any of them. This implies unsolvability.
        if min_robot_box_dist == float('inf'):
             return float('inf')


        # The heuristic is the sum of box-to-goal distances plus the robot's
        # distance to the closest box that needs moving.
        return box_goal_sum_dist + min_robot_box_dist
