import re
from collections import deque
# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def parse_location(loc_str):
    """Parses a location string like 'loc_X_Y' into a tuple (X, Y)."""
    match = re.match(r'loc_(\d+)_(\d+)', loc_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def bfs(graph, start_node):
    """
    Performs BFS from a start node to find shortest distances to all reachable nodes.
    graph: adjacency list representation {node: [neighbor1, neighbor2, ...]}
    start_node: the node to start BFS from
    Returns: dictionary {node: distance}
    """
    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        if current_node in graph: # Ensure node exists in graph keys
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


# Assuming Heuristic base class is available from heuristics.heuristic_base
# If not, a minimal definition like below might be needed:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of shortest path distances for each box
    to its goal location on the grid graph, plus the shortest path distance
    from the robot to the nearest box that is not at its goal.
    Distances are computed on the graph defined by 'adjacent' facts.
    Returns float('inf') if the state is likely unsolvable (e.g., box/robot
    in invalid/unreachable location).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        extracting box goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the graph of traversable locations based on adjacent facts
        self.graph = {} # Adjacency list: { (r1, c1): [(r2, c2), ...], ... }

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1_str, loc2_str, direction = parts[1], parts[2], parts[3]
                loc1_coords = parse_location(loc1_str)
                loc2_coords = parse_location(loc2_str)

                if loc1_coords is None or loc2_coords is None:
                    # Skip invalid location formats in adjacent facts
                    continue

                if loc1_coords not in self.graph:
                    self.graph[loc1_coords] = []
                if loc2_coords not in self.graph:
                     self.graph[loc2_coords] = []

                # Add bidirectional edges for adjacency
                self.graph[loc1_coords].append(loc2_coords)
                self.graph[loc2_coords].append(loc1_coords)

        # Remove duplicates from adjacency lists
        for node in self.graph:
             self.graph[node] = list(set(self.graph[node]))

        # Precompute shortest path distances between all pairs of traversable locations
        self.distances = {} # { ((r1, c1), (r2, c2)): distance, ... }
        for start_node in self.graph:
            # BFS from each node to find distances to all others within the graph
            node_distances = bfs(self.graph, start_node)
            for end_node, dist in node_distances.items():
                self.distances[(start_node, end_node)] = dist

        # Store goal locations for each box (as coordinates)
        self.box_goals = {} # { box_name: (row, col), ... }
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box_name, goal_loc_str = parts[1], parts[2]
                goal_loc_coords = parse_location(goal_loc_str)
                if goal_loc_coords is not None:
                    self.box_goals[box_name] = goal_loc_coords
                # Note: We don't add goal locations to self.graph unless they
                # appeared in adjacent facts. This ensures we only compute
                # distances within the actual traversable grid. Goals outside
                # this grid will result in infinite heuristic.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Sum of shortest path distances for each box to its goal
        + shortest path distance from robot to nearest unsolved box.
        Returns float('inf') if the state is likely unsolvable (e.g., box/robot
        in invalid/unreachable location).
        """
        state = node.state

        # Find current location of robot and boxes
        current_box_locations_str = {} # { box_name: loc_str, ... }
        robot_location_str = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj_name, loc_str = parts[1], parts[2]
                # Check if the object is a box by seeing if it has a goal
                if obj_name in self.box_goals:
                    current_box_locations_str[obj_name] = loc_str
            elif parts[0] == 'at-robot' and len(parts) == 2:
                 robot_location_str = parts[1]

        total_heuristic = 0
        unsolved_boxes_coords = [] # List of (box_name, current_coords) for unsolved boxes

        # Calculate sum of box-to-goal distances
        for box_name, goal_coords in self.box_goals.items():
            current_loc_str = current_box_locations_str.get(box_name)

            # This case should ideally not happen in a valid state representation
            if current_loc_str is None:
                 # Box is expected but not found in state facts
                 return float('inf') # Indicate unsolvable/invalid state

            current_coords = parse_location(current_loc_str)

            # Check for invalid location formats
            if current_coords is None or goal_coords is None:
                 return float('inf') # Invalid location string format

            # If box is at goal, continue
            if current_coords == goal_coords:
                continue

            # Check if current or goal location is in the traversable graph
            if current_coords not in self.graph or goal_coords not in self.graph:
                 # Box is in/needs to go to a non-traversable location (wall/outside grid)
                 return float('inf') # Likely unsolvable

            # Get distance from precomputed table
            dist = self.distances.get((current_coords, goal_coords), float('inf'))

            if dist == float('inf'):
                 # Goal is unreachable from current location within the traversable graph
                 return float('inf') # Likely unsolvable

            total_heuristic += dist
            unsolved_boxes_coords.append((box_name, current_coords))

        # Add robot distance if there are unsolved boxes
        if unsolved_boxes_coords:
            robot_coords = parse_location(robot_location_str)

            # Check if robot location is valid and traversable
            if robot_coords is None or robot_coords not in self.graph:
                 # Robot is in an invalid or non-traversable location
                 return float('inf') # Robot cannot reach any box

            min_robot_to_box_dist = float('inf')
            for box_name, box_coords in unsolved_boxes_coords:
                 # box_coords is guaranteed to be in self.graph here
                 dist = self.distances.get((robot_coords, box_coords), float('inf'))
                 min_robot_to_box_dist = min(min_robot_to_box_dist, dist)

            if min_robot_to_box_dist == float('inf'):
                 # Robot cannot reach any of the unsolved boxes within the traversable graph
                 return float('inf') # Likely unsolvable

            total_heuristic += min_robot_to_box_dist

        # If total_heuristic is 0 here, it means unsolved_boxes_coords was empty,
        # which means all boxes are at their goals. The heuristic is 0. Correct.

        return total_heuristic
