from collections import deque
# Assuming Heuristic base class is available externally
from heuristics.heuristic_base import Heuristic

# Manual parsing helper
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

# The class name requested is sokobanHeuristic
class sokobanHeuristic(Heuristic):
    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Grid structure and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Collect all unique location strings from adjacent facts
        location_strings = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                location_strings.add(parts[1])
                location_strings.add(parts[2])

        # Ensure locations from goals are included, just in case the goal is outside the adjacent graph
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at' and len(parts) == 3:
                 location_strings.add(parts[2])

        graph_nodes = list(location_strings)
        self.graph = {node: [] for node in graph_nodes}

        # 2. Build adjacency list graph using location strings
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                # Ensure locations are in our collected set before adding edges
                if l1 in self.graph and l2 in self.graph:
                     self.graph[l1].append(l2)
                     # Assuming adjacency is symmetric, add the reverse edge
                     self.graph[l2].append(l1)

        # Remove duplicates from adjacency lists
        for node in self.graph:
            self.graph[node] = list(set(self.graph[node]))

        # 3. Compute all-pairs shortest paths using BFS
        self.dist_map = {}
        for start_node in graph_nodes:
            distances = {node: float('inf') for node in graph_nodes}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                curr = queue.popleft()

                # Check if curr is a valid node in the graph (should be)
                if curr not in self.graph:
                    continue

                for neighbor in self.graph.get(curr, []): # Use .get for safety
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[curr] + 1
                        queue.append(neighbor)

            # Store distances from start_node to all reachable nodes
            for end_node in graph_nodes:
                 if distances[end_node] != float('inf'):
                    self.dist_map[(start_node, end_node)] = distances[end_node]

        # 4. Store goal locations for each box (using location strings)
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box_name, goal_loc_str = parts[1], parts[2]
                # Ensure goal location is in the graph
                if goal_loc_str in self.graph:
                    self.box_goals[box_name] = goal_loc_str
                # else: goal location not in graph? Problematic instance.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Get robot location string
        robot_loc_str = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc_str = parts[1]
                break
        # If robot location is not found or not in the graph, state is likely invalid/unreachable
        if robot_loc_str is None or robot_loc_str not in self.graph:
             return float('inf')

        # 2. Get box locations (using location strings)
        box_current_loc_str = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if it's an 'at' fact for a box we care about (i.e., a box with a goal)
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.box_goals:
                 box_name, loc_str = parts[1], parts[2]
                 box_current_loc_str[box_name] = loc_str

        # 3. Calculate heuristic components
        total_box_distance = 0
        min_box_goal_distance = float('inf')
        loc_str_of_box_closest_to_its_goal = None

        for box_name, goal_loc_str in self.box_goals.items():
            current_loc_str = box_current_loc_str.get(box_name)

            # If a box with a goal is not found in the current state or its location is not in the graph,
            # the state is likely invalid or unsolvable.
            if current_loc_str is None or current_loc_str not in self.graph:
                 return float('inf')

            # Check if the box is already at its goal
            if current_loc_str != goal_loc_str:
                # Check if distance is computable (locations are connected)
                if (current_loc_str, goal_loc_str) not in self.dist_map:
                    # Goal is unreachable for this box. Return infinity.
                    return float('inf')

                d = self.dist_map[(current_loc_str, goal_loc_str)]
                total_box_distance += d

                # Find the box closest to its goal (in terms of distance)
                if d < min_box_goal_distance:
                    min_box_goal_distance = d
                    loc_str_of_box_closest_to_its_goal = current_loc_str

        # If total_box_distance is 0, all boxes are at their goals.
        if total_box_distance == 0:
            return 0

        # Calculate robot distance to the location of the box closest to its goal
        # This location is `loc_str_of_box_closest_to_its_goal`.
        # Ensure robot location and the target box location are in the distance map
        # loc_str_of_box_closest_to_its_goal must be set if total_box_distance > 0
        if loc_str_of_box_closest_to_its_goal is None:
             # This should logically not happen if total_box_distance > 0, but as a safeguard
             return float('inf')

        if (robot_loc_str, loc_str_of_box_closest_to_its_goal) not in self.dist_map:
             # Robot cannot reach the target box location. Problematic state/instance.
             return float('inf')

        robot_distance = self.dist_map[(robot_loc_str, loc_str_of_box_closest_to_its_goal)]

        # The heuristic is the sum of box distances plus the robot's distance to the closest box
        return total_box_distance + robot_distance
