from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections
import math

# Helper function for parsing facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function for BFS distances
def bfs_distances(graph, start_node):
    """
    Performs BFS from start_node on the given graph to find distances to all reachable nodes.
    Returns a dictionary {node: distance}. Unreachable nodes are not included.
    Assumes graph is an adjacency dictionary where keys are nodes and values are lists of neighbors.
    """
    distances = {}
    queue = collections.deque()

    # Ensure start_node is a valid node in the graph
    if start_node not in graph:
        return {}

    queue.append(start_node)
    distances[start_node] = 0
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        dist = distances[current_node]

        # Neighbors must be in the graph keys (or at least valid nodes)
        # The graph construction should ensure neighbors are valid nodes that are keys in the graph if they have outgoing edges.
        for neighbor in graph.get(current_node, []): # Use .get for safety
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                queue.append(neighbor)

    return distances

# Helper function for single BFS distance
def bfs_distance(graph, start_node, end_node):
    """
    Performs BFS from start_node to end_node on the given graph.
    Returns the distance or float('inf') if unreachable or nodes are invalid.
    Assumes graph is an adjacency dictionary.
    """
    if start_node == end_node:
        # Check if the node is actually a valid node in the graph
        if start_node in graph:
             return 0
        else:
             # Should not happen if start_node comes from valid locations
             return float('inf')

    # Ensure start node is valid in the graph to start BFS
    if start_node not in graph:
         return float('inf')

    distances = bfs_distances(graph, start_node)
    return distances.get(end_node, float('inf'))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location to its goal location, considering only valid moves on the static location graph (defined by 'adjacent' predicates). This estimates the minimum number of pushes required for all boxes.
    2. The shortest path distance for the robot from its current location to the nearest clear location that is adjacent to any box requiring a push. This distance is calculated on the graph of currently clear locations, estimating the cost for the robot to get into a position to perform a useful push.

    # Assumptions
    - The problem is defined on a graph of locations connected by 'adjacent' predicates.
    - Boxes need to be pushed to specific goal locations.
    - The cost of any action (move or push) is 1.
    - The heuristic assumes that moving a box one step towards its goal requires one push action, and the robot needs to reach a suitable position (a clear adjacent location) to perform this push.
    - If a box's goal is unreachable on the static location graph, or if the robot cannot reach any clear location adjacent to a box that needs moving, the state is considered a dead end (heuristic returns infinity).

    # Heuristic Initialization
    - Extracts all unique location names present in the task's facts.
    - Builds the static location graph (adjacency list) from 'adjacent' facts.
    - Extracts the goal location for each box from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state to identify the robot's current location, the current location of each box, and the set of currently clear locations.
    2. Identify which boxes are not yet at their goal locations. If all boxes are at their goals, the heuristic is 0.
    3. Calculate the first component (Box Distance Sum):
       - Initialize sum = 0.
       - For each box not at its goal:
         - Calculate the shortest path distance from the box's current location to its goal location using BFS on the static location graph (`self.adj`).
         - If the goal is unreachable for any box, return `float('inf')` (dead end).
         - Add this distance to the sum.
    4. Calculate the second component (Robot Access Distance):
       - Identify the set of "access locations": these are locations that are currently clear AND are adjacent (in the static location graph) to any box that is not yet at its goal.
       - If no such access locations exist, return `float('inf')` (robot cannot get into a push position).
       - Build a dynamic graph for robot movement: nodes are the robot's current location plus all currently clear locations. Edges connect adjacent locations if the destination location is in the set of currently clear locations.
       - Perform a BFS from the robot's current location on this robot movement graph to find distances to all reachable nodes.
       - Find the minimum distance from the robot's current location to any of the "access locations" identified in step 4a.
       - If none of the access locations are reachable by the robot, return `float('inf')` (robot cannot reach a push position).
       - This minimum distance is the Robot Access Distance.
    5. The total heuristic value is the sum of the Box Distance Sum (step 3) and the Robot Access Distance (step 4).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        all_facts = task.facts # All possible ground facts in the domain/problem

        # Extract all unique location names from all possible ground facts
        self.all_locations = set()
        for fact in all_facts:
             parts = get_parts(fact)
             # Infer location arguments based on predicate arity and type in domain
             if parts and parts[0] in ['at-robot', 'clear']:
                 if len(parts) > 1: self.all_locations.add(parts[1])
             elif parts and parts[0] == 'at': # (at ?o ?l)
                 if len(parts) > 2: self.all_locations.add(parts[2])
             elif parts and parts[0] == 'adjacent': # (adjacent ?l1 ?l2 ?d)
                 if len(parts) > 2:
                     self.all_locations.add(parts[1])
                     self.all_locations.add(parts[2])
             # Add other predicates if they introduce locations (e.g., 'airport', 'location' in Logistics, but not in Sokoban)
             # For Sokoban, the above predicates cover all location uses.


        # Build the static location graph from "adjacent" facts.
        self.adj = {}
        for loc in self.all_locations:
            self.adj[loc] = [] # Initialize adjacency list for all locations

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent':
                if len(parts) > 2:
                    loc1, loc2 = parts[1], parts[2]
                    # Add bidirectional edges as adjacent is symmetric
                    if loc1 in self.adj:
                        self.adj[loc1].append(loc2)
                    if loc2 in self.adj:
                        self.adj[loc2].append(loc1)

        # Remove duplicates from adjacency lists (optional but clean)
        for loc in self.adj:
            self.adj[loc] = list(set(self.adj[loc]))


        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are of the form (at box_name location_name)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1].startswith('box'):
                package, location = parts[1], parts[2]
                # Ensure goal location is a known location
                if location in self.all_locations:
                    self.goal_locations[package] = location
                # else: problem might be malformed, ignore this goal or handle error

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse state
        robot_loc = None
        box_locations = {} # {box_name: location_name}
        clear_locations = set() # {location_name}

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot' and len(parts) > 1:
                robot_loc = parts[1]
            elif parts and parts[0] == 'at' and len(parts) > 2 and parts[1].startswith('box'): # Assuming objects starting with 'box' are boxes
                box_locations[parts[1]] = parts[2]
            elif parts and parts[0] == 'clear' and len(parts) > 1:
                clear_locations.add(parts[1])

        # --- Component 1: Box Distance Sum ---
        box_goal_distance_sum = 0
        needed_boxes_exist = False
        boxes_not_at_goal = []

        # Iterate through all boxes we know about from the goals
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # If the box is not in the state or not at its goal
            # (A box should always be in the state if the problem is well-formed)
            if current_loc is None:
                 # This indicates a problem state representation issue or a box disappeared
                 # Treat as unreachable goal? Or ignore? Let's return inf.
                 return float('inf')

            if current_loc != goal_loc:
                needed_boxes_exist = True
                boxes_not_at_goal.append(box)
                # Calculate box distance using the static location graph (self.adj)
                dist = bfs_distance(self.adj, current_loc, goal_loc)
                if dist == float('inf'):
                    # Box is in a location from which its goal is unreachable on the static graph
                    return float('inf') # Dead end
                box_goal_distance_sum += dist

        # If no boxes need moving, we are at the goal
        if not needed_boxes_exist:
            return 0

        # --- Component 2: Robot Access Distance ---
        # Robot needs to reach a clear location adjacent to a box that needs moving.
        locations_adjacent_to_needed_boxes = set()
        for box in boxes_not_at_goal:
            b_loc = box_locations[box]
            # Check neighbors using the static adjacency graph
            for adj_loc in self.adj.get(b_loc, []):
                # Robot needs to reach a *clear* location adjacent to the box
                if adj_loc in clear_locations:
                     locations_adjacent_to_needed_boxes.add(adj_loc)

        # If there are no clear locations adjacent to any needed box,
        # the robot cannot immediately get into a push position.
        # This is likely a dead end unless the robot can clear a path.
        # For this heuristic, we treat this as unreachable.
        if not locations_adjacent_to_needed_boxes:
             return float('inf') # Cannot reach a push position

        # Set of nodes the robot can potentially be on or move into for BFS
        # This includes the robot's current location and all clear locations it can move into.
        robot_bfs_nodes = set(clear_locations)
        if robot_loc: # Robot location should always be valid
             robot_bfs_nodes.add(robot_loc)
        else:
             # Should not happen in a valid state, but handle defensively
             return float('inf') # Robot location unknown

        # Build the robot movement graph: edges connect l1 to l2 if l2 is clear
        robot_adj = {}
        # Initialize robot_adj with all potential nodes
        for l in robot_bfs_nodes:
            robot_adj[l] = []

        for l1 in robot_bfs_nodes:
            # Check neighbors using the static adjacency graph
            for l2 in self.adj.get(l1, []):
                # Robot can move from l1 to l2 if l2 is clear
                if l2 in clear_locations:
                    robot_adj[l1].append(l2)

        # Calculate distances from the robot to all reachable nodes in the robot graph
        distances_from_robot = bfs_distances(robot_adj, robot_loc)

        min_robot_dist = float('inf')
        for access_loc in locations_adjacent_to_needed_boxes:
            # Check if the access_loc is reachable by the robot
            if access_loc in distances_from_robot:
                min_robot_dist = min(min_robot_dist, distances_from_robot[access_loc])

        # If robot cannot reach any of the access locations
        if min_robot_dist == float('inf'):
            return float('inf') # Robot cannot reach a push position

        robot_access_distance = min_robot_dist

        # Total heuristic is sum of box distances + robot access distance
        total_heuristic = box_goal_distance_sum + robot_access_distance

        return total_heuristic
