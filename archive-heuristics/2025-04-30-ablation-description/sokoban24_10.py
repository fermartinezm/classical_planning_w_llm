from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for BFS
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes in the graph."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        # Ensure current_node is in graph and has neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location and adding the shortest
    path distance from the robot to the closest box that is not yet at its goal.

    # Assumptions
    - The environment is a grid defined by 'adjacent' predicates.
    - Shortest path distances in the grid are used as a lower bound for movement costs.
    - The heuristic ignores dynamic obstacles (other boxes, robot position) when calculating box-goal distances, assuming a clear path in the underlying grid structure.
    - The heuristic adds the robot's distance to the closest box that needs moving, assuming the robot must reach a box to push it.

    # Heuristic Initialization
    - Builds a graph representation of the grid from 'adjacent' static facts.
    - Precomputes all-pairs shortest path distances in the grid graph using BFS.
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot.
    2. Identify the current location of each box that has a goal location.
    3. Initialize the heuristic value `h` to 0.
    4. Initialize `min_robot_box_dist` to infinity.
    5. Iterate through each box for which a goal location is defined:
       - Get the box's current location and its goal location.
       - If the box is not at its goal location:
         - Add the precomputed shortest path distance from the box's current location to its goal location to `h`.
         - Calculate the precomputed shortest path distance from the robot's current location to the box's current location.
         - Update `min_robot_box_dist` to be the minimum of its current value and the calculated robot-to-box distance.
    6. After iterating through all boxes, if there was at least one box not at its goal (checked via `min_robot_box_dist` not being infinity):
       - Add `min_robot_box_dist` to `h`.
    7. Return the final value of `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the grid graph and precomputing distances."""
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph from adjacent facts
        self.graph = {}
        locations = set()

        # First pass to collect all locations
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1:]
                locations.add(loc1)
                locations.add(loc2)

        # Initialize graph adjacency lists and ensure all locations are keys
        for loc in locations:
            self.graph.setdefault(loc, [])

        # Second pass to add edges (undirected)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1:]
                # Add edge in both directions
                if loc2 not in self.graph[loc1]:
                    self.graph[loc1].append(loc2)
                if loc1 not in self.graph[loc2]:
                    self.graph[loc2].append(loc1)

        # Precompute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.graph:
            self.dist[start_loc] = bfs(self.graph, start_loc)

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at <box> <location>)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                # Assume any object in a goal 'at' predicate is a box we need to move
                self.goal_locations[obj] = loc

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        robot_loc = None
        box_locations = {} # Map box name to location

        # Find current robot and box locations
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                # Only track locations for boxes that have a goal
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        # If robot location is not found or not in graph, something is wrong.
        if robot_loc is None or robot_loc not in self.dist:
             return float('inf') # Should not happen in valid states with connected graphs

        h = 0
        min_robot_box_dist = float('inf')
        boxes_need_moving = False

        # Calculate sum of box-goal distances and find closest box
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # Ensure the box is in the current state and its location is in the graph
            if current_box_loc is None or current_box_loc not in self.dist:
                 return float('inf') # Should not happen in valid states

            if current_box_loc != goal_loc:
                boxes_need_moving = True
                # Add box-goal distance
                if goal_loc in self.dist[current_box_loc]:
                     h += self.dist[current_box_loc][goal_loc]
                else:
                     # Goal location is not reachable from current box location in the grid graph.
                     return float('inf') # Unsolvable path for box

                # Update min robot-box distance
                if current_box_loc in self.dist[robot_loc]:
                     min_robot_box_dist = min(min_robot_box_dist, self.dist[robot_loc][current_box_loc])
                else:
                     # Box location is not reachable from robot location in the grid graph.
                     return float('inf') # Robot cannot reach this box

        # Add the cost for the robot to reach the closest box, if any box needs moving
        if boxes_need_moving:
            if min_robot_box_dist != float('inf'):
                h += min_robot_box_dist
            else:
                 # This case should be covered by the checks inside the loop,
                 # but as a safeguard, return infinity.
                 return float('inf')

        return h
