# Helper function to parse PDDL fact string
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Example: '(at box1 loc_4_4)' -> ['at', 'box1', 'loc_4_4']
    # Example: '(adjacent loc_4_2 loc_4_3 right)' -> ['adjacent', 'loc_4_2', 'loc_4_3', 'right']
    return fact[1:-1].split()

# Helper function for BFS on the graph
def bfs(graph, start):
    """Computes shortest path distances from start to all reachable nodes."""
    distances = {location: float('inf') for location in graph}
    if start not in graph:
        # Start location might not be in the graph if it's isolated
        # and not mentioned in adjacent facts or goals.
        # This shouldn't happen in typical Sokoban problems but handle defensively.
        return distances

    distances[start] = 0
    queue = deque([start]) # Use deque for efficient pop(0)
    visited = {start}

    while queue:
        current_loc = queue.popleft() # Use popleft()

        if current_loc in graph: # Ensure current_loc is a valid key
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

    return distances

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its goal location.
    2. The shortest path distance from the robot's current location to the
       nearest box that is not yet at its goal location.
    Distances are computed on the static adjacency graph of locations, ignoring
    dynamic obstacles (other boxes and the robot).

    Assumptions:
    - The PDDL instance uses 'loc_r_c' naming convention for locations, although
      the heuristic relies only on the 'adjacent' facts to build the graph.
    - The graph of locations defined by 'adjacent' facts is connected, at least
      within the relevant areas containing initial and goal positions of boxes
      and the robot. Unreachable locations will result in infinite distance.
    - The goal is specified as a conjunction of '(at box location)' facts.

    Heuristic Initialization:
    - Parses the goal facts to identify the target location for each box.
    - Builds an undirected graph of locations based on the 'adjacent' static facts.
    - Collects all unique locations mentioned in adjacent facts and goal facts.
    - Computes all-pairs shortest paths on this graph using BFS and stores them
      for quick lookup during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If all goal boxes are at their
       respective goal locations, the heuristic is 0.
    2. Find the current location of the robot by iterating through the state facts.
    3. Find the current location of each box that has a goal location by iterating
       through the state facts.
    4. Initialize the heuristic value `h` to 0.
    5. Identify the set of boxes that are not currently at their goal locations.
    6. For each box in this set:
       a. Get its current location and its goal location.
       b. Retrieve the precomputed shortest path distance between the current box location
          and its goal location from the distance table.
       c. If the distance is infinite (unreachable), the state is likely unsolvable
          or in a bad path, return infinity.
       d. Add this distance to `h`. This estimates the minimum number of pushes
          required for this box, ignoring obstacles.
    7. If there are boxes not at their goal:
       a. Initialize minimum robot-box distance to infinity.
       b. For each box that needs moving, retrieve the precomputed shortest path
          distance from the robot's current location to the box's current location.
       c. Update the minimum robot-box distance.
       d. If the minimum robot-box distance is still infinity (robot cannot reach
          any box needing movement), return infinity.
       e. Add this minimum distance to `h`. This estimates the minimum number of
          robot moves required to reach a box that needs pushing, ignoring obstacles
          and the specific positioning needed for a push.
    8. Return the calculated `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goal locations for each box
        self.goal_locations = {}
        all_locations = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
                all_locations.add(location)

        # 2. Build the adjacency graph from static facts
        graph = {}

        # Collect all locations mentioned in adjacent facts and add to set
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2 = parts[1], parts[2]
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Initialize graph with all known locations
        graph = {loc: [] for loc in all_locations}

        # Populate graph with bidirectional edges from adjacent facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2 = parts[1], parts[2]
                # Ensure locations exist in the graph dict before appending
                if loc1 in graph and loc2 in graph:
                    graph[loc1].append(loc2)
                    graph[loc2].append(loc1) # Add reverse edge for undirected graph
                # If a location from adjacent fact wasn't in all_locations, it's an issue,
                # but the current logic adds all locations first.

        self.graph = graph # Store graph for potential debugging or future use

        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.graph:
            self.distances[start_loc] = bfs(self.graph, start_loc)

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        # This check is important for the heuristic to be 0 at the goal
        if self.goals <= state:
             return 0

        # Find current robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
                break

        if robot_loc is None or robot_loc not in self.graph:
             # Robot location unknown or not in the known graph (shouldn't happen in valid problems)
             return float('inf')

        # Find current box locations for goal boxes
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in self.goal_locations:
                box_locs[parts[1]] = parts[2]

        # Calculate heuristic components
        h = 0
        boxes_to_move = []

        # Component 1: Sum of box-goal distances
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locs.get(box)
            if current_loc is None or current_loc not in self.graph:
                 # Box location unknown or not in the known graph (shouldn't happen)
                 return float('inf')

            if current_loc != goal_loc:
                boxes_to_move.append(box)
                # Get distance from precomputed table
                # Use .get() with default float('inf') for robustness
                dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                if dist == float('inf'):
                    # Goal location unreachable from current box location
                    return float('inf')
                h += dist

        # Component 2: Distance from robot to nearest box that needs moving
        # Only add this component if there are boxes still needing to reach their goal
        if boxes_to_move:
            min_robot_box_dist = float('inf')
            for box in boxes_to_move:
                box_loc = box_locs[box]
                # Get distance from precomputed table
                # Use .get() with default float('inf') for robustness
                dist = self.distances.get(robot_loc, {}).get(box_loc, float('inf'))
                min_robot_box_dist = min(min_robot_box_dist, dist)

            if min_robot_box_dist == float('inf'):
                 # Robot cannot reach any box that needs moving
                 return float('inf')

            h += min_robot_box_dist

        return h
