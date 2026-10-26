from collections import deque
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function for BFS
def bfs(graph, start_node):
    """
    Performs BFS from a start node to find distances to all reachable nodes.
    graph: adjacency list {node: set(neighbor1, neighbor2, ...)}
    start_node: the node to start BFS from
    Returns: {node: distance}
    """
    # Initialize distances to infinity for all nodes in the graph
    distances = {node: float('inf') for node in graph}

    # If the start node is not in the graph, no other node is reachable from it.
    if start_node not in graph:
        return distances # All distances remain infinity

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph dictionary
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required by summing the shortest
    path distances for each misplaced box from its current location to its goal location.
    The distance is calculated on the grid graph defined by the 'adjacent' predicates.
    This heuristic is a relaxation that ignores the robot's position and clear space
    requirements, focusing solely on the box-to-goal distances.

    # Assumptions
    - The primary cost driver is moving boxes to their goal locations.
    - The cost of moving a box is approximated by the shortest path distance on the grid.
    - Robot movement and clear space requirements are ignored in the distance calculation (relaxation).
    - The heuristic is non-admissible.
    - The goal is defined solely by the locations of the boxes.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an undirected graph of the locations based on the 'adjacent' static facts.
    - Pre-computes all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of every box from the state.
    2. For each box that has a specified goal location (extracted during initialization):
       - Compare the box's current location to its goal location.
       - If the box is not at its goal location:
         - Retrieve the pre-computed shortest path distance between the box's current location and its goal location from the pre-computed distances table.
         - If the goal location is unreachable from the box's current location in the grid graph, the state is likely unsolvable. In this case, return a very high heuristic value (infinity) immediately.
         - Add this distance to a running total.
    3. The total sum of distances for all misplaced boxes is the heuristic value.
    4. If the total sum of distances is 0, it means all boxes are at their goal locations,
       which corresponds to a goal state in this domain (given the typical Sokoban goal definition).
       The heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.
        """
        self.task = task # Store task object if needed later (e.g., for goal_reached check)
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # Assuming objects starting with 'box' are the boxes we care about
                if obj.startswith("box"):
                    self.box_goals[obj] = location

        # Build the undirected graph from adjacent facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Add reverse edge for undirected graph

        # Ensure all locations mentioned in goals are included in the graph nodes
        # even if they are isolated (though unlikely in Sokoban grids).
        for loc in self.box_goals.values():
             if loc not in self.graph:
                  self.graph[loc] = set() # Add location with no neighbors

        # Collect all unique locations found in graph building
        all_graph_locations = list(self.graph.keys())

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in all_graph_locations:
            self.distances[start_node] = bfs(self.graph, start_node)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current box locations.
        current_box_locations = {}
        # Robot location is not needed for this simple heuristic, but we can extract it.
        # robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts:
                predicate = parts[0]
                if predicate == "at" and len(parts) == 3:
                    obj, location = parts[1], parts[2]
                    if obj.startswith("box"):
                        current_box_locations[obj] = location
                # elif predicate == "at-robot" and len(parts) == 2:
                #     robot_loc = parts[1]

        total_heuristic = 0  # Initialize action cost counter.

        # If there are no boxes specified in the goals, the task is trivially solved w.r.t. boxes.
        # The heuristic should be 0.
        if not self.box_goals:
             return 0

        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box)

            # If a box required by the goal is not found in the current state,
            # it indicates an issue or an unsolvable state.
            if current_location is None:
                 # This should not happen in standard Sokoban problems.
                 # Treat as unsolvable.
                 return float('inf')

            if current_location != goal_location:
                # Get the shortest distance from current location to goal location for this box
                # Ensure the current location is a valid key in the distances table
                if current_location not in self.distances:
                     # Location from state was not in the graph built from static facts.
                     # This implies an unreachable location or malformed problem.
                     return float('inf')

                # Look up the distance from the pre-computed table
                dist = self.distances[current_location].get(goal_location)

                # If the goal location is unreachable from the box's current location
                # (distance is infinity or None if goal_location wasn't in graph keys),
                # the state is likely unsolvable.
                if dist is None or dist == float('inf'):
                    return float('inf')

                total_heuristic += dist

        # The total heuristic is the sum of distances.
        # If the sum is 0, it means all boxes are at their goal locations,
        # which is the definition of the goal state in this domain.
        # Thus, h=0 only for goal states is satisfied.
        return total_heuristic
