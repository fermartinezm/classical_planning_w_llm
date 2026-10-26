from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest path distances
    for each box from its current location to its goal location. This distance is calculated
    as the minimum number of 'pushes' required for the box, ignoring the robot's position
    and potential blockages by other objects or walls.

    # Assumptions
    - The grid structure is defined by 'adjacent' predicates.
    - 'adjacent' relationships define the connectivity between locations.
    - The primary cost driver is moving boxes. Robot movement and unblocking are secondary or ignored.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Extracts the goal location for each box from the goal state.
    - Builds an undirected adjacency list representation of the grid graph from 'adjacent' static facts.
      This graph is used to compute shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each box specified in the goal state:
       a. Determine the box's current location and its goal location.
       b. If the box is already at its goal location, it contributes 0 to the heuristic.
       c. If the box is not at its goal location, calculate the shortest path distance
          between the box's current location and its goal location on the grid graph.
          This distance represents the minimum number of 'push' actions needed for this
          specific box if it could move freely. A standard Breadth-First Search (BFS)
          is used for this shortest path calculation.
       d. If no path exists between the box's current location and its goal location,
          the state is likely unsolvable, and the heuristic returns infinity.
       e. Add the calculated shortest path distance for the box to the total heuristic cost.
    4. The final heuristic value is the sum of these shortest path distances for all boxes
       not yet at their goal. Robot position and the cost/feasibility of reaching
       the push position are ignored for simplicity and computational efficiency.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's (at box location)
                box, location = args
                self.goal_locations[box] = location

        # Build undirected graph adjacency list for location connectivity
        self.location_graph = {}
        for fact in static_facts:
             # Match (adjacent loc1 loc2 dir)
             if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                # Add edge in both directions if not already present
                if loc2 not in self.location_graph[loc1]:
                     self.location_graph[loc1].append(loc2)
                if loc1 not in self.location_graph[loc2]:
                     self.location_graph[loc2].append(loc1)


    def bfs_locations(self, start, end):
        """
        Performs BFS on the location graph to find the shortest path distance.
        This represents the minimum number of steps (moves or pushes) between locations.
        Returns float('inf') if no path exists.
        """
        if start == end:
            return 0

        # Ensure start and end locations are part of the graph
        if start not in self.location_graph or end not in self.location_graph:
             # If a location isn't in the graph, it's likely a wall or unreachable area
             # and no path exists through it.
             return float('inf')


        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            # current_loc is guaranteed to be in self.location_graph here because
            # we checked the start node and only add neighbors from the graph.

            for neighbor_loc in self.location_graph[current_loc]:
                if neighbor_loc not in visited:
                    if neighbor_loc == end:
                        return dist + 1
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

        return float('inf') # No path found


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Find current locations of robot and boxes
        robot_loc = None
        box_locs = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at-robot" and len(args) == 1:
                robot_loc = args[0]
            elif predicate == "at" and len(args) == 2:
                box, loc = args
                box_locs[box] = loc

        # Basic validation: robot and all goal boxes must be in the state
        # This check might be overly strict depending on how states are generated,
        # but for valid states from the initial state, this should hold.
        if robot_loc is None or not all(box in self.goal_locations for box in box_locs):
             # This indicates an invalid state representation, return infinity
             return float('inf')

        total_heuristic = 0  # Initialize action cost counter.

        # Calculate heuristic based on box distances to goals
        for box, goal_location in self.goal_locations.items():
            current_box_location = box_locs[box]

            if current_box_location != goal_location:
                # Calculate shortest path distance for the box
                # This is the minimum number of pushes needed if the box could move freely
                box_dist = self.bfs_locations(current_box_location, goal_location)

                if box_dist == float('inf'):
                    # If a box cannot reach its goal, the state is unsolvable
                    return float('inf')

                total_heuristic += box_dist

        # The heuristic is the sum of box distances. Robot distance is ignored
        # in this simple version for efficiency and non-admissibility.

        return total_heuristic
