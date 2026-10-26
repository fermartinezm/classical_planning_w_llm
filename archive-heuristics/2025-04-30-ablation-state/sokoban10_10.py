from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, although valid PDDL facts should conform
        return []
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box not at its goal,
    the shortest path distance from the box's current location to its goal location (representing minimum pushes needed)
    and the shortest path distance from the robot's current location to the box's current location (representing robot effort to reach the box).

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic assumes reachability on the grid graph; states where a box or the robot cannot reach a required location are assigned a large heuristic value.
    - This heuristic is non-admissible.

    # Heuristic Initialization
    - The grid graph is built from the `adjacent` facts in the static information. The graph nodes are locations, and edges connect locations that are adjacent according to the PDDL facts. The graph is treated as undirected for distance calculation.
    - The goal locations for each box are extracted from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Build the grid graph from `adjacent` predicates during initialization. The graph nodes are locations, and edges connect locations that are adjacent according to the PDDL facts. The graph is treated as undirected for distance calculation.
    2. Store the goal location for each box that appears in the goal conditions.
    3. In the heuristic computation (`__call__`):
        a. Identify the robot's current location from the state facts.
        b. Identify the current location of each box that has a goal location, from the state facts.
        c. Initialize the total heuristic cost to 0.
        d. For each box that is not currently at its goal location:
            i. Calculate the shortest path distance from the box's current location to its goal location using Breadth-First Search (BFS) on the grid graph. This distance represents a lower bound on the number of push actions required for this box.
            ii. Calculate the shortest path distance from the robot's current location to the box's current location using BFS on the grid graph. This estimates the robot's movement cost to get near the box to potentially push it.
            iii. Add these two distances (box-to-goal distance and robot-to-box distance) to the total heuristic cost.
            iv. If any required BFS calculation returns a large value (representing infinity, meaning no path exists), the state is likely unsolvable or a deadlock; return this large value immediately.
        e. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and storing goal locations.
        """
        self.goals = task.goals
        self.static = task.static

        # Build the grid graph from adjacent facts
        self.graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            # Check if the fact is an adjacent predicate with correct number of parts
            if len(parts) == 4 and parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add loc1 -> loc2 edge
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                self.graph[loc1].add(loc2)
                # Add loc2 -> loc1 edge (assuming adjacency is symmetric for distance calculation)
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc2].add(loc1)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal is an 'at' predicate for a box
            # We assume objects with goals are boxes in this domain
            if len(parts) == 3 and parts[0] == 'at':
                 obj, location = parts[1], parts[2]
                 self.goal_locations[obj] = location

        # Large value to represent infinity for unreachable locations
        self.infinity = 1_000_000 # Use a large integer

    def bfs(self, graph, start, end):
        """
        Performs Breadth-First Search to find the shortest path distance
        between start and end locations in the grid graph.
        Returns the distance or self.infinity if no path exists.
        """
        if start == end:
            return 0
        # Ensure start and end locations exist in the graph
        if start not in graph or end not in graph:
             # This might happen if the PDDL defines locations not connected
             # by any adjacent facts, or if start/end are not valid locations.
             # Treat as unreachable.
             return self.infinity

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Check if current_loc has neighbors in the graph dictionary
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # If the loop finishes without finding the end location
        return self.infinity

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state  # The state is a frozenset of fact strings

        # Find robot location and current box locations
        robot_location = None
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'at-robot':
                robot_location = parts[1]
            elif len(parts) == 3 and parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 # Only track locations for objects that are boxes and have goals
                 if obj in self.goal_locations:
                    current_box_locations[obj] = loc

        # If robot location is not found, something is wrong with the state representation
        # or it's an unreachable state.
        if robot_location is None:
             return self.infinity

        total_cost = 0

        # Calculate cost for each box not at its goal
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If box is not in the state (shouldn't happen in Sokoban) or already at goal, skip
            if current_location is None or current_location == goal_location:
                continue

            # Calculate box distance to goal
            box_dist = self.bfs(self.graph, current_location, goal_location)

            # Calculate robot distance to box
            robot_dist = self.bfs(self.graph, robot_location, current_location)

            # If either distance is infinity, this box or goal might be unreachable from current positions
            if box_dist == self.infinity or robot_dist == self.infinity:
                 return self.infinity # Indicate unsolvable/deadlock state

            # Add costs for this box
            # The box_dist is the minimum number of grid steps the box needs to move (minimum pushes).
            # The robot_dist is the minimum number of grid steps the robot needs to move to reach the box's location.
            # This sum is a simple non-admissible estimate of the total effort.
            total_cost += box_dist + robot_dist

        return total_cost
