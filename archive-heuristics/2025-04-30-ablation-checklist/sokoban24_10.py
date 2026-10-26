import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing
    two components for each box that is not yet at its goal location:
    1. The minimum number of pushes required to move the box from its
       current location to its goal location, ignoring dynamic obstacles
       but respecting the grid structure and requiring the target square
       to be clear.
    2. The minimum number of robot moves required to get from the robot's
       current location to any location adjacent to the box, respecting
       dynamic obstacles (other boxes, walls) and requiring traversed
       squares to be clear.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Locations are named `loc_R_C`.
    - The goal is a set of `(at ?box ?location)` facts.

    # Heuristic Initialization
    - Build an undirected graph representing the grid connectivity from
      `adjacent` static facts.
    - Store the goal location for each box.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each box that is not at its goal:
    2. Calculate the minimum number of pushes (`box_dist`) needed for the box
       to reach its goal location. This is the shortest path distance on the
       grid graph where traversal is only allowed into locations that are
       currently clear (or the goal location itself).
    3. Calculate the minimum number of robot moves (`robot_dist`) needed for
       the robot to reach *any* location adjacent to the box. This is the
       shortest path distance on the grid graph where traversal is only
       allowed into locations that are currently clear (or the target
       neighbor location itself).
    4. The heuristic for a single box is `box_dist + robot_dist`.
    5. The total heuristic is the sum of these costs for all boxes not at their goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - The grid graph from static `adjacent` facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph (adjacency list) from adjacent facts.
        # The graph is undirected for distance calculations.
        self.graph = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Add reverse edge for undirected graph

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                obj, location = args
                # Assuming only boxes are objects with 'at' goals
                if obj.startswith('box'):
                    self.goal_locations[obj] = location

    def _bfs(self, start, goals, graph, state):
        """
        Performs a Breadth-First Search to find the shortest distance
        from a start location to any of the goal locations.

        Traversal is restricted to locations that are currently clear
        in the state, or are one of the goal locations.

        Args:
            start (str): The starting location.
            goals (set): A set of target locations.
            graph (dict): The adjacency list representation of the grid.
            state (frozenset): The current state facts.

        Returns:
            int: The shortest distance, or float('inf') if no goal is reachable.
        """
        if not goals:
            return float('inf') # No goals to reach

        queue = collections.deque([(start, 0)])
        visited = {start}

        is_clear = lambda loc: f'(clear {loc})' in state

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc in goals:
                return dist

            # Explore neighbors
            for neighbor in graph.get(current_loc, []):
                if neighbor not in visited:
                    # A location is traversable if it's clear OR it's one of the goals
                    if is_clear(neighbor) or neighbor in goals:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # Goals not reachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot and box locations
        robot_location = None
        current_box_locations = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_location = parts[1]
            elif parts[0] == "at" and parts[1].startswith('box'):
                current_box_locations[parts[1]] = parts[2]
            elif parts[0] == "clear":
                clear_locations.add(parts[1])

        total_cost = 0

        # Calculate cost for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If box is not in the state or already at goal, skip
            if current_loc is None or current_loc == goal_loc:
                continue

            # --- Component 1: Box path cost (minimum pushes) ---
            # BFS for the box from its current location to its goal.
            # The box can only be pushed into a clear square or its goal square.
            # We use the _bfs helper, which checks for (clear loc) or loc in goals.
            box_dist = self._bfs(current_loc, {goal_loc}, self.graph, state)

            # --- Component 2: Robot approach cost ---
            # BFS for the robot from its current location to any neighbor of the box.
            # The robot can only move into clear squares or the target neighbor square.
            # We need the neighbors of the box's current location.
            box_neighbors = set(self.graph.get(current_loc, []))

            # The robot cannot move onto the square the box currently occupies.
            # The robot BFS should only traverse through locations that are clear
            # or are one of the target neighbors.
            robot_dist = self._bfs(robot_location, box_neighbors, self.graph, state)

            # Add costs, handling unreachable cases
            if box_dist == float('inf') or robot_dist == float('inf'):
                 # If a box cannot reach its goal or the robot cannot reach the box,
                 # this state might be a dead end or require complex unblocking.
                 # Returning infinity guides the search away from such states.
                 return float('inf')

            total_cost += box_dist + robot_dist

        return total_cost

