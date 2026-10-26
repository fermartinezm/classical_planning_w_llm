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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the shortest path distances for each box from its current
    location to its goal location. It uses Breadth-First Search (BFS) on the
    static grid graph defined by 'adjacent' predicates to find these distances.

    # Assumptions
    - Each box has a unique goal location.
    - The grid structure is defined by 'adjacent' predicates.
    - The heuristic ignores the robot's position and the dynamic obstacles
      (other boxes, walls not part of the static grid) when calculating
      box-goal distances. This makes it non-admissible but efficient.

    # Heuristic Initialization
    - Builds a graph representation of the grid from 'adjacent' facts.
    - Extracts the goal location for each box from the task goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1.  **Build the Grid Graph:** Parse the `adjacent` facts from the static
        information to create an adjacency list representation of the grid.
        Since movement is bidirectional, add edges for both directions
        for each `adjacent` fact. This is done once during initialization.
    2.  **Identify Box Goals:** Parse the goal conditions to find the target
        location for each box. This is done once during initialization.
    3.  **Find Current Box Locations:** In the `__call__` method, parse the
        current state to determine the location of each box.
    4.  **Calculate Box-Goal Distances:** For each box that is not yet at its
        goal location, use BFS on the pre-built grid graph to find the shortest
        path distance from the box's current location to its goal location.
    5.  **Sum Distances:** The heuristic value is the sum of these shortest
        path distances for all boxes not at their goals.

    This heuristic is non-admissible because it doesn't consider the robot's
    effort to get into position to push a box, nor does it account for
    dynamic obstacles (other boxes) that might block the shortest path.
    However, it provides a direct measure of how far the boxes are from
    their targets, which is a primary driver of cost in Sokoban.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and extracting
        box goal locations.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph from adjacent facts
        self.grid_graph = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                # Add bidirectional edges as movement is possible in reverse direction
                self.grid_graph[loc1].add(loc2)
                self.grid_graph[loc2].add(loc1)

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location

    def bfs(self, start_node, end_node):
        """
        Performs Breadth-First Search to find the shortest path distance
        between two locations on the static grid graph.

        Args:
            start_node (str): The starting location name.
            end_node (str): The target location name.

        Returns:
            int: The shortest distance (number of moves) or float('inf')
                 if the end_node is unreachable from the start_node.
        """
        if start_node == end_node:
            return 0

        # Handle cases where nodes might not be in the graph (shouldn't happen in valid PDDL)
        if start_node not in self.grid_graph or end_node not in self.grid_graph:
             # print(f"Warning: Node {start_node} or {end_node} not in graph.")
             return float('inf') # Or a large number

        queue = collections.deque([(start_node, 0)])
        visited = {start_node}

        while queue:
            current_loc, distance = queue.popleft()

            if current_loc == end_node:
                return distance

            for neighbor in self.grid_graph.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        # If the queue is empty and the end_node was not reached
        return float('inf') # Or a large number

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            int: The estimated cost to reach the goal state.
        """
        state = node.state

        # Find current locations of all boxes
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in self.box_goals: # Check if it's a box we care about
                 box, location = parts[1:]
                 current_box_locations[box] = location

        total_distance = 0

        # Sum the shortest distances for each box to its goal
        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box) # Use .get() in case a box isn't found (shouldn't happen)

            if current_location is None:
                 # This indicates an issue with state parsing or task definition
                 # print(f"Warning: Location for box {box} not found in state.")
                 continue # Skip this box or handle as error

            if current_location != goal_location:
                # Calculate shortest path distance on the static grid
                box_to_goal_dist = self.bfs(current_location, goal_location)

                # If a box goal is unreachable, return infinity (or a large cost)
                if box_to_goal_dist == float('inf'):
                    return float('inf') # This state is likely a dead end or unreachable goal

                total_distance += box_to_goal_dist

        return total_distance

