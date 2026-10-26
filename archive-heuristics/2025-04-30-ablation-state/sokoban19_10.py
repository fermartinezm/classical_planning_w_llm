from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args if no wildcards are used
    # or if wildcards are used appropriately. Simple zip check is sufficient for this domain.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its goal location.
    The distance is calculated on the grid graph defined by the 'adjacent'
    predicates, ignoring obstacles (other boxes, robot) and robot movement costs.
    This is a relaxed problem where boxes can move freely to their goals.

    # Assumptions
    - The grid structure is defined by 'adjacent' predicates, forming a connected graph.
    - Each box specified in the goal has a unique goal location.
    - The shortest path distance on the grid represents the minimum number of
      'push' actions required for a box in an unblocked grid.
    - Robot movement cost and the cost of clearing paths/target locations are ignored.
    - Deadlocks are not explicitly detected or penalized.

    # Heuristic Initialization
    - Parse 'adjacent' facts to build an undirected graph representing the
      connectivity of locations.
    - Extract the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box that needs to be moved to a goal location (as specified in the task goals):
    2. Find the box's current location in the current state.
    3. Retrieve the box's goal location (pre-calculated during initialization).
    4. If the box is not currently at its goal location:
       a. Calculate the shortest path distance between the box's current location
          and its goal location using Breadth-First Search (BFS) on the location graph.
          This distance is the minimum number of steps the box would need to take
          if it could move freely.
       b. Add this calculated distance to a running total heuristic value.
    5. The final heuristic value for the state is the sum of these distances for all
       boxes that are not yet in their goal positions.
    6. If all boxes are at their goal locations, the sum will be zero, correctly
       indicating a goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the location graph and goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.adj_list = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                self.locations.add(loc1)
                self.locations.add(loc2)
                # Add undirected edges
                self.adj_list.setdefault(loc1, []).append(loc2)
                self.adj_list.setdefault(loc2, []).append(loc1)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's an (at ?box ?location) goal
                box, location = args
                # Assuming goal specifies location for objects of type 'box'
                # We can refine this by checking object types if needed, but PDDL structure implies this.
                self.goal_locations[box] = location

    def bfs_distance(self, start_loc, end_loc):
        """
        Calculate the shortest path distance between two locations using BFS on the grid graph.
        Returns float('inf') if end_loc is unreachable from start_loc.
        """
        if start_loc == end_loc:
            return 0

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Get neighbors from the adjacency list, default to empty list if location not found
            for neighbor in self.adj_list.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If BFS completes without finding the end_loc, it's unreachable
        return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of facts

        # Find current location of each box that has a goal
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate and the object is one of the goal boxes
            if parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_locations:
                 box, location = parts[1:]
                 current_box_locations[box] = location

        total_heuristic = 0

        # Sum distances for boxes not at their goals
        for box, goal_location in self.goal_locations.items():
            # Get the box's current location from the state.
            # Use .get() with None default in case a box is not 'at' any location in the state
            # (e.g., carried by robot in some domains, though not in this Sokoban).
            # For this Sokoban, boxes are always 'at' a location.
            current_location = current_box_locations.get(box)

            # If the box exists in the state and is not at its goal location
            if current_location and current_location != goal_location:
                 # Calculate the shortest distance on the grid
                 dist = self.bfs_distance(current_location, goal_location)

                 # Add the distance to the total heuristic.
                 # If dist is inf (unreachable), the total becomes inf.
                 total_heuristic += dist

        return total_heuristic
