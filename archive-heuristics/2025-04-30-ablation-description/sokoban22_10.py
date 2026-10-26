from fnmatch import fnmatch
from collections import deque
# Assuming the Heuristic base class is available in the environment as described.
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at ball1 rooma)" -> ["at", "ball1", "rooma"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have the same number of components as the pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
# Inherit from Heuristic base class if available, otherwise define a dummy
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base class is not found
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found.")


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    for each box from its current location to its goal location. The distance
    is calculated using Breadth-First Search (BFS) on the graph defined by
    the 'adjacent' predicates. This heuristic provides a lower bound on the
    number of pushes required to move the boxes to their goals, ignoring
    robot movement and potential blockages by other boxes or static obstacles
    not captured by the adjacency graph.

    # Assumptions
    - The 'adjacent' predicates define an undirected graph representing
      walkable connections between locations. While the PDDL includes
      directions (up, down, left, right), the connectivity itself is assumed
      to be bidirectional for the purpose of building the graph.
    - The heuristic ignores the robot's position and the cost of robot
      movement to reach a position from which it can push a box.
    - The heuristic ignores potential blockages caused by other boxes or
      static obstacles (like walls) that are not represented by missing
      'adjacent' predicates in the graph.
    - The heuristic is admissible with respect to the number of box pushes
      (each push moves a box one step closer in terms of graph distance),
      but not necessarily with respect to the total number of actions
      (moves + pushes).

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
      This is stored in `self.goal_locations` mapping box name to its target location.
    - Builds an undirected adjacency list graph representing the connections
      between locations based on the 'adjacent' static facts. This graph is
      stored in `self.adjacency_list`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Access the current state from the input `node`.
    2. Identify the current location of each box by iterating through the facts
       in the current state and looking for `(at ?box ?location)` predicates
       where `?box` is one of the boxes specified in the goals.
    3. Initialize a `total_heuristic` value to 0.
    4. Iterate through each box and its corresponding goal location stored
       during initialization (`self.goal_locations`).
    5. For the current box, retrieve its current location found in step 2.
    6. If the box's current location is different from its goal location:
       a. Calculate the shortest path distance between the box's current
          location and its goal location using the `bfs_distance` helper method
          on the pre-built location graph (`self.adjacency_list`).
       b. If `bfs_distance` returns `float('inf')`, it means the goal location
          is unreachable from the box's current location within the defined graph.
          In this case, the state is likely unsolvable, so return `float('inf')`
          immediately as the heuristic value.
       c. Add the calculated distance to the `total_heuristic`.
    7. After processing all boxes, return the final `total_heuristic` value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph from static facts.
        """
        # Store task goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each box
        # self.goal_locations will be a dictionary: {box_name: goal_location_name}
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically '(at box_name loc_name)'
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location

        # Build undirected adjacency graph from static facts
        # self.adjacency_list will be a dictionary: {location_name: [adjacent_location_names]}
        self.adjacency_list = {}
        for fact in self.static:
            # Adjacent facts are typically '(adjacent loc1 loc2 direction)'
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                # Ensure locations exist as keys in the graph dictionary
                if loc1 not in self.adjacency_list:
                    self.adjacency_list[loc1] = []
                if loc2 not in self.adjacency_list:
                    self.adjacency_list[loc2] = []
                # Add undirected edge: add loc2 to loc1's list and loc1 to loc2's list
                # assuming adjacency is symmetric based on typical Sokoban grids
                self.adjacency_list[loc1].append(loc2)
                self.adjacency_list[loc2].append(loc1)

        # Remove potential duplicates in adjacency lists that might arise from
        # redundant adjacent facts or adding both directions explicitly
        for loc in self.adjacency_list:
            self.adjacency_list[loc] = list(set(self.adjacency_list[loc]))


    def bfs_distance(self, start_loc, end_loc):
        """
        Calculates the shortest path distance between two locations using BFS
        on the pre-built adjacency graph (`self.adjacency_list`).
        Returns float('inf') if no path exists between the locations.
        """
        # If start and end are the same, distance is 0
        if start_loc == end_loc:
            return 0

        # If either location is not in the graph, they are unreachable from each other
        if start_loc not in self.adjacency_list or end_loc not in self.adjacency_list:
             return float('inf')

        # Initialize BFS queue with the start location and distance 0
        queue = deque([(start_loc, 0)])
        # Keep track of visited locations to avoid cycles and redundant processing
        visited = {start_loc}

        # Perform BFS
        while queue:
            # Get the current location and distance from the front of the queue
            current_loc, dist = queue.popleft()

            # If the current location is the target, return the distance
            if current_loc == end_loc:
                return dist

            # Explore neighbors of the current location
            # Ensure current_loc is still a valid key in case graph was modified (shouldn't happen here)
            if current_loc in self.adjacency_list:
                for neighbor in self.adjacency_list[current_loc]:
                    # If the neighbor hasn't been visited yet
                    if neighbor not in visited:
                        # Mark it as visited
                        visited.add(neighbor)
                        # Add the neighbor to the queue with incremented distance
                        queue.append((neighbor, dist + 1))

        # If the loop finishes and the end_loc was not reached, it's unreachable
        return float('inf')


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        The value is the sum of the shortest path distances for each box
        from its current location to its goal location.
        """
        state = node.state # The current state is a frozenset of facts (strings)

        # Find current box locations by scanning the state facts
        current_box_locations = {}
        # We also need the robot location for some heuristics, but this one doesn't use it
        # robot_location = None

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, loc = args
                # Check if the object is one of the boxes we need to move to a goal
                if obj in self.goal_locations:
                     current_box_locations[obj] = loc
            # elif predicate == "at-robot" and len(args) == 1:
            #     robot_location = args[0]

        total_heuristic = 0

        # Iterate through each box and its goal location defined in the task
        for box, goal_loc in self.goal_locations.items():
            # Get the box's current location from the state
            current_loc = current_box_locations.get(box)

            # If the box's current location is known and it's not already at the goal
            if current_loc and current_loc != goal_loc:
                # Calculate the shortest distance for this box to its goal
                dist = self.bfs_distance(current_loc, goal_loc)

                # If the goal is unreachable for this box, the state is unsolvable
                if dist == float('inf'):
                    return float('inf')

                # Add the distance for this box to the total heuristic
                total_heuristic += dist

        # The total heuristic is the sum of the minimum pushes needed for each box
        # independently to reach its goal.
        return total_heuristic
