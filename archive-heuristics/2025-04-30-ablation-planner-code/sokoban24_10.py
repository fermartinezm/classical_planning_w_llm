from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    return fact.strip()[1:-1].split()

# BFS function to find shortest path distance on the location graph
def bfs_distance(adj_list, start_loc, end_loc):
    """
    Finds the shortest path distance between start_loc and end_loc
    on the graph defined by adj_list.
    Returns distance or float('inf') if unreachable.
    """
    if start_loc == end_loc:
        return 0

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, current_dist = queue.popleft()

        # Neighbors are locations adjacent to current_loc
        # Use .get() with default empty list for locations with no adjacencies
        for neighbor_loc in adj_list.get(current_loc, []):
            if neighbor_loc not in visited:
                visited.add(neighbor_loc)
                if neighbor_loc == end_loc:
                    return current_dist + 1
                queue.append((neighbor_loc, current_dist + 1))

    return float('inf') # Target unreachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    from each misplaced box to its goal location. The distance is calculated
    on the static location graph, ignoring the robot's position and other
    dynamic obstacles. This estimates the minimum number of 'pushes' required
    if the path were always clear and the robot were always in the correct
    position to push.

    # Assumptions
    - The goal state specifies a unique goal location for each box that needs
      to be moved.
    - The location graph defined by 'adjacent' predicates is static and
      effectively symmetric for reachability purposes (if A is adjacent to B,
      B is adjacent to A).
    - The heuristic ignores the robot's position and the presence of other
      boxes or the robot as dynamic obstacles when calculating distances.
      It only considers the static layout defined by 'adjacent' predicates.

    # Heuristic Initialization
    - Builds an adjacency list representation of the location graph from
      the 'adjacent' static facts.
    - Extracts the goal location for each box from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of each box in the given state by parsing
       '(at box location)' facts.
    2. Compare the current location of each box (that has a specified goal)
       to its goal location (determined during initialization).
    3. For each box that is not at its goal location (a "misplaced" box):
       a. Calculate the shortest path distance from the box's current location
          to its goal location using Breadth-First Search (BFS) on the static
          location graph (built from 'adjacent' facts). This BFS considers
          only the connections defined by 'adjacent' predicates and ignores
          dynamic elements like the robot or other boxes.
       b. If the goal location is unreachable from the box's current location
          on the static graph, the state is likely unsolvable or requires
          complex maneuvers not captured by this simple distance. The heuristic
          returns infinity in this case.
       c. Add the calculated distance to a running total.
    4. The total heuristic value is the sum of these distances for all
       misplaced boxes. If no boxes specified in the goal are misplaced,
       the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the adjacency list for the location graph.
        self.adj_list = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check for the correct predicate name and number of arguments
            if len(parts) == 4 and parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1:]
                # Add bidirectional edges. BFS doesn't care about direction.
                self.adj_list.setdefault(loc1, []).append(loc2)
                self.adj_list.setdefault(loc2, []).append(loc1) # Assuming symmetric adjacency

        # Store goal locations for each box.
        self.goal_locations = {}
        # task.goals can be a single fact string or a frozenset of fact strings
        if isinstance(self.goals, str):
             # Single goal fact
             predicate, *args = get_parts(self.goals)
             if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location
        elif isinstance(self.goals, frozenset):
             # Conjunction of goals
             for sub_goal in self.goals:
                 # Ensure sub_goal is a string fact
                 if isinstance(sub_goal, str):
                     predicate, *args = get_parts(sub_goal)
                     if predicate == "at" and len(args) == 2:
                        box, location = args
                        self.goal_locations[box] = location
        # Note: This only extracts 'at' goals for boxes. Other goal types are ignored.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Find current locations of all boxes.
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check for the correct predicate name and number of arguments
            if len(parts) == 3 and parts[0] == 'at': # (at ?o - box ?l - location)
                box, location = parts[1:]
                box_locations[box] = location

        total_distance = 0
        all_goals_reached = True

        # Calculate distance for each box that has a specified goal location.
        # We iterate through goal_locations to ensure we only consider boxes
        # that are part of the goal specification.
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # If a box required by the goal is not found in the current state's
            # 'at' predicates, it's an unexpected state. Treat as unreachable.
            if current_location is None:
                 return float('inf')

            if current_location != goal_location:
                all_goals_reached = False
                # Calculate shortest path distance on the static graph
                # Obstacles are ignored in this simple heuristic
                dist = bfs_distance(self.adj_list, current_location, goal_location)

                if dist == float('inf'):
                    # If any box cannot reach its goal on the static graph,
                    # the state is likely unsolvable or requires complex sequences
                    # not captured by this simple distance. Return infinity.
                    return float('inf')

                total_distance += dist

        # If all boxes specified in the goal are at their goal locations,
        # the heuristic is 0. Otherwise, it's the sum of distances.
        if all_goals_reached:
             return 0
        else:
             return total_distance
