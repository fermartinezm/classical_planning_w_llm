from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_4_4)" -> ['at', 'box1', 'loc_4_4']
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_4_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distance(graph, start, end):
    """
    Calculates the shortest path distance between two locations in the graph.
    Returns float('inf') if no path exists.
    """
    if start == end:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    # Ensure start node is in the graph keys, otherwise it's isolated
    if start not in graph:
        return float('inf')

    while queue:
        current_loc, current_dist = queue.popleft()

        # Ensure current_loc is in graph keys before accessing neighbors
        if current_loc not in graph:
             continue

        for neighbor in graph[current_loc]:
            if neighbor == end:
                return current_dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_dist + 1))

    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its goal location.
    The distance is calculated on the graph of locations defined by the
    'adjacent' predicates.

    # Assumptions
    - The goal is defined by the final locations of specific boxes.
    - The grid structure and connectivity are defined solely by the 'adjacent'
      predicates in the static facts.
    - The heuristic ignores the robot's position and the 'clear' predicate,
      treating the locations as freely traversable for the purpose of distance
      calculation. This is a relaxation and makes the heuristic non-admissible
      but can be useful for guiding a greedy search.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds a graph representation of the locations and their adjacencies
      based on the 'adjacent' predicates in the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box that needs to be moved. This information is stored during initialization.
    2. For the current state:
       - Identify the current location of each box that has a goal.
    3. Initialize the total heuristic cost to 0.
    4. For each box that has a goal location:
       - Get the box's current location and its goal location.
       - If the box is not already at its goal location:
         - Calculate the shortest path distance between the box's current
           location and its goal location using Breadth-First Search (BFS)
           on the location graph built during initialization. The graph represents
           all possible movements between adjacent locations, ignoring obstacles
           and the robot.
         - Add this calculated distance to the total heuristic cost.
         - If BFS indicates no path exists (returns infinity), the state is
           likely unsolvable or represents a significant problem, so add
           infinity to the total cost.
    5. Return the total heuristic cost. If the state is a goal state, the cost
       will be 0 as all boxes will be at their goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Build the location graph from adjacent facts.
        # Graph is represented as an adjacency list: {location: [adjacent_locations]}
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Adjacent facts are like (adjacent loc_1_1 loc_1_2 right)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1:]
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add bidirectional edges as movement is possible in both directions
                # even if the predicate only lists one direction.
                # e.g., (adjacent A B right) implies robot can move A->B and B->A,
                # and box can be pushed A->B (robot at loc before A) or B->A (robot at loc before B).
                # The graph for distance calculation should represent all traversable links.
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Store goal locations for each box.
        # Goals are typically (at ?b ?l) for specific boxes.
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal fact is of the form (at ?box ?location)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1:]
                # Assume any 'at' predicate in the goal refers to a box
                # needing to be at a specific location.
                self.box_goals[box] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the sum of box-to-goal distances.
        """
        state = node.state  # Current world state.

        # If the current state is the goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Find the current location of each box that has a goal.
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Look for facts like (at box1 loc_4_4)
            if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1:]
                 # Only consider objects that are boxes with defined goals
                 if obj in self.box_goals:
                    current_box_locations[obj] = loc

        total_cost = 0.0  # Use float to handle potential infinity

        # Calculate heuristic for each misplaced box
        for box, goal_location in self.box_goals.items():
            # Ensure the box exists in the current state's 'at' facts.
            # If not, something is wrong with the state representation or problem.
            # We'll assume it exists and its location is in current_box_locations.
            current_location = current_box_locations.get(box)

            if current_location is None:
                 # This shouldn't happen in a valid state, but as a fallback
                 # return infinity if a box is missing.
                 return float('inf')

            if current_location != goal_location:
                # Calculate shortest path distance for the box on the location graph.
                # This is a relaxed distance ignoring robot, other boxes, and clear status.
                distance = bfs_distance(self.graph, current_location, goal_location)

                # If any box's goal is unreachable, the state is likely unsolvable
                # or very poor, so the heuristic should reflect this.
                if distance == float('inf'):
                    return float('inf')

                total_cost += distance

        # The heuristic is the sum of distances. If all boxes are at their goals,
        # the loop calculates 0, which is handled by the initial check.
        # If some boxes are misplaced, the cost is > 0.
        # If any box is unreachable, the cost is inf.

        return total_cost
