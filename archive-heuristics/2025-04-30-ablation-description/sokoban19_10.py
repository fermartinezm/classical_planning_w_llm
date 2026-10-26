from fnmatch import fnmatch
# Assuming the Heuristic base class is provided in the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class definition if not provided externally
# This is just for ensuring the code structure is correct if run outside the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
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
    This heuristic estimates the minimum number of box push actions required
    to move each box from its current location to its goal location. It is
    calculated as the sum of the shortest path distances for each box to its
    respective goal, where distance is measured by the number of adjacent
    locations traversed. This heuristic is non-admissible as it ignores
    robot movement costs, the need for clear paths, and potential interactions
    between boxes.

    # Assumptions
    - The grid structure is implicitly defined by the 'adjacent' predicates.
    - Each box specified in the goals has a unique target location.
    - The shortest path between locations can be computed using BFS on the
      undirected adjacency graph derived from 'adjacent' predicates.

    # Heuristic Initialization
    - Build an undirected graph representing the locations and their adjacencies
      from the static 'adjacent' facts.
    - Extract the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box that has a specified goal location in the task:
    2. Find the box's current location in the current state facts.
    3. Retrieve the box's goal location stored during initialization.
    4. If the box is already at its goal location, the cost for this box is 0.
    5. If the box is not at its goal, compute the shortest path distance between
       the box's current location and its goal location using a Breadth-First Search (BFS)
       on the adjacency graph built from 'adjacent' facts. This distance represents
       the minimum number of 'push' actions needed for this box if there were no
       robot positioning or blocking issues.
    6. Sum the calculated distances for all boxes that are not yet at their goals.
    7. If any box's goal location is unreachable from its current location via the
       adjacency graph, the state is considered a dead end or part of an unsolvable
       path, and the heuristic returns infinity.
    8. The total sum (or infinity) is the heuristic value for the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the adjacency graph and storing
        goal locations for boxes.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency graph from static facts
        self.adjacency_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check if the fact is an 'adjacent' predicate with 3 arguments
            if len(parts) == 4 and parts[0] == "adjacent":
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Treat graph as undirected for distance calculation
                if loc1 not in self.adjacency_graph:
                    self.adjacency_graph[loc1] = set()
                if loc2 not in self.adjacency_graph:
                     self.adjacency_graph[loc2] = set()
                self.adjacency_graph[loc1].add(loc2)
                self.adjacency_graph[loc2].add(loc1) # Add reverse edge

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at box_name location_name)
            parts = get_parts(goal)
            # Check if the goal fact is an 'at' predicate with 2 arguments
            if len(parts) == 3 and parts[0] == "at":
                 obj, location = parts[1], parts[2]
                 # Assuming only boxes are specified in goals using 'at'
                 # A more robust check might involve checking object types if available
                 self.goal_locations[obj] = location


    def _bfs_distance(self, start_loc, goal_loc):
        """
        Computes the shortest path distance between two locations using BFS
        on the adjacency graph. Returns float('inf') if goal is unreachable.
        """
        # If start or goal location is not in the graph, they are isolated or invalid
        # unless start == goal and that location isn't adjacent to anything.
        if start_loc == goal_loc:
            return 0

        if start_loc not in self.adjacency_graph or goal_loc not in self.adjacency_graph:
             return float('inf')


        queue = [(start_loc, 0)] # (location, distance)
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.pop(0) # Use pop(0) for BFS queue behavior

            if current_loc == goal_loc:
                return dist

            # current_loc is guaranteed to be in self.adjacency_graph here
            for neighbor in self.adjacency_graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # Goal is unreachable from start_loc


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # Find current locations of all objects (including boxes)
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Find 'at' facts for objects that are in our goal list (assuming they are boxes)
            if len(parts) == 3 and parts[0] == "at" and parts[1] in self.goal_locations:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            # We don't strictly need the robot location for this specific heuristic calculation
            # but it's good practice to know how to extract it if needed.
            # elif len(parts) == 2 and parts[0] == "at-robot":
            #      loc = parts[1]
            #      current_locations["robot"] = loc


        total_heuristic = 0

        # Calculate heuristic for each box that has a goal location
        for box, goal_loc in self.goal_locations.items():
            # Ensure the box's current location is known in the state
            if box in current_locations:
                current_box_loc = current_locations[box]

                # If the box is already at its goal, cost is 0 for this box
                if current_box_loc == goal_loc:
                    continue

                # Calculate distance from current box location to its goal location
                distance = self._bfs_distance(current_box_loc, goal_loc)

                # If the goal is unreachable for a box, the state is likely a dead end
                # or part of an unsolvable path. A high heuristic value is appropriate.
                if distance == float('inf'):
                    return float('inf') # Return infinity immediately if any box is stuck

                total_heuristic += distance
            else:
                 # If a box from the goal list is not found in the current state's 'at' facts,
                 # it implies an issue or an unreachable state.
                 # Returning infinity indicates this state is likely not on a path to the goal.
                 return float('inf')


        # The heuristic is the sum of distances. It is 0 iff all distances are 0,
        # which means all boxes are at their goal locations.
        return total_heuristic
