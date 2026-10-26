from fnmatch import fnmatch
import collections

# Assuming Heuristic base is available
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
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
    # Check if the number of parts matches the number of args, considering trailing wildcards
    if len(parts) < len(args):
        return False
    # Check if all specified parts match the pattern args
    if not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
        return False
    # If args has fewer elements than parts, the remaining parts must be covered by a wildcard if args is not empty
    if len(parts) > len(args) and (not args or args[-1] != '*'):
         return False
    return True


# Define the heuristic class, potentially inheriting from a base class
# class sokobanHeuristic(Heuristic):
class sokobanHeuristic: # Use this if Heuristic base is not provided as a separate file
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box not yet at its goal:
    1. The shortest path distance the box needs to travel to reach its goal location (minimum pushes).
    2. The shortest path distance the robot needs to travel to reach the box's current location (minimum moves to get near the box).
    The shortest path distances are computed on the graph defined by the 'adjacent' predicates.

    # Assumptions
    - The grid structure and connectivity are defined by the 'adjacent' predicates.
    - Shortest path distances on this grid are a reasonable estimate for movement costs for both the robot and the box (as a proxy for pushes).
    - The cost of moving the robot to the box and pushing the box are roughly additive.
    - Dead ends (like pushing a box into a corner) are not explicitly detected or penalized beyond making the goal location unreachable in the graph.

    # Heuristic Initialization
    - Build a graph representing the locations and their adjacencies based on 'adjacent' facts. The graph contains a directed edge from loc1 to loc2 if '(adjacent loc1 loc2 dir)' exists.
    - Compute all-pairs shortest path distances on this graph using BFS starting from each location.
    - Extract the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the current location of the robot by finding the fact '(at-robot ?l)' in the state.
    3. Identify the current location of each box that is specified in the goal conditions by finding facts like '(at ?b ?l)' in the state.
    4. Initialize the total heuristic cost to 0.
    5. For each box and its corresponding goal location stored during initialization:
       - Get the box's current location from the state.
       - If the box's current location is different from its goal location:
         - Calculate the shortest path distance from the box's current location to its goal location using the precomputed distances. This distance is added to the total cost, representing the estimated number of pushes required for this box. If the goal is unreachable from the box's current location, the distance will be infinity.
         - Calculate the shortest path distance from the robot's current location to the box's current location using the precomputed distances. This distance is added to the total cost, representing the estimated number of robot moves required to reach the box. If the box is unreachable by the robot, the distance will be infinity.
         - If either of the distances calculated in the previous steps is infinity, it suggests the problem might be unsolvable from this state (e.g., box in a dead end, robot trapped). In this case, the heuristic should return infinity.
    6. Return the calculated total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing
        shortest paths, and extracting box goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Collect all unique locations mentioned in adjacent facts
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                all_locations.add(parts[1])
                all_locations.add(parts[2])

        # Build the graph from adjacent facts
        # Graph is represented as an adjacency list: {location: {neighbor1, neighbor2, ...}}
        self.graph = {loc: set() for loc in all_locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add directed edge loc1 -> loc2
                self.graph[loc1].add(loc2)

        # Compute all-pairs shortest paths using BFS from each location
        # distances[start_node][end_node] = shortest_distance
        self.distances = {}
        for start_node in self.graph.keys():
            self.distances[start_node] = self._bfs(start_node)

        # Extract box goal locations
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location

    def _bfs(self, start_node):
        """
        Perform Breadth-First Search starting from start_node to find distances
        to all reachable nodes in the graph.
        Returns a dictionary {node: distance}.
        """
        # Initialize distances for all nodes in the graph
        distances = {node: float('inf') for node in self.graph.keys()}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Get neighbors from the graph (handle nodes with no outgoing edges)
            neighbors = self.graph.get(current_node, set())

            for neighbor in neighbors:
                # If the neighbor hasn't been visited (distance is still infinity)
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is reached
        if self.goals <= state:
             return 0

        # Find current robot location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_location = parts[1]
                break
        # Assuming robot_location is always found in a valid state

        # Find current box locations for boxes relevant to the goal
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if it's an 'at' fact and the object is one of the goal boxes
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.box_goals:
                 box, location = parts[1], parts[2]
                 box_locations[box] = location

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each box not at its goal
        for box, goal_location in self.box_goals.items():
            current_location = box_locations.get(box)

            # If a box is somehow not in the state or its location is unknown,
            # or if its current/goal location is not in the graph,
            # it implies an unreachable state or invalid problem setup.
            # Return infinity in such cases.
            if current_location is None or current_location not in self.distances or goal_location not in self.distances:
                 return float('inf') # Should not happen in valid problems/states

            if current_location != goal_location:
                # Add box distance (estimated pushes)
                box_dist = self.distances[current_location].get(goal_location, float('inf'))

                # Add robot distance to the box (estimated moves)
                robot_dist = self.distances[robot_location].get(current_location, float('inf'))

                # If either distance is infinity, the state is likely unsolvable
                if box_dist == float('inf') or robot_dist == float('inf'):
                    return float('inf')

                total_cost += box_dist
                total_cost += robot_dist

        # If goal is not reached but total_cost is 0, this would be an issue.
        # However, if goals <= state is False, at least one box is not at its goal.
        # For that box, current_location != goal_location is true.
        # If current_location is reachable from itself (dist=0) and goal_location is reachable from current_location (dist > 0), box_dist > 0.
        # If robot_location is reachable from itself (dist=0) and current_location is reachable from robot_location (dist >= 0), robot_dist >= 0.
        # So, total_cost will be > 0 unless all non-goal boxes are unreachable (handled by inf check).

        return total_cost
