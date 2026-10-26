# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from start_node
    to all other reachable nodes in the graph.

    Args:
        graph: An adjacency list representation {node: [neighbor1, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary {node: distance} containing shortest distances.
    """
    # Initialize distances for all nodes in the graph
    distances = {node: float('inf') for node in graph}
    if start_node not in distances:
        # Start node is not in the graph, cannot reach anything from here
        return distances # All distances remain inf

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'): # Check if visited
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

# Define the heuristic class
# Replace 'object' with 'Heuristic' if the base class is available
class sokobanHeuristic(object): # Inherit from object or Heuristic
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The shortest path distance for each box from its current location to its goal location.
    2. The shortest path distance for the robot from its current location to a location adjacent to *any* box that is not yet at its goal.

    # Assumptions
    - The environment is a graph defined by `adjacent` predicates.
    - Shortest path distances are calculated using BFS on this graph.
    - The cost of moving a box one step towards its goal is at least 1 (the push action).
    - The robot must be adjacent to a box to push it. The cost to get the robot into position is approximated by the minimum distance to *any* adjacent location of *any* box needing a push.
    - This heuristic is non-admissible.

    # Heuristic Initialization
    - Builds an undirected graph from `adjacent` facts.
    - Computes all-pairs shortest paths using BFS and stores them.
    - Extracts the goal location for each box from the task goals.
    - Identifies all box objects.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Initialize the total heuristic cost to 0.
    3. Initialize the minimum robot-to-box-adjacent distance to infinity.
    4. For each box:
       - Get its current location and its goal location.
       - If the box is not at its goal location:
         - Add the precomputed shortest path distance from the box's current location to its goal location to the total cost. This estimates the minimum number of pushes needed for this box.
         - Find all locations adjacent to the box's current location using the precomputed graph.
         - For each adjacent location, calculate the precomputed shortest path distance from the robot's current location to this adjacent location.
         - Update the minimum robot-to-box-adjacent distance found so far with the minimum of these distances.
    5. If the minimum robot-to-box-adjacent distance is not infinity (meaning there was at least one box not at its goal), add this minimum distance to the total cost. This estimates the cost for the robot to get into position for the first necessary push.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph,
        computing shortest paths, and extracting goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                # Add bidirectional edges
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1)

        # Collect all locations mentioned in static facts and goals
        all_locations = set(locations)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, _, loc = get_parts(goal)
                 all_locations.add(loc)

        # Ensure all locations are keys in the graph dictionary, even if they have no neighbors
        # This is important so BFS can compute distances to/from them.
        for loc in all_locations:
             self.graph.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.graph:
             self.distances[start_loc] = bfs(self.graph, start_loc)

        # Store goal locations for each box
        self.goal_locations = {}
        self.boxes = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location
                self.boxes.add(box) # Collect box names

        # Collect box names from initial state as well, just in case
        # (though goals define which boxes are relevant for the heuristic)
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, _ = get_parts(fact)[1:]
                 # Assuming anything typed 'box' is a box object based on domain/examples
                 # A more robust way would be to parse object definitions if available
                 if obj.startswith('box'):
                     self.boxes.add(obj)


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, handling potential missing locations."""
        # Check if the start location is in our precomputed distances map
        if loc1 not in self.distances:
             # This location was not part of the graph built from adjacent facts
             # and was not added from goals. It's an unknown location.
             return float('inf')

        # Return the precomputed distance. This will be inf if loc2 is unreachable
        # from loc1 within the graph.
        return self.distances[loc1].get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        if robot_loc is None:
             # Should not happen in a valid state according to domain definition
             return float('inf') # Cannot proceed without robot location

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.boxes: # Only track locations of relevant boxes
                    current_box_locations[obj] = loc

        total_h = 0
        min_robot_to_box_adj_dist = float('inf')
        found_box_not_at_goal = False

        for box in self.boxes:
            box_loc = current_box_locations.get(box)
            goal_loc = self.goal_locations.get(box)

            # Ensure box and goal locations are known. If not, something is wrong with the state/task.
            if box_loc is None or goal_loc is None:
                 # This box is either not in the current state or has no goal location defined.
                 # This shouldn't happen in a well-formed problem instance.
                 continue # Skip this box, or return inf? Let's skip for robustness.

            if box_loc != goal_loc:
                found_box_not_at_goal = True
                # Add box-goal distance
                box_goal_dist = self.get_distance(box_loc, goal_loc)
                if box_goal_dist == float('inf'):
                    # Box is in a part of the graph disconnected from its goal
                    return float('inf') # Unsolvable state
                total_h += box_goal_dist

                # Find minimum robot distance to adjacent locations of this box
                # Ensure box_loc is a node in the graph (should be if get_distance didn't return inf)
                if box_loc in self.graph:
                    for adj_loc in self.graph[box_loc]:
                        robot_adj_dist = self.get_distance(robot_loc, adj_loc)
                        # We only consider reachable adjacent locations
                        if robot_adj_dist != float('inf'):
                             min_robot_to_box_adj_dist = min(min_robot_to_box_adj_dist, robot_adj_dist)
                # else: box_loc is isolated or not in graph, handled by box_goal_dist check.


        # Add the minimum robot distance to get near *any* box that needs pushing
        # Only add this cost if there are boxes not at goals AND the robot can actually reach
        # an adjacent location of at least one such box.
        if found_box_not_at_goal:
             if min_robot_to_box_adj_dist == float('inf'):
                 # There are boxes not at goals, but the robot cannot reach any location
                 # adjacent to any of them.
                 return float('inf')
             else:
                 total_h += min_robot_to_box_adj_dist


        return total_h
