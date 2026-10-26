# Add necessary imports
from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class is available from the planning framework
# If running standalone for testing, define a dummy class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError

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
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Map directions to their opposites (not strictly needed for this push_from_loc logic, but good practice)
# OPPOSITE_DIR = {
#     'up': 'down',
#     'down': 'up',
#     'left': 'right',
#     'right': 'left'
# }

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of shortest path distances for each box
    to its goal, plus the minimum shortest path distance for the robot
    to reach a position from which it can push any box towards its goal.

    This heuristic is non-admissible.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        precomputing all-pairs shortest paths.
        """
        super().__init__(task)

        self.locations = set()
        # graph: location -> list[(neighbor_location, direction)] (l1 -> l2 via dir)
        self.graph = {}
        # rev_graph: location -> list[(neighbor_location, direction)] (l2 -> l1 via dir)
        # Used to find location 'p' such that adjacent(p, l, dir) is true.
        self.rev_graph = {}

        # Build the graphs from adjacent facts
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.rev_graph:
                    self.rev_graph[loc2] = []
                self.graph[loc1].append((loc2, direction))
                # In rev_graph, store the location you came *from* (loc1) and the direction
                # you used to get *to* loc2. So, if adjacent(loc1, loc2, dir), then loc1
                # is a predecessor of loc2 via direction dir.
                self.rev_graph[loc2].append((loc1, direction))

        # Ensure all locations from goals/initial state are included, even if isolated
        for fact in task.initial_state:
             if match(fact, "at-robot", "*"):
                 _, loc = get_parts(fact)
                 self.locations.add(loc)
             elif match(fact, "at", "*", "*"):
                 _, box, loc = get_parts(fact)
                 self.locations.add(loc)
        for goal in task.goals:
             if match(goal, "at", "*", "*"):
                 _, box, loc = get_parts(goal)
                 self.locations.add(loc)

        # Initialize graph entries for isolated locations
        for loc in self.locations:
            if loc not in self.graph:
                self.graph[loc] = []
            if loc not in self.rev_graph:
                self.rev_graph[loc] = []


        # Precompute all-pairs shortest paths using BFS on the FORWARD graph
        # This calculates the minimum number of 'move' actions between any two locations.
        # This is also the minimum number of 'push' actions a box needs if it could move freely.
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node) # BFS uses self.graph

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node on the forward graph to find distances
        to all reachable nodes. Returns a dictionary {location: distance}.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in graph before iterating
            # This check is technically redundant if self.graph is built correctly from self.locations
            # but harmless.
            if current_node in self.graph:
                for neighbor, _ in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_loc = get_parts(fact)
                break
        if robot_loc is None or robot_loc not in self.locations:
             # Robot location unknown or not in graph (invalid state)
             return float('inf')

        # Find current box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        total_box_distance = 0
        min_robot_to_any_push_pos_distance = float('inf')
        any_box_not_at_goal = False

        # Consider only boxes that have a goal location specified
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            if current_location is None or current_location not in self.locations:
                 # Box location unknown or not in graph (invalid state)
                 return float('inf')

            if current_location == goal_location:
                continue # This box is already at its goal

            any_box_not_at_goal = True

            # Add box-goal distance (minimum pushes required for this box)
            box_dist = self.distances[current_location].get(goal_location, float('inf'))
            if box_dist == float('inf'):
                # Box goal is unreachable from its current location
                return float('inf') # Unsolvable state
            total_box_distance += box_dist

            # Calculate minimum robot distance to get behind *this* box
            # Robot needs to be at push_from_loc such that adjacent(push_from_loc, current_location, dir_forward)
            # where dir_forward is the direction from current_location to a neighbor 'n' that is closer to the goal.
            # This means push_from_loc is the location from which you can move to current_location using dir_forward.
            # We find such push_from_loc by looking in rev_graph[current_location] for entries with direction dir_forward.

            min_dist_this_box = float('inf')

            # Iterate through possible locations 'n' where the box could be pushed to from current_location
            if current_location in self.graph: # Ensure current_location is in our graph nodes
                for next_box_loc, dir_forward in self.graph[current_location]:
                    # Check if pushing to next_box_loc moves the box strictly closer to the goal
                    dist_to_goal_from_next = self.distances[next_box_loc].get(goal_location, float('inf'))
                    # Need to re-get box_dist as it might be inf if goal is unreachable
                    current_box_dist_to_goal = self.distances[current_location].get(goal_location, float('inf'))

                    if dist_to_goal_from_next < current_box_dist_to_goal:
                        # This is a useful push direction (current_location -> next_box_loc in dir_forward)

                        # Find the location push_from_loc such that adjacent(push_from_loc, current_location, dir_forward)
                        # Look in rev_graph[current_location] for entries with direction dir_forward
                        push_from_loc = None
                        if current_location in self.rev_graph:
                            for potential_push_loc, d in self.rev_graph[current_location]:
                                if d == dir_forward:
                                    push_from_loc = potential_push_loc
                                    break

                        if push_from_loc is not None and push_from_loc in self.locations:
                            robot_dist_this_push = self.distances[robot_loc].get(push_from_loc, float('inf'))
                            if robot_dist_this_push != float('inf'):
                                min_dist_this_box = min(min_dist_this_box, robot_dist_this_push)

            # Update the minimum robot distance needed to get behind *any* box
            if min_dist_this_box != float('inf'):
                 min_robot_to_any_push_pos_distance = min(min_robot_to_any_push_pos_distance, min_dist_this_box)


        # If all boxes are at their goals, heuristic is 0
        if not any_box_not_at_goal:
            return 0

        # Heuristic is sum of box distances + minimum robot distance to get behind any box
        h = total_box_distance

        # Add robot cost only if there is at least one box not at its goal
        if min_robot_to_any_push_pos_distance == float('inf'):
             # Robot cannot reach a position to push any box towards its goal.
             # This could indicate a dead end or a complex sequence is needed.
             # Return a large value to discourage this state.
             # This might happen if the robot is in an isolated part of the graph
             # or if all boxes are in dead ends where they cannot be pushed closer.
             return 1000000 # A large finite number

        h += min_robot_to_any_push_pos_distance

        return h
