from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract the components of a PDDL fact string.
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_4_4)".
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
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its assigned goal location.
    The distance is computed on the graph defined by the 'adjacent' predicates.
    This heuristic is non-admissible as it ignores the robot's position, the cost
    of moving the robot, and potential blockages or deadlocks.

    # Assumptions
    - Each box has a unique goal location specified in the problem.
    - The grid structure and connectivity are defined solely by the 'adjacent' predicates.
    - The heuristic ignores the robot's position and the cost/feasibility of moving the robot to push boxes.
    - The heuristic ignores potential deadlocks where a box is pushed into a corner or blocked.

    # Heuristic Initialization
    - Parse the goal conditions to determine the target location for each box.
    - Parse the static 'adjacent' facts to build a graph representation of the locations.
    - Precompute the shortest path distances between all pairs of locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task's goal conditions.
    2. Build an undirected graph where locations are nodes and edges connect locations
       if they are declared 'adjacent' in the static facts.
    3. Compute shortest path distances between all pairs of locations in this graph
       using BFS starting from each node. Store these distances in a dictionary
       for quick lookup.
    4. For a given state:
       - Find the current location of each box that has a goal.
       - Initialize the total heuristic value to 0.
       - For each box that has a goal:
         - Get the box's current location.
         - If the box is not at its goal location:
           - Look up the precomputed shortest distance from the box's current location
             to its goal location using the precomputed distances.
           - If the goal location is unreachable from the current box location
             (distance is infinity), add a large penalty to the total heuristic value.
           - Otherwise, add the computed distance to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for distance precomputation.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            # Assuming goal is always (at ?box ?location)
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.box_goals[box] = location

        # 2. Build the location graph from adjacent facts
        self.graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add undirected edge
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Ensure all locations mentioned in goals or initial state are in the graph keys,
        # even if they have no adjacent facts (isolated). This prevents KeyError during BFS.
        locations_to_add = set()
        for loc in self.box_goals.values():
             locations_to_add.add(loc)
        for fact in task.initial_state:
             if match(fact, "at-robot", "*"):
                 _, loc = get_parts(fact)
                 locations_to_add.add(loc)
             elif match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 # Only add locations of boxes that have goals
                 if obj in self.box_goals:
                     locations_to_add.add(loc)

        for loc in locations_to_add:
             if loc not in self.graph:
                 self.graph[loc] = []


        # Remove duplicates in adjacency lists
        for loc in self.graph:
            self.graph[loc] = list(set(self.graph[loc]))

        # 3. Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.graph:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        # Initialize distances to infinity for all nodes in the graph
        distances = {node: float('inf') for node in self.graph}
        # Distance from start_node to itself is 0
        distances[start_node] = 0
        # Queue for BFS, starting with the start_node
        queue = deque([start_node])

        # Perform BFS
        while queue:
            current_node = queue.popleft()

            # Check if the current_node exists in the graph keys (should always be true here)
            if current_node in self.graph:
                # Explore neighbors
                for neighbor in self.graph[current_node]:
                    # If the neighbor hasn't been visited yet (distance is infinity)
                    if distances[neighbor] == float('inf'):
                        # Update distance
                        distances[neighbor] = distances[current_node] + 1
                        # Add neighbor to the queue
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state as a frozenset of facts

        # Find current locations of all boxes that have goals
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                if obj in self.box_goals: # Only track boxes that have goals
                    box_locations[obj] = loc

        total_h = 0  # Initialize the total heuristic value

        # Sum distances for boxes not at their goals
        for box, goal_loc in self.box_goals.items():
            # Get the current location of the box. Use .get() for safety,
            # although in valid Sokoban states, a box should always be 'at' a location.
            current_loc = box_locations.get(box)

            # If for some reason a box with a goal isn't 'at' any location in the state,
            # this state is likely invalid or problematic. Assign a large penalty.
            if current_loc is None:
                 total_h += 10000 # Large penalty for missing box location
                 continue

            # If the box is not yet at its goal location
            if current_loc != goal_loc:
                # Look up the precomputed shortest distance from the current box location
                # to its goal location.
                # Use .get() to handle cases where current_loc or goal_loc might not
                # be in the precomputed distances (e.g., truly isolated locations).
                # Default to infinity if not found.
                distance = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                # If the goal location is unreachable from the current box location
                # (distance is infinity), this state is likely a dead end or part
                # of an unsolvable path. Assign a large penalty.
                if distance == float('inf'):
                    # Assign a large penalty. This value should be larger than
                    # any possible finite distance in the graph.
                    total_h += 1000 # Arbitrary large penalty
                else:
                    # Add the shortest distance to the total heuristic value
                    total_h += distance

        return total_h
