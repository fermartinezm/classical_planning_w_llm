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
    # Ensure the number of parts matches the number of args if no wildcards are used,
    # or handle wildcards appropriately. zip handles unequal lengths by stopping
    # at the shortest sequence, which is fine for matching prefixes or exact patterns.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated costs for each box
    that is not yet at its goal location. The estimated cost for a single box is the shortest
    path distance from the box's current location to its goal location, plus the shortest path
    distance from the robot's current location to the box's current location. This is a relaxation
    that ignores the specific robot positioning required for pushing and potential blockages by
    other boxes or walls.

    # Assumptions
    - The goal state specifies the target location for each box using `(at box_name goal_location)`.
    - There is a one-to-one mapping between boxes and goal locations defined by the goal state.
    - The cost of moving the robot and pushing a box is 1.
    - The heuristic ignores potential blockages by other boxes or walls, and the specific robot positioning required for pushing.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goal conditions.
    - Build a graph representation of the locations and their adjacencies based on static facts.
    - Precompute shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is the goal state. If yes, the heuristic is 0.
    2. Identify the current location of the robot by finding the fact `(at-robot ?l)`.
    3. Identify the current location of each box by finding facts `(at ?b ?l)` where `?b` is a box.
    4. Initialize the total heuristic value to 0.
    5. For each box that has a specified goal location (extracted during initialization):
       - Get the box's current location from the state.
       - Get the box's goal location (stored during initialization).
       - If the box's current location is not the same as its goal location:
         - Calculate the shortest path distance from the box's current location to its goal location using the precomputed distances. This is a lower bound on the number of pushes needed for this box if the path were clear and the robot always in position.
         - Calculate the shortest path distance from the robot's current location to the box's current location using the precomputed distances. This is an estimate of the robot movement needed to reach the box initially.
         - If either distance is infinite (meaning the location is unreachable in the graph), the state is likely unsolvable or very bad; return infinity.
         - Add the sum of these two distances to the total heuristic value.
    6. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the location graph,
        and precomputing shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        # Assumes goal facts are of the form (at box_name location)
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Check if the goal fact is an 'at' predicate for a box
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location

        # Build the graph of locations based on adjacent facts.
        # The graph is undirected as adjacency is symmetric (e.g., up/down).
        self.graph = {}
        locations = set()
        for fact in static_facts:
            # Match adjacent facts with 3 arguments (loc1, loc2, dir)
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1) # Add reverse edge

        # Precompute shortest path distances between all pairs of locations using BFS.
        self.distances = {}
        for start_node in locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Perform a Breadth-First Search starting from start_node to find distances
        to all reachable nodes.
        """
        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Ensure current_node is a valid key in the graph
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if neighbor not in distances:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """
        Get the precomputed shortest path distance between loc1 and loc2.
        Returns float('inf') if loc2 is unreachable from loc1 or if loc1 is not in the graph.
        """
        # Check if the start location exists in our precomputed distances
        if loc1 in self.distances:
            # Check if the target location is reachable from the start location
            if loc2 in self.distances[loc1]:
                return self.distances[loc1][loc2]
            # If start exists but target is not in its distance map, it's unreachable
            return float('inf')
        # If start location itself is not in the graph/distances, it's unreachable
        return float('inf')

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Heuristic is 0 iff the goal is reached.
        if self.goals <= state:
             return 0

        # Find robot location.
        robot_location = None
        for fact in state:
            # Match the (at-robot ?l) fact
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not found, something is fundamentally wrong with the state.
        if robot_location is None:
             return float('inf') # Indicate an invalid or unreachable state.

        # Find box locations.
        box_locations = {}
        for fact in state:
            # Match (at box_name location) facts
            if match(fact, "at", "box*", "*"):
                box, location = get_parts(fact)[1:]
                box_locations[box] = location

        total_heuristic = 0

        # Calculate heuristic contribution for each box that needs to reach a goal.
        # We iterate through the boxes defined in the goal state.
        for box, goal_location in self.goal_locations.items():
            current_box_location = box_locations.get(box) # Get box's current location, None if not found

            # If a box from the goal is not found in the state facts, or it's not at its goal location,
            # calculate its contribution.
            if current_box_location is None or current_box_location != goal_location:
                # If the box is missing from the state facts, it's an invalid state for this problem.
                if current_box_location is None:
                     return float('inf')

                # Distance from box's current location to its goal location.
                # This is a lower bound on the pushes needed.
                box_to_goal_dist = self.get_distance(current_box_location, goal_location)

                # Distance from robot's current location to the box's current location.
                # The robot needs to get near the box to push it.
                robot_to_box_dist = self.get_distance(robot_location, current_box_location)

                # If either the box cannot reach its goal or the robot cannot reach the box,
                # this state is likely unsolvable or very bad.
                if box_to_goal_dist == float('inf') or robot_to_box_dist == float('inf'):
                     return float('inf')

                # Add the combined distance for this box.
                # This heuristic is non-admissible. It sums distances independently
                # and doesn't account for coordination or blockages.
                total_heuristic += box_to_goal_dist + robot_to_box_dist

        return total_heuristic
