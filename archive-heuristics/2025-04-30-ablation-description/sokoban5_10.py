# Imports needed
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available from heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

def bfs(graph, start, end):
    """
    Performs BFS on the location graph to find the shortest distance.
    Returns distance or float('inf') if unreachable.
    """
    if start == end:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Ensure current_loc is a valid node in the graph before accessing neighbors
        # A location should be in the graph if it appears in any 'adjacent' fact.
        # If a box is placed on a location not in the graph, BFS from there is impossible.
        # Assuming all locations mentioned in 'at' facts are part of the grid graph.
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # else: # If current_loc is not in graph, it has no neighbors to explore.

    return float('inf') # Should not happen for reachable goals in solvable problems

# Heuristic class definition
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    minimum grid distances for each box from its current location to its
    assigned goal location. The grid distance is calculated using Breadth-First Search (BFS)
    on the graph defined by the 'adjacent' facts, ignoring the robot's position
    and other obstacles.

    # Assumptions
    - There is a one-to-one mapping between boxes and goal locations defined in the task goals.
    - The grid defined by 'adjacent' facts is connected such that all relevant locations are reachable from each other (ignoring dynamic obstacles like boxes and the robot).
    - The heuristic assumes boxes can move freely, ignoring the robot's presence and the push mechanics. This makes it admissible.

    # Heuristic Initialization
    - Build a graph representation of the grid based on the 'adjacent' facts from the static information.
    - Extract the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of each box in the state.
    2. For each box, determine its corresponding goal location based on the task goals.
    3. If a box is already at its goal location, its contribution to the heuristic is 0.
    4. If a box is not at its goal location, calculate the shortest distance between the box's current location and its goal location using BFS on the pre-computed grid graph.
    5. Sum the distances calculated in step 4 for all boxes.
    6. The total sum is the heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and extracting
        box goal locations.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add edge loc1 -> loc2
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append(loc2)
                # Assuming adjacent facts are provided symmetrically in PDDL,
                # we don't need to explicitly add loc2 -> loc1 here.
                # If they weren't symmetric, we would add:
                # if loc2 not in self.graph:
                #     self.graph[loc2] = []
                # self.graph[loc2].append(loc1)


        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            # Goal is expected to be of the form (at ?b ?l)
            parts = get_parts(goal)
            # Check if it's an 'at' predicate with 3 parts (predicate, obj, loc)
            if parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location
            # Note: If the goal is a conjunction (and (...)(...)), task.goals will be the set of facts inside the 'and'.
            # So the current parsing logic is correct for goals like (and (at b1 l1) (at b2 l2)).


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find current locations of all boxes that are part of the goal
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate for a box
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.box_goals:
                box, location = parts[1], parts[2]
                current_box_locations[box] = location

        total_heuristic = 0

        # Calculate sum of distances for each box to its goal
        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box)

            # If a box listed in the goals is not found in the current state's 'at' facts,
            # something is wrong with the state representation or the problem definition.
            # Assuming valid states where all goal boxes have an 'at' fact.
            if current_location is None:
                 # This case should ideally not happen in a valid state representation
                 # where all objects exist and are located somewhere.
                 # If it could happen, returning infinity might be safer.
                 # For this simple heuristic, let's assume valid states and skip.
                 continue


            if current_location != goal_location:
                # Calculate distance using BFS on the location graph
                distance = bfs(self.graph, current_location, goal_location)

                # If distance is infinity, the goal location is unreachable on the grid
                # from the box's current location. This state is likely unsolvable
                # or indicates a malformed problem/grid. Return infinity.
                if distance == float('inf'):
                    return float('inf')

                total_heuristic += distance

        # The heuristic is the sum of distances. It is 0 iff all distances are 0,
        # which means all boxes are at their goal locations.

        return total_heuristic
