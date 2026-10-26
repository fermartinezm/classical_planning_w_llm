from fnmatch import fnmatch
from collections import deque
import math # Used for float('inf')

# Assuming the Heuristic base class is available in the specified path
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a mock Heuristic class for standalone testing if the base is not available
    print("Warning: heuristics.heuristic_base not found. Using mock Heuristic.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError
    # Mock node for testing
    class MockNode:
        def __init__(self, state):
            self.state = state
            self.cost = 0
            self.parent = None
            self.action = None
    # Mock task for testing
    class MockTask:
        def __init__(self, name, facts, initial_state, goals, operators, static):
            self.name = name
            self.facts = facts
            self.initial_state = initial_state
            self.goals = goals
            self.operators = operators
            self.static = static
        def goal_reached(self, state):
             return self.goals <= state


def get_parts(fact):
    """Helper function to split a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the shortest path distances for each box to its respective goal location
        and adding the shortest path distance from the robot to the nearest box
        that still needs to be moved. It is designed for greedy best-first search
        and is not admissible, but aims to guide the search efficiently.

    Assumptions:
        - The domain uses the predicates `at-robot`, `at`, `adjacent`, and `clear`.
        - The connectivity of locations is defined solely by `adjacent` facts,
          forming a graph structure.
        - Goal states are defined by the target locations for specific boxes
          using `(at box location)` predicates.
        - Location names follow a consistent format (e.g., `loc_X_Y`), although
          the heuristic relies only on the string names and the `adjacent` facts,
          not the coordinate structure.
        - The state representation is a frozenset of PDDL fact strings.
        - The heuristic is called on valid states reachable from the initial state.

    Heuristic Initialization:
        In the constructor (`__init__`), the heuristic performs the following steps:
        1.  Parses the `static_facts` provided by the task to build a graph
            representing the locations and their adjacencies based on the
            `(adjacent l1 l2 dir)` predicates.
        2.  Identifies all unique locations present in the graph.
        3.  Computes the shortest path distance between every pair of locations
            in the graph using Breadth-First Search (BFS). These distances are
            stored in a dictionary `self.distances` for quick lookup.
        4.  Extracts the target location for each box from the `task.goals`.
            This mapping is stored in `self.goal_locations`.

    Step-By-Step Thinking for Computing Heuristic:
        In the call method (`__call__`), for a given state:
        1.  It first checks if the current state satisfies all the goal conditions.
            If it does, the heuristic value is 0.
        2.  It identifies the current location of the robot by finding the fact
            `(at-robot ?l)` in the state.
        3.  It identifies the current location of each box that is specified
            in the goal state by finding the facts `(at ?b ?l)` for relevant boxes.
        4.  It initializes a sum `box_distance_sum` to 0 and creates a list
            `boxes_to_move` for boxes not yet at their goal.
        5.  For each box specified in the goal, it checks if the box is at its
            goal location in the current state.
        6.  If a box is not at its goal location, it retrieves the precomputed
            shortest path distance from the box's current location to its goal
            location using `self.distances`. This distance is added to
            `box_distance_sum`. If the goal location is unreachable from the
            box's current location, the heuristic returns `float('inf')`,
            indicating an likely unsolvable state.
        7.  After processing all boxes, if `boxes_to_move` is empty, it means
            all relevant boxes are at their goals, and the heuristic returns 0
            (this case should ideally be caught by the initial goal check,
            but serves as a fallback).
        8.  If there are boxes to move, it calculates the shortest path distance
            from the robot's current location to the location of the *nearest*
            box in the `boxes_to_move` list. This distance is stored in
            `min_robot_dist_to_box`. If the robot cannot reach any box that
            needs moving, the heuristic returns `float('inf')`.
        9.  The final heuristic value is the sum of `box_distance_sum` and
            `min_robot_dist_to_box`. This combines the effort needed to move
            the boxes themselves with the effort needed for the robot to get
            to a position to start pushing.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by building the location graph,
        computing all-pairs shortest paths, and extracting goal locations.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Heuristic Initialization

        # 1. Build the graph from adjacent facts
        self.graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                # Store neighbor. Direction is not needed for simple distance.
                self.graph[loc1].append(loc2)

        self.locations = list(all_locations) # Store as list

        # 2. Compute All-Pairs Shortest Paths (APSP) using BFS
        self.distances = {}
        for start_loc in self.locations:
            self._compute_shortest_paths_from(start_loc)

        # 3. Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _compute_shortest_paths_from(self, start_loc):
        """
        Computes shortest path distances from start_loc to all other locations
        using BFS and stores them in self.distances.
        """
        distances = {loc: math.inf for loc in self.locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr_loc = queue.popleft()

            # Ensure curr_loc is in graph (some locations might be isolated or only appear as neighbors)
            if curr_loc not in self.graph:
                 continue

            for neighbor_loc in self.graph[curr_loc]:
                if distances[neighbor_loc] == math.inf:
                    distances[neighbor_loc] = distances[curr_loc] + 1
                    queue.append(neighbor_loc)

        # Store results
        for end_loc in self.locations:
            self.distances[(start_loc, end_loc)] = distances[end_loc]


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer or float('inf') representing the estimated cost to reach the goal.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Step-By-Step Thinking for Computing Heuristic

        # 1. Find current locations of robot and boxes
        robot_loc = None
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1] in self.goal_locations:
                 current_box_locations[parts[1]] = parts[2]

        # Basic validation (should hold for valid states)
        if robot_loc is None or len(current_box_locations) != len(self.goal_locations):
             # This state might be malformed or represent an unhandled case.
             # Returning infinity is a safe default for potentially unsolvable states.
             return math.inf

        # 2. Calculate sum of box-to-goal distances
        box_distance_sum = 0
        boxes_to_move = []
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)
            if current_loc is not None and current_loc != goal_loc:
                # Add distance from current box location to goal box location
                dist = self.distances.get((current_loc, goal_loc), math.inf)
                if dist == math.inf:
                    # Box is in a location from which its goal is unreachable.
                    return math.inf
                box_distance_sum += dist
                boxes_to_move.append(box)

        # If all relevant boxes are at their goals, but the goal state wasn't met
        # by the initial check, it implies the goal includes non-box predicates
        # which this heuristic doesn't account for. Assuming standard Sokoban goals.
        if not boxes_to_move:
             return 0 # Should only happen if goal was reached

        # 3. Calculate robot's contribution
        # Add distance from robot to the nearest box that needs moving.
        min_robot_dist_to_box = math.inf
        for box in boxes_to_move:
            box_loc = current_box_locations[box]
            dist = self.distances.get((robot_loc, box_loc), math.inf)
            if dist != math.inf:
                 min_robot_dist_to_box = min(min_robot_dist_to_box, dist)

        # If robot cannot reach any box that needs moving, it's likely unsolvable
        if min_robot_dist_to_box == math.inf:
             return math.inf

        # Total heuristic is sum of box distances + robot distance to nearest box
        total_heuristic = box_distance_sum + min_robot_dist_to_box

        return total_heuristic

