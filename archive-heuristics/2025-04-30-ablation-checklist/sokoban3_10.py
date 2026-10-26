from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper functions to parse PDDL facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for Breadth-First Search (BFS)
def bfs_distance(start, end, graph):
    """
    Calculates the shortest path distance between two locations in the graph
    using BFS. Returns math.inf if the end is unreachable from the start.
    """
    if start == end:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return math.inf # Goal not reachable

# Helper function for BFS to find distance to any node in a set
def bfs_distance_to_set(start, targets, graph):
    """
    Calculates the shortest path distance from a start location to the nearest
    location in a set of target locations using BFS.
    Returns math.inf if no target is reachable from the start.
    """
    if start in targets:
        return 0 # Robot is already at a target location

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc in targets:
            return dist

        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return math.inf # No target reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations and position the robot to start pushing.

    The heuristic is calculated as the sum of:
    1. The shortest path distance for each box from its current location to its
       goal location (on the empty grid, ignoring obstacles). This estimates
       the minimum number of push actions needed for the boxes.
    2. The shortest path distance for the robot from its current location to
       the nearest location adjacent to any box that is not yet at its goal.
       This estimates the cost for the robot to get into a position to start
       working on an off-goal box.

    # Assumptions:
    - The grid structure is defined by `adjacent` facts.
    - Shortest paths on the empty grid are relevant estimates.
    - The robot needs to be adjacent to a box to push it.
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).

    # Heuristic Initialization
    - Extracts goal locations for each box from the task goals.
    - Builds an adjacency graph representing the traversable locations
      and connections from static `adjacent` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box from the state.
    3. Determine which boxes are not yet at their goal locations by comparing
       current box locations with the pre-computed goal locations.
    4. If there are no boxes off-goal, the heuristic value is 0 (goal state).
    5. Calculate the 'box_distances' component:
       - For each box that is not at its goal:
         - Find the shortest path distance from the box's current location
           to its goal location using BFS on the adjacency graph.
         - Sum these distances.
       - If any box's goal is unreachable on the empty grid, the state is
         likely unsolvable, return math.inf.
    6. Calculate the 'robot_distance' component:
       - Identify all locations that are adjacent to *any* box that is
         currently not at its goal. These are potential locations where the
         robot could position itself to push a box.
       - Find the shortest path distance from the robot's current location
         to the nearest location in this set of potential robot locations
         using BFS.
       - If the robot cannot reach any such location, the state might be
         unsolvable or require complex moves not captured, return math.inf.
    7. The total heuristic value is the sum of 'box_distances' and
       'robot_distance'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the adjacency graph from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goal predicates are always (at boxX loc_Y_Z)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.box_goals[box] = location

        # Build the adjacency graph from static facts.
        # The graph maps a location to a set of adjacent locations.
        self.adjacency_list = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            # Assuming adjacent predicates are always (adjacent loc1 loc2 dir)
            if predicate == "adjacent" and len(args) == 3:
                loc1, loc2, direction = args
                if loc1 not in self.adjacency_list:
                    self.adjacency_list[loc1] = set()
                self.adjacency_list[loc1].add(loc2)
                # Assuming adjacency is symmetric, add the reverse direction
                if loc2 not in self.adjacency_list:
                    self.adjacency_list[loc2] = set()
                self.adjacency_list[loc2].add(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at-robot" and len(args) == 1:
                robot_loc = args[0]
                break

        # If robot location is not found, state is invalid for this domain
        if robot_loc is None:
             return math.inf

        # Find box locations
        box_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            # Assuming box location predicates are always (at boxX loc_Y_Z)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                box_locations[box] = location

        # Identify boxes not at their goal
        off_goal_boxes = {
            box for box, goal_loc in self.box_goals.items()
            if box_locations.get(box) != goal_loc
        }

        # If all boxes are at their goals, the heuristic is 0
        if not off_goal_boxes:
            return 0

        # Calculate sum of box distances to goals
        box_distances = 0
        for box in off_goal_boxes:
            current_loc = box_locations.get(box)
            goal_loc = self.box_goals.get(box) # Use .get() in case a box in state isn't in goals

            # If a box is in the state but not in goals, or its location/goal
            # is not in the graph, this state is problematic.
            if current_loc is None or goal_loc is None or \
               current_loc not in self.adjacency_list or goal_loc not in self.adjacency_list:
                 return math.inf # Indicate an invalid or unsolvable state

            dist = bfs_distance(current_loc, goal_loc, self.adjacency_list)

            # If a box cannot reach its goal on the empty grid, this state is likely unsolvable
            if dist == math.inf:
                return math.inf

            box_distances += dist

        # Calculate robot distance to nearest location adjacent to an off-goal box
        target_robot_locations = set()
        for box in off_goal_boxes:
            box_loc = box_locations.get(box)
            if box_loc in self.adjacency_list: # Ensure box location is in the graph
                 # Add all locations adjacent to the box's current location
                 target_robot_locations.update(self.adjacency_list[box_loc])

        # If there are no locations adjacent to any off-goal box (e.g., box is in a corner
        # with no adjacent clear spots in the graph), this might be a dead end or requires complex
        # maneuvers not captured by this simple adjacency check. Return inf.
        # Note: This might be too strict if the box needs to be moved first to open up an adjacent spot.
        # However, for a greedy heuristic, returning inf for potentially blocked states is reasonable.
        if not target_robot_locations:
             return math.inf

        robot_distance = bfs_distance_to_set(robot_loc, target_robot_locations, self.adjacency_list)

        # If robot cannot reach any location adjacent to an off-goal box
        if robot_distance == math.inf:
            return math.inf

        # Total heuristic is sum of box distances and robot distance to get started
        # This is non-admissible as it sums independent box goals and adds robot cost.
        return box_distances + robot_distance

