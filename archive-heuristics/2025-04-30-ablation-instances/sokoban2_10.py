from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing two components:
    1. The sum of shortest path distances for each misplaced box to its goal location.
    2. The shortest path distance for the robot to the nearest misplaced box.
    The shortest path distances are calculated on the grid graph defined by adjacent locations.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Goal states are defined by the `at` predicate for boxes.
    - The heuristic does not explicitly check for complex deadlocks (e.g., boxes pushed into corners or blocking other boxes' goal paths). It only returns infinity if a box cannot reach its goal or the robot cannot reach any misplaced box based on grid connectivity.
    - The cost of moving the robot and pushing a box is implicitly assumed to be related to grid distance.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an undirected graph representing the grid connectivity from the `adjacent` static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot by finding the fact `(at-robot ?l)` in the state.
    2. Identify the current location of each box by finding facts `(at ?b ?l)` in the state.
    3. Determine which boxes are not at their goal locations by comparing current box locations with the goal locations stored during initialization. These are the misplaced boxes.
    4. If there are no misplaced boxes, the state is a goal state, and the heuristic is 0.
    5. Initialize the total heuristic value `h` to 0.
    6. Calculate the sum of box-to-goal distances: For each misplaced box:
       a. Get its current location and its goal location from the stored goal locations.
       b. Calculate the shortest path distance between the box's current location and its goal location using a Breadth-First Search (BFS) on the grid graph built from `adjacent` facts.
       c. Add this distance to `h`. If the BFS returns infinity (meaning no path exists between the box's current location and its goal location), the state is likely unsolvable or in a severe deadlock; return infinity immediately.
    7. Calculate the robot-to-nearest-misplaced-box distance:
       a. Find the misplaced box whose current location has the minimum shortest path distance from the robot's current location using BFS on the grid graph.
       b. Calculate this minimum distance.
       c. Add this minimum distance to `h`. If the minimum distance is infinity (meaning the robot cannot reach any misplaced box), the state is likely unsolvable; return infinity immediately.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.goal_locations[box] = loc

        # Build the adjacency graph from adjacent facts.
        self.graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Graph is undirected

    def bfs(self, start_loc, end_loc):
        """
        Find the shortest path distance between two locations on the grid graph using BFS.
        Returns float('inf') if no path exists or if start/end locations are not in the graph.
        """
        if start_loc == end_loc:
            return 0

        # Ensure start and end locations are valid nodes in the graph
        if start_loc not in self.graph or end_loc not in self.graph:
             return float('inf') # Cannot navigate if locations are not part of the defined grid

        queue = deque([(start_loc, 0)]) # (location, distance)
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Neighbors are guaranteed to be in the graph if current_loc is
            for neighbor in self.graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # No path found

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find robot location
        robot_l = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_l = get_parts(fact)
                break

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        # Identify misplaced boxes
        misplaced_boxes = []
        # Iterate through goal locations to find boxes that are not there
        for box, goal_l in self.goal_locations.items():
            # Check if the box exists in the current state and is not at its goal
            if box in box_locations and box_locations[box] != goal_l:
                 misplaced_boxes.append(box)
            # Also consider boxes that are in the goal but shouldn't be there according to goals
            # (This is less common in Sokoban goals, but good practice)
            # However, the typical Sokoban goal is just (at boxX goalY), not (not (at boxX nonGoalZ))
            # So we focus on boxes that *should* be at a goal but aren't.

        # If no misplaced boxes (all boxes are at their goal locations), goal reached
        if not misplaced_boxes:
            return 0

        h = 0

        # Calculate sum of box-to-goal distances
        for box in misplaced_boxes:
            box_l = box_locations[box]
            goal_l = self.goal_locations[box]
            dist = self.bfs(box_l, goal_l)
            if dist == float('inf'):
                 # If a box cannot reach its goal location on the grid, it's likely unsolvable.
                 return float('inf')
            h += dist

        # Calculate robot-to-nearest-misplaced-box distance
        min_robot_dist = float('inf')
        for box in misplaced_boxes:
            box_l = box_locations[box]
            dist = self.bfs(robot_l, box_l)
            min_robot_dist = min(min_robot_dist, dist)

        if min_robot_dist == float('inf'):
            # If the robot cannot reach any misplaced box location on the grid, it's likely trapped.
            return float('inf')

        h += min_robot_dist

        return h
