from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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
    # Ensure pattern length matches fact parts length for a valid match attempt
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box not at its goal,
    the shortest path distance from the box's current location to its goal location,
    plus the shortest path distance from the robot's current location to any location adjacent to the box.
    This captures the effort required to move the box and the effort required for the robot to get into a pushing position.
    It is a non-admissible heuristic designed to guide a greedy best-first search.

    # Assumptions
    - The goal is to move all boxes to their specified goal locations.
    - Locations are connected by 'adjacent' predicates, forming an undirected graph for distance calculation.
    - Box names start with "box".
    - Goal facts are of the form (at box_name goal_location).
    - The locations mentioned in the state and goals are part of the graph defined by static 'adjacent' facts.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph representation of the locations based on the 'adjacent' static facts. This graph is used for shortest path calculations (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state is a goal state by comparing the state facts with the task's goal facts. If it is a goal state, the heuristic value is 0.
    2.  Identify the robot's current location from the state facts. If the robot's location cannot be found, return infinity (indicating an invalid or unsolvable state).
    3.  Identify the current location of each box from the state facts. Store these in a dictionary mapping box names to locations.
    4.  Initialize the total heuristic value to 0.
    5.  Iterate through each box and its corresponding goal location as extracted during initialization.
    6.  For the current box:
        a.  Get its current location from the dictionary created in step 3. If the box is not found in the current state, return infinity.
        b.  If the box is already at its goal location, its contribution to the heuristic is 0; continue to the next box.
        c.  If the box is not at its goal location:
            i.  Calculate the shortest path distance (`box_dist`) between the box's current location and its goal location using Breadth-First Search (BFS) on the location graph. This estimates the minimum number of steps the box needs to move.
            ii. Calculate the minimum shortest path distance (`min_robot_dist_to_adj`) from the robot's current location to *any* location that is directly adjacent to the box's current location. This also uses BFS. This estimates the minimum number of steps the robot needs to move to get into a position from which it *could* potentially push the box.
            iii. If either `box_dist` or `min_robot_dist_to_adj` is infinity (meaning the goal location is unreachable from the box, or no location adjacent to the box is reachable by the robot), the state is likely unsolvable or represents a very poor path choice; return infinity.
            iv. Add the sum `box_dist + min_robot_dist_to_adj` to the `total_heuristic`. This sum represents the estimated effort for this specific box to reach its goal, considering both the box's movement and the robot's positioning.
    7.  After processing all boxes, return the `total_heuristic` value.
    """
    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.
        """
        self.goals = task.goals # Store goals to check for goal state

        # Extract goal locations for boxes
        self.goal_locations = {}
        for goal in task.goals:
            # Goal facts are typically (at box_name goal_location)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts
        # The graph is undirected for simple distance calculation
        self.graph = {}
        for fact in task.static:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent" and len(args) == 3:
                loc1, loc2, direction = args
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add edges in both directions
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Remove duplicates from adjacency lists (BFS handles cycles, but unique neighbors are cleaner)
        for loc in self.graph:
             self.graph[loc] = list(set(self.graph[loc]))

    def bfs_distance(self, start, end):
        """
        Calculates the shortest path distance between two locations using BFS.
        Returns float('inf') if end is unreachable from start or if start/end are not in the graph.
        """
        if start == end:
            return 0
        if start not in self.graph or end not in self.graph:
             return float('inf') # Cannot calculate distance if locations are unknown

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Ensure current_loc is still valid in graph during traversal
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # End is unreachable

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached first for efficiency and correctness (h=0 at goal)
        if self.goals <= state:
             return 0

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # Robot location must be known in a valid state
             return float('inf') # Should not happen in a solvable problem

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            # Assuming box names start with "box" and are the first argument to "at"
            if match(fact, "at", "box*", "*"):
                 box, loc = get_parts(fact)[1:]
                 current_box_locations[box] = loc

        total_heuristic = 0

        # Calculate heuristic for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If a box from the goal is not found in the current state, something is wrong
            if current_loc is None:
                 return float('inf') # State is invalid or box disappeared

            if current_loc != goal_loc:
                # Distance from box to its goal
                box_dist = self.bfs_distance(current_loc, goal_loc)

                # Distance from robot to a location adjacent to the box
                min_robot_dist_to_adj = float('inf')
                # Need to find locations adjacent to the box's current location
                if current_loc in self.graph:
                    for adj_loc in self.graph[current_loc]:
                        # Calculate distance from robot to this adjacent location
                        dist = self.bfs_distance(robot_loc, adj_loc)
                        min_robot_dist_to_adj = min(min_robot_dist_to_adj, dist)
                else:
                     # Box is at a location not in the graph? Invalid state.
                     return float('inf')

                # If box goal is unreachable or robot cannot reach box-adjacent location,
                # the state is likely unsolvable or very bad.
                if box_dist == float('inf') or min_robot_dist_to_adj == float('inf'):
                     return float('inf')

                # Add the heuristic contribution for this box.
                # This heuristic sums the effort for the box to move and the robot to get into position.
                # It's non-admissible as it doesn't consider conflicts (other boxes, walls)
                # or the fact that moving one box might help/hinder moving another.
                total_heuristic += box_dist + min_robot_dist_to_adj

        return total_heuristic
