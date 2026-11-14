from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(adjacent box1 location2)".
    - `args`: The expected pattern parts (wildcards `*` allowed).
    - Returns True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    Summary:
    This heuristic estimates how many actions are approximately needed to move all boxes into their goal adjacent positions (marked by adjacent-2 predicates).
    It uses the facts about box current adjacency to locations, robot position, and which boxes are already in the goal configuration
    (adjacent-2 predicate). Given the domain operators, pushing boxes and moving the robot are essential actions. The heuristic reflects
    the number of boxes that still need to be pushed adjacent-2 plus estimated costs for the robot moving among locations to achieve it.

    Assumptions:
    - The goal is that all boxes satisfy the (adjacent-2 ?box) predicate.
    - Each box adjacent to some location can be pushed one step to become adjacent-2.
    - The robot moves between locations (at-robot-2 predicates) and may need to reposition to push boxes.
    - Moving costs and pushing costs are approximated linearly (one action per move or push).
    - The heuristic does not have to be admissible; it aims for close estimates with efficiency.

    Heuristic Initialization:
    - Extract all boxes from facts.
    - Extract static adjacency relations from static facts for robot movement.
    - Build a graph of locations connected by 'clear' predicates to approximate robot shortest path distances.
    - Extract static box-to-robot adjacency (at-robot ?box ?location) from static facts.
    - Precompute adjacency sets to enable quick lookup during heuristic calls.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Count how many boxes are NOT yet in the goal position (do not have (adjacent-2 ?box)).
    2. For each box not yet adjacent-2, estimate the cost to:
       a) Move the robot adjacent to the box (using robot position and adjacency graph).
       b) Push that box once to achieve adjacent-2.
    3. Sum the pushing actions needed (equal to count of boxes not yet adjacent-2).
    4. Add a heuristic estimate of robot movement cost:
       - Use shortest path between robot location and the closest box that needs pushing,
       - plus the sum of minimal distances between boxes that need to be pushed,
       - as the robot must move among them to push all boxes.
       - Since computing exact minimal traveling path is expensive, approximate by summing distances from robot to each box's adjacency location.
    5. Return sum of pushes plus a scaled robot movement estimate to yield a heuristic guiding the search efficiently.

    Note:
    - The heuristic uses a very rough distance estimate but balances informativeness and computational cost.
    - It always returns 0 if the state is a goal state.
    """

    def __init__(self, task):
        """
        Initialize heuristic by extracting static adjacency and robot-box adjacency maps
        and build a graph of locations from clear predicates for navigating robot.

        Parameters:
        - task: The planning task containing goals, operators, static facts, etc.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all boxes appearing in static facts
        # Note: boxes appear in at-robot facts: (at-robot box location)
        self.boxes = set()
        for fact in task.static:
            if match(fact, "at-robot", "*", "*"):
                parts = get_parts(fact)
                self.boxes.add(parts[1])

        # Extract all locations appearing in static facts (locations appear in clear predicates and at-robot facts)
        self.locations = set()
        for fact in task.static:
            if match(fact, "clear", "*", "*"):
                parts = get_parts(fact)
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif match(fact, "at-robot", "*", "*"):
                parts = get_parts(fact)
                self.locations.add(parts[2])
            elif match(fact, "clear", "*", "*"):
                parts = get_parts(fact)
                self.locations.add(parts[1])
                self.locations.add(parts[2])
        # Defensive fallback: also add from goals and adjacent predicates if any missing
        for fact in task.static:
            if match(fact, "adjacent", "*", "*"):
                parts = get_parts(fact)
                self.boxes.add(parts[1])
                self.locations.add(parts[2])
            if match(fact, "adjacent-2", "*"):
                parts = get_parts(fact)
                self.boxes.add(parts[1])

        # Build robot movement graph between locations using clear facts (bidirectional edges)
        self.robot_graph = {loc: set() for loc in self.locations}
        for fact in task.static:
            if match(fact, "clear", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in self.robot_graph and loc2 in self.robot_graph:
                    self.robot_graph[loc1].add(loc2)
                    self.robot_graph[loc2].add(loc1)

        # Map from each box to the location(s) where the robot can push it,
        # derived from static facts (at-robot box location)
        self.box_to_robot_adj_locs = {}
        for fact in task.static:
            if match(fact, "at-robot", "*", "*"):
                parts = get_parts(fact)
                box = parts[1]
                loc = parts[2]
                if box not in self.box_to_robot_adj_locs:
                    self.box_to_robot_adj_locs[box] = set()
                self.box_to_robot_adj_locs[box].add(loc)

        # To speed up computations, cache location distances with BFS
        self.location_dist_cache = {}

    def location_distance(self, start, goal):
        """
        Compute shortest distance between two locations using BFS on robot_graph.
        Uses simple caching to speed up repeated calls.
        Returns integer distance (number of edges) or None if no path exists.
        """
        if start == goal:
            return 0
        key = (start, goal)
        if key in self.location_dist_cache:
            return self.location_dist_cache[key]
        # BFS
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.robot_graph.get(current, []):
                if neighbor == goal:
                    self.location_dist_cache[key] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        self.location_dist_cache[key] = None
        return None

    def __call__(self, node):
        """
        Compute heuristic estimate from given state node.

        Steps:
        1. If goal is reached (boxes all adjacent-2), return 0.
        2. Otherwise, for each box not adjacent-2:
           - Estimate robot moves needed to get adjacent to box's push location.
           - Add one push action to move box to adjacent-2.
        3. Sum pushes and robot movements estimates to produce heuristic.
        """
        state = node.state

        # Quick goal check: if all goals satisfied -> heuristic=0
        if self.goals <= state:
            return 0

        # Extract boxes that are already adjacent-2 in current state
        boxes_adjacent_2 = set()
        for fact in state:
            if match(fact, "adjacent-2", "*"):
                parts = get_parts(fact)
                boxes_adjacent_2.add(parts[1])

        # Identify boxes that still need to be pushed into adjacent-2 position
        boxes_to_push = self.boxes - boxes_adjacent_2
        if not boxes_to_push:
            # All boxes in place; no further pushes needed
            return 0

        # Find robot current location (at-robot-2 ?location)
        robot_location = None
        for fact in state:
            if match(fact, "at-robot-2", "*"):
                parts = get_parts(fact)
                robot_location = parts[1]
                break
        if robot_location is None:
            # Defensive: no robot location known, return large heuristic
            return 10**6

        # For each box, find minimal robot distance to one of its push locations
        # Sum the estimated push cost (1 action per box) plus robot movement approximated sum of distances
        total_push_cost = len(boxes_to_push)  # Each box must be pushed once at least

        # To estimate robot moves, sum minimal distance from robot to each box's push location
        total_robot_move_estimate = 0
        for box in boxes_to_push:
            push_locs = self.box_to_robot_adj_locs.get(box)
            if not push_locs:
                # No known push location for box, heuristic is pessimistic very high
                return 10**6
            min_dist = None
            for loc in push_locs:
                dist = self.location_distance(robot_location, loc)
                if dist is not None and (min_dist is None or dist < min_dist):
                    min_dist = dist
            if min_dist is None:
                # No path for robot to push location: unsolvable (heuristic very high)
                return 10**6
            # Add distance to reach this box for pushing
            total_robot_move_estimate += min_dist

        # Combine costs: pushing boxes + robot moves needed to reach them
        # Weight robot moves less to reflect approximate effort (heuristic not admissible)
        heuristic_value = total_push_cost + 0.5 * total_robot_move_estimate

        return heuristic_value
