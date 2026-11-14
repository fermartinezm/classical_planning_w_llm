from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    For example, "(adjacent box1 location3)" -> ["adjacent", "box1", "location3"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.
    - fact: full fact string e.g. "(adjacent box1 location3)"
    - args: pattern elements e.g. "adjacent", "*", "location3"
    Returns True if all parts match, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to push all boxes into their
    goal condition, which is to have the predicate (adjacent-2 box) true for each box.
    Essentially, the goal is to push each box so it has predicate adjacent-2.

    Assumptions
    -----------
    - The goal predicates specify which boxes must have (adjacent-2 ?box).
    - Boxes must be pushed from current positions (possibly "at" or "adjacent") to a state
      where they satisfy the (adjacent-2 ?box) predicate.
    - Robot movements are necessary to position at the right location to push boxes.
    - The domain includes two kinds of moves (move-2 and push-2) involving the robot and locations,
      and moves for boxes (move, push) involving boxes and locations.
    - Static facts provide relationships like (adjacent box location) and (at-robot box location).

    Heuristic Initialization
    ------------------------
    During initialization, we extract:
    - The set of boxes that must satisfy (adjacent-2 box) in the goal.
    - For each box, the static facts providing which locations it is "adjacent" to.
    - For each box, which locations the robot is initially adjacent to it (from static facts).
    - We store the set of locations and clear relationships to estimate minimal push distances.
    - The heuristic uses only information derivable from static facts and goal conditions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all boxes that still need to satisfy (adjacent-2 box) in the current state.
       These are boxes not yet adjacent-2.

    2. For each such box:
       a) Check if (adjacent box locationX) holds for some locationX.
          This likely indicates where the box currently is adjacent to.
       b) Using static knowledge of locations and adjacencies, estimate how many pushes
          are minimally required to move the box to a location that will satisfy (adjacent-2 box).
          However, since the exact target location is not given explicitly, we estimate the minimal
          steps from the current box location to any goal adjacency, assuming adjacency relations as edges.

    3. Count also robot moves needed to get into pushing position:
       Since robot movement between locations is required for pushing boxes, estimate the minimal
       number of robot moves needed to go close to each box to push it onward.
       For efficiency, approximate as one robot move per box pushed (a rough estimate).

    4. Sum over all boxes the estimated pushes plus minimal robot positioning moves.

    5. Return the total as heuristic value.

    Note:
    - The heuristic is not guaranteed to be admissible, but aims to be close enough for guided search.
    - It returns zero if the state satisfies all goal predicates.
    - All operations are designed to be efficient and avoid expensive search inside heuristic.

    """

    def __init__(self, task):
        """
        Initialize heuristic helper data structures from the task's static facts and goals.

        - Extract goal boxes needing (adjacent-2 box)
        - Extract static facts to build:
          * box_to_adjacent_locations: which locations each box is adjacent to (from static facts)
          * box_to_robot_adjacent_locs: which locations robot is adjacent to each box
          * adjacency graph over locations based on (clear location1 location2)
        
        Since the instances don't specify explicit "target" locations,
        heuristic uses adjacency sets and connectivity to estimate push distances.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal boxes that must have adjacent-2 predicate true.
        self.goal_boxes = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.add(parts[1])

        # Map box -> set of locations it is adjacent to (static facts)
        self.box_to_adjacent_locations = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.box_to_adjacent_locations.setdefault(box, set()).add(loc)

        # Map box -> locations where robot is adjacent to box (static facts)
        self.box_to_robot_adjacent_locs = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.box_to_robot_adjacent_locs.setdefault(box, set()).add(loc)

        # Build location adjacency graph from (clear location1 location2) static facts:
        # This undirected graph will be used to estimate robot moves.
        self.location_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "clear" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of actions required to achieve the goal.

        The heuristic estimates:
        - For each box not yet adjacent-2, the minimal number of pushes to place it adjacent-2.
        - For each push, assume robot needs at least one move to position itself.
        The sum of pushes + robot moves is returned as heuristic value.

        Returns 0 if the state satisfies all goal predicates.
        """
        state = node.state

        # Quick check: if goal achieved, heuristic is zero.
        if self.goal_boxes <= {get_parts(f)[1] for f in state if f.startswith("(adjacent-2")}:
            return 0

        # Identify boxes not yet with adjacent-2 in state but must have it (goal boxes).
        boxes_to_push = self.goal_boxes - {get_parts(f)[1] for f in state if f.startswith("(adjacent-2")}

        # For each box, determine current location(s) from state:
        # We try to find:
        # - Locations where (adjacent box loc) holds (box's current adjacency)
        # - Locations where robot is currently at (at-robot-2 loc) to estimate robot distance.
        # We estimate the minimal pushes needed as the shortest path length between some
        # current box adjacencies and any location that would allow adjacent-2 (assumed as the
        # locations in box_to_robot_adjacent_locs).
        #
        # Since exact goal target location is unknown, heuristic uses minimal distance between
        # current box adjacency locations and any location where robot can push from.

        # Extract robot current location(s) from state (predicate "at-robot-2 location")
        robot_locations = {get_parts(f)[1] for f in state if f.startswith("(at-robot-2")}

        total_cost = 0

        for box in boxes_to_push:
            # Get current locations where box is adjacent to.
            current_box_locs = {
                get_parts(f)[2]
                for f in state
                if f.startswith("(adjacent ") and get_parts(f)[1] == box
            }
            if not current_box_locs:
                # Fallback: if no adjacency info in state for box, assume some default,
                # e.g. locations where statically adjacent. If none, continue.
                current_box_locs = self.box_to_adjacent_locations.get(box, set())

            # Candidate push target locations: where robot can push the box to achieve adjacent-2
            goal_adjacent_locs = self.box_to_robot_adjacent_locs.get(box, set())
            if not goal_adjacent_locs:
                # If no robot adjacency info, fallback to current box adjacency.
                goal_adjacent_locs = self.box_to_adjacent_locations.get(box, set())
            if not goal_adjacent_locs:
                # If still empty, assume zero distance (heuristic estimate).
                total_cost += 1  # Count minimal push for safety.
                continue

            # Compute minimal distance between current box location and any goal adjacency location.
            # Use BFS in the location graph.

            def bfs_min_distance(starts, goals):
                from collections import deque

                visited = set()
                queue = deque()
                for s in starts:
                    if s in goals:
                        return 0
                    queue.append((s, 0))
                    visited.add(s)

                while queue:
                    loc, dist = queue.popleft()
                    for neighbor in self.location_graph.get(loc, []):
                        if neighbor in goals:
                            return dist + 1
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                # If unreachable, return a large number (heuristic fallback)
                return 10

            min_pushes = bfs_min_distance(current_box_locs, goal_adjacent_locs)

            # Estimate minimal robot moves:
            # Robot needs to reach one of the box's adjacency locations from current robot location.
            # Take minimal distance from any robot_loc to any current_box_loc.
            if robot_locations and current_box_locs:
                robot_to_box_dist = bfs_min_distance(robot_locations, current_box_locs)
            else:
                robot_to_box_dist = 0  # If no info, assume zero for heuristic.

            # Total cost for this box is approx pushes + robot moves
            # We add +1 to min_pushes to count the push action itself.
            total_cost += max(1, min_pushes) + robot_to_box_dist

            # Assume after pushing this box, robot location moves to box adjacency location,
            # so update robot_locations to box adjacency to approximate next box computations.
            robot_locations = current_box_locs

        return total_cost
