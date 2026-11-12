from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


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


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to push all boxes
    to their goal locations. It combines shortest path distances for pushing
    each box to its goal and an estimate of robot travel cost to reach and push
    boxes efficiently.

    Assumptions
    -----------
    - The heuristic is not necessarily admissible but aims to be efficiently computable
      and informative enough to guide a greedy best-first search.
    - The robot moves on a graph defined by adjacency of locations.
    - Boxes can be pushed only if their front location is clear.
    - The heuristic ignores complex deadlocks or irreversible states.
    - The heuristic returns zero only when the goal is reached.

    Heuristic Initialization
    ------------------------
    - Extract the adjacency graph (edges) between locations from static facts.
    - Extract all box names from goal facts.
    - Extract goal positions of all boxes from the goal facts.
    - Precompute a lookup for adjacency in both directions.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1) Extract the robot location and the boxes' current locations from the state facts.

    2) For each box, compute the minimal cost to push it from its current location to its goal location:
       - This is computed as a shortest path over locations ignoring obstacles (assuming box can be pushed along),
         but considering only adjacency.
       - Distances between locations are computed with BFS on the adjacency graph.
       - The "push distance" between box current and goal position approximates the minimum number of push actions.

    3) Compute approximate distance from the robot to each box to estimate robot travel cost:
       - Robot moves via adjacency edges ignoring boxes (optimistic).
       - Use BFS to find shortest distance from robot to each box.

    4) For each box, combine:
       - Distance for robot to get behind the box to push it (approximate as the distance from robot to box location, or zero if robot is already at box location),
       - Plus push distance of that box from current to goal location.

    5) Sum over all boxes the minimal combined cost as an estimate of total required actions.

    6) Return this sum as the heuristic value.

    7) If the state is a goal state, the heuristic returns zero.

    Note:
    - Movement and pushing cost actions are counted uniformly as 1 per action.
    - The heuristic ignores whether a push is actually executable (push preconditions),
      but uses shortest distances based on adjacency to estimate effort.
    - The heuristic uses BFS multiple times but keeps the neighborhood graph static.

    """

    def __init__(self, task):
        """
        Initialize heuristic by extracting:
        - Adjacency graph extracted from (adjacent l1 l2 dir) static facts, storing neighbors both ways.
        - Goal box positions extracted from goal facts (at box loc).
        """
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency graph as a dict {location: set(neighboring_locations)}
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.adjacent.setdefault(l1, set()).add(l2)
                # The domain defines adjacency both ways explicitly, but just in case:
                self.adjacent.setdefault(l2, set()).add(l1)

        # Extract goal box positions: map box -> goal_location
        self.goal_box_positions = {}
        for fact in self.goals:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                predicate, box, loc = parts[0], parts[1], parts[2]
                if predicate == "at":
                    self.goal_box_positions[box] = loc

        # Store boxes in goal to know which boxes are relevant
        self.goal_boxes = set(self.goal_box_positions.keys())

    def __call__(self, node):
        """
        Compute heuristic estimation for the current state.

        1) Return 0 if goal reached.

        2) Extract robot location, box locations.

        3) Compute push distances box->goal using BFS on adjacency graph ignoring obstacles.

        4) Compute robot distances to boxes.

        5) Estimate total cost as sum of (robot to box + push distance).

        """
        state = node.state

        # Goal check: if all goal facts true, return 0
        if self.goals <= state:
            return 0

        # Extract robot current location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
            # No robot location found, heuristic cannot estimate reliably, return large cost
            return 1000

        # Extract box current locations: map box -> loc
        box_locs = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                if box in self.goal_boxes:
                    box_locs[box] = loc

        # If any goal box missing in state (i.e. not placed anywhere), assume large cost
        if len(box_locs) < len(self.goal_boxes):
            return 1000

        # BFS helper to compute shortest path distance between two locations on the adjacency graph
        def bfs_distance(start, goal, blocked=None):
            # blocked: a set of locations to be considered blocked (ignored)
            if start == goal:
                return 0
            visited = set()
            queue = [(start, 0)]
            visited.add(start)
            while queue:
                current, dist = queue.pop(0)
                for nb in self.adjacent.get(current, []):
                    if blocked and nb in blocked:
                        continue
                    if nb not in visited:
                        if nb == goal:
                            return dist + 1
                        visited.add(nb)
                        queue.append((nb, dist + 1))
            # no path found
            return None

        # To estimate push distance: since box must be pushed from current loc to goal loc,
        # we approximate the push distance as the shortest path length (number of pushes),
        # ignoring robot constraints or blocking boxes.
        # We ignore currently occupied locations by boxes to keep heuristic optimistic.

        # We also compute distance from robot to box location, ignoring boxes (optimistic).

        # Compute robot distance to boxes
        # We treat boxes as obstacles only if desired; here ignoring for efficiency.
        def robot_distance_to(loc):
            if loc == robot_loc:
                return 0
            visited = set()
            queue = [(robot_loc, 0)]
            visited.add(robot_loc)
            while queue:
                current, dist = queue.pop(0)
                for nb in self.adjacent.get(current, []):
                    if nb not in visited:
                        if nb == loc:
                            return dist
                        visited.add(nb)
                        queue.append((nb, dist + 1))
            return None  # unreachable

        total_cost = 0
        # For each box sum cost: robot travel + pushes
        for box, box_pos in box_locs.items():
            goal_pos = self.goal_box_positions[box]

            push_dist = bfs_distance(box_pos, goal_pos)
            if push_dist is None:
                # No path to push box to goal (blocked?), assign large penalty
                push_dist = 100

            rob_dist = robot_distance_to(box_pos)
            if rob_dist is None:
                # Box unreachable by robot
                rob_dist = 100

            # Heuristic contribution:
            # robot must get near box (we use robot->box loc distance as proxy for that),
            # plus the number of pushes needed.
            # Each push requires the robot to be behind the box, so robot travel cost is needed.
            total_cost += rob_dist + push_dist

        # Return the total heuristic estimate
        return total_cost
