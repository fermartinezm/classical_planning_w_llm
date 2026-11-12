from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    Args:
        fact (str): The fact as a string, e.g., "(adjacent box1 location6)".
        args: Pattern elements, e.g. "adjacent", "*", "location6".

    Returns:
        bool: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal by focusing on
    how many boxes are not yet in their goal predicate state `(adjacent-2 ?box)`, i.e., the boxes
    that the robot needs to push into adjacency (presumably the goal configuration).

    The heuristic estimates the cost to reach the goal using:
    - The count of boxes that are not yet adjacent-2 (i.e., not at goal adjacency).
    - The minimal estimated number of moves to bring the robot to relevant positions to push those boxes.
    - The interaction between robot location and boxes that still need moving (moving and pushing).
    This yields an estimate of required push and move actions.

    Assumptions
    -----------
    - The goal is to have all boxes satisfy `(adjacent-2 ?box)`.
    - Robot moves (`move-2`, `push-2`) happen between locations.
    - Pushing a box changes its `(adjacent)` and `(at)` predicates and results in `(adjacent-2 ?box)` at goal.
    - The robot can only push boxes if adjacent and at its location.
    - The clear relations indicate which moves between locations are possible.
    - The heuristic does not guarantee admissibility but tries to be informative and efficiently computable.
    - Distances between locations are approximated by shortest paths computed on the static `clear` graph.

    Heuristic Initialization
    ------------------------
    Upon initialization, the heuristic extracts the following from the static facts:
    - The mapping of `(at-robot ?box ?location)` facts: box initial robot-adjacent locations.
    - The adjacency relations `(adjacent ?box ?location)` for boxes.
    - The connectivity graph over locations from `(clear ?loc1 ?loc2)`.
    - The set of all boxes to track those that have to reach `(adjacent-2 ?box)` goals.
    - The set of goal boxes from the goal facts `(adjacent-2 ?box)`.

    It precomputes shortest-path distances between all locations using Floyd–Warshall algorithm on the clear graph
    (as the location graph could be sparse and small enough for this; otherwise Dijkstra from robot location could suffice).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which boxes have already reached their goal adjacent-2 status (no cost).
    2. For each box not yet in `(adjacent-2 ?box)`:
       - Find:
         a) The box's current location(s) from `(adjacent box location)` facts (boxes adjacent to locations).
         b) The robot's current location from `(at-robot-2 ?loc)`.
         c) The location(s) adjacent to the box and the robot's needed distance to these locations.
         d) The minimal cost consists of:
            - Estimating the minimal robot moves to reach a location adjacent to the box,
              so that it can push the box.
            - Adding 1 step for pushing the box to achieve `(adjacent-2 ?box)`.
    3. Sum across all boxes the heuristic cost estimate obtained above.
    4. Since boxes may block each other, and the heuristic ignores such interactions for efficiency,
       it serves as a rough but informative estimate.
    5. Return 0 if all boxes satisfy goal `(adjacent-2 ?box)`.

    The heuristic thus combines location graph distances for the robot and box adjacency relations to predict required moves and pushes.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract boxes from static facts by looking at either 'at-robot' or 'adjacent' predicates.
        # Also collect all locations and build clear graph for distances.
        self.boxes = set()
        self.locations = set()

        # Also keep track of adjacency (box <-> location)
        self.box_to_adjacent_locs = dict()

        # Parse static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]

            if pred == "at-robot":
                # (at-robot box location)
                box = parts[1]
                self.boxes.add(box)
                loc = parts[2]
                self.locations.add(loc)
                # Note: These are static facts about where the box can be adjacent for robot pushing.
                # But to save from confusion, we do not use here as adjacency is dynamic in the problem.
            elif pred == "adjacent":
                # (adjacent box location)
                box = parts[1]
                loc = parts[2]
                self.boxes.add(box)
                self.locations.add(loc)
                self.box_to_adjacent_locs.setdefault(box, set()).add(loc)
            elif pred == "clear":
                # (clear location1 location2) - locations are connected
                loc1 = parts[1]
                loc2 = parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Also gather goal boxes that need to be adjacent-2:
        self.goal_boxes = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "adjacent-2":
                box = parts[1]
                self.goal_boxes.add(box)

        # Build undirected graph for location connectivity from clear facts
        # We treat the `clear` relations as edges between locations that robot can move along.
        self.adj_graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "clear" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Clear relation might be one-directional, so add edges both ways for path distances:
                self.adj_graph[loc1].add(loc2)
                self.adj_graph[loc2].add(loc1)

        # Precompute shortest path distances between locations using Floyd–Warshall (since number might be small)
        loc_list = list(self.locations)
        self.loc_index = {loc: i for i, loc in enumerate(loc_list)}
        n = len(loc_list)

        # Initialize distance matrix
        INF = 10 ** 9
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for loc in self.adj_graph:
            i = self.loc_index[loc]
            for nbr in self.adj_graph[loc]:
                j = self.loc_index[nbr]
                dist[i][j] = 1

        # Floyd–Warshall
        for k in range(n):
            for i in range(n):
                di = dist[i]
                dik = di[k]
                if dik == INF:
                    continue
                dk = dist[k]
                for j in range(n):
                    val = dik + dk[j]
                    if val < di[j]:
                        di[j] = val

        self.dist = dist
        self.loc_list = loc_list

    def shortest_distance(self, loc1, loc2):
        """Return shortest path distance between two locations (or large number if unreachable)."""
        if loc1 not in self.loc_index or loc2 not in self.loc_index:
            return 10**9
        return self.dist[self.loc_index[loc1]][self.loc_index[loc2]]

    def __call__(self, node):
        state = node.state

        # Fast check for goal: if all goal boxes have adjacent-2 fact true, heuristic is 0.
        goal_reached = True
        for box in self.goal_boxes:
            if f"(adjacent-2 {box})" not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Determine robot current location: (at-robot-2 ?loc)
        robot_location = None
        for fact in state:
            if fact.startswith("(at-robot-2 "):
                parts = get_parts(fact)
                if len(parts) == 2:
                    robot_location = parts[1]
                    break
        if robot_location is None:
            # No known robot location: large heuristic penalty (unlikely)
            return 10**9

        total_cost = 0

        # For each box in goal but not yet adjacent-2: estimate cost to push
        for box in self.goal_boxes:
            if f"(adjacent-2 {box})" in state:
                # Already placed correctly
                continue

            # Find current adjacency/box position from state facts: (adjacent box location)
            # If multiple, pick one minimal cost; if none, assume large cost.
            box_adjacent_locs = []
            for fact in state:
                if fact.startswith("(adjacent "):
                    parts = get_parts(fact)
                    if len(parts) == 3 and parts[1] == box:
                        box_adjacent_locs.append(parts[2])

            if not box_adjacent_locs:
                # Box adjacency unknown: heuristic penalty
                total_cost += 5  # Arbitrary penalty for unknown position
                continue

            # For each adjacency location of the box, compute cost for robot to get there + 1 push
            # (pushing action needed)
            min_cost_for_box = 10**9
            for box_loc in box_adjacent_locs:
                # Distance for the robot to reach the box location to push it
                dist_robot_to_box_loc = self.shortest_distance(robot_location, box_loc)
                # 1 for pushing the box to achieve adjacent-2 (goal)
                cost = dist_robot_to_box_loc + 1
                if cost < min_cost_for_box:
                    min_cost_for_box = cost

            total_cost += min_cost_for_box

        # Provide a heuristic estimate of total actions needed (moves + pushes)
        # Must be finite for solvable states, and 0 for goal states.
        # Return the computed estimate.
        return total_cost
