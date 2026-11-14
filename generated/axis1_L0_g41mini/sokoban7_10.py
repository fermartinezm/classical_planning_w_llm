from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Summary:
    ----------
    This heuristic estimates the number of actions (moves and pushes) necessary 
    to reach the goal from the current state. It is designed to guide greedy
    best-first search efficiently without guaranteeing admissibility.

    Assumptions:
    ------------
    - The robot can move orthogonally on a grid defined by locations connected by adjacency.
    - Boxes need to be pushed onto specified goal locations.
    - The robot can only push one box at a time.
    - Clear locations can be occupied by the robot or boxes accordingly.
    - The heuristic focuses on distances ignoring possible deadlocks or complex tunnel traps,
      providing a fast estimate that greatly reduces expanded nodes.
    - Multiple boxes and goals can exist, but boxes are distinct and must be paired to their goals.

    Heuristic Initialization:
    --------------------------
    - Extract all location adjacencies (bidirectional) to build a graph representation of the map.
    - Extract box goal locations from the task goals (we assume goals only specify (at box loc)).
    - Precompute location neighbors for quick distance computations.
    - Use BFS on the graph to compute shortest distances.
    - Extract all box objects and goal locations from the goals and static facts.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify the current robot location.
    2. Identify current box locations and goal locations.
    3. For each box, compute the shortest distance from its current position to any goal 
       location assigned to that box in the goals.
       - We assume box-to-goal matching by name (box1 → goal for box1).
    4. Compute the shortest distance from robot location to each box location.
    5. For each box, estimate cost components:
       - Cost to reach the box by the robot (at least distance robot -> box).
       - Cost to push the box to its goal (at least distance box -> goal).
    6. Sum the individual box costs to obtain the heuristic value.
    7. Since robot must move between boxes, add a minimal overhead for traveling between boxes.
       To keep it efficient, just add the total robot-to-box distances (not minimal TSP).
    8. If the current state is a goal state, return 0.
    9. The heuristic value is always finite for solvable states as distances exist over the graph.

    Implementation notes:
    ---------------------
    - Distances between locations are computed using BFS on the adjacency graph.
    - If a box is already on its goal, its push cost is zero.
    - The heuristic ignores possible deadlocks or blocked moves, hence is not admissible.
    """

    def __init__(self, task):
        self.goals = task.goals  # set of goal facts
        self.static = task.static  # static facts (e.g., adjacency)
        self.facts = task.facts

        # Parse adjacency relations to build an undirected graph {loc: set(neighbor_locs)}
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _dir = get_parts(fact)
                # add bidirectional edges
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

        # Extract goal box locations: dict box -> goal location
        # The goals are facts like (at box loc)
        self.box_goal_locations = {}
        for fact in self.goals:
            if match(fact, "at", "*", "*"):
                _, box, goal_loc = get_parts(fact)
                self.box_goal_locations[box] = goal_loc

        # Extract all box objects from goals (we rely on boxes must be in goals)
        self.boxes = set(self.box_goal_locations.keys())

    def bfs_distance(self, start, targets):
        """
        BFS on adjacency graph to find shortest distance from start to any of targets.
        Returns the minimum distance or None if no path found.
        """
        if start in targets:
            return 0
        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in self.adjacency.get(current, ()):
                if neighbor in visited:
                    continue
                if neighbor in targets:
                    return dist + 1
                visited.add(neighbor)
                frontier.append((neighbor, dist + 1))
        return None  # no path found

    def __call__(self, node):
        state = node.state

        # 1) Check if goal reached
        if self.goals <= state:
            return 0

        # 2) Extract robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
            # defensive fallback, robot location unknown
            return 0

        # 3) Extract box locations from the state
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, box_loc = get_parts(fact)
                # only consider boxes we have goals for
                if box in self.boxes:
                    box_locations[box] = box_loc

        # Defensive: if some boxes are missing, estimate conservatively (skip here)
        if len(box_locations) < len(self.boxes):
            # some box locations missing: cannot estimate accurately
            # return a large heuristic estimate
            return 1000

        total_cost = 0

        # 4) Compute push cost per box: distance from box_loc to its goal.
        for box, box_goal in self.box_goal_locations.items():
            box_loc = box_locations.get(box)
            if box_loc is None:
                # missing box location: penalty
                total_cost += 100
                continue

            if box_loc == box_goal:
                push_cost = 0
            else:
                dist = self.bfs_distance(box_loc, {box_goal})
                if dist is None:
                    # unreachable goal, assign high cost
                    push_cost = 1000
                else:
                    push_cost = dist

            total_cost += push_cost

        # 5) Add cost for robot traveling to boxes to push them.
        # We estimate the sum of distances robot->box for all boxes not on goal.
        robot_to_boxes_dist = 0
        for box, box_goal in self.box_goal_locations.items():
            box_loc = box_locations.get(box)
            if box_loc is None:
                continue
            if box_loc != box_goal:
                dist = self.bfs_distance(robot_loc, {box_loc})
                if dist is None:
                    # unreachable box location, high cost
                    robot_to_boxes_dist += 1000
                else:
                    robot_to_boxes_dist += dist

        total_cost += robot_to_boxes_dist

        # 6) Return heuristic estimate
        return total_cost
