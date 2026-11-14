from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(adjacent box1 location6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (Sokoban) domain.

    Summary:
    ----------
    This heuristic estimates the minimal number of actions required to push all boxes
    into adjacency with locations (i.e., reach each box's goal predicate `adjacent-2`).
    It captures the main costs:
       - Moving the robot close to boxes to prepare push actions.
       - Pushing boxes until they reach the goal adjacency condition.
    The heuristic is not admissible, but efficiently computable and guides greedy best-first search.

    Assumptions:
    ------------
    - Each box must achieve the goal predicate `(adjacent-2 box)` to be considered placed correctly.
    - The robot can only push boxes if it is adjacent to the box's location (robot "at-robot-2" locations).
    - The robot's and boxes' initial and goal locations are embedded in the predicates:
      `(adjacent box location)` indicates a box is adjacent to a location,
      `(at-robot-2 location)` indicates robot’s current location.
    - The static facts `(at-robot box location)` provide the fixed position a box is at relative to the robot,
      and `(clear loc1 loc2)` provides connectivity for robot moves.
    - The domain requires the robot to move and push boxes iteratively until all boxes satisfy `(adjacent-2 box)`.
    - No deadlocks or immovable situations are assumed; all boxes can eventually reach the goal.

    Heuristic Initialization:
    -------------------------
    - Extract the mapping from each box to its fixed "at-robot" location (its current position related to the robot).
    - Build a graph of locations and their clear relations to model feasible robot moves.
    - Extract the robot's current location `(at-robot-2 loc)`.
    - Keep track of all boxes and their current `(adjacent box location)` relations.
    - Extract the set of goal boxes (those not yet `(adjacent-2 box)`).
    - Precompute adjacency relations between locations to estimate shortest paths approximately by BFS.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify boxes not yet at goal adjacency (not `(adjacent-2 box)`).
    2. For each such box:
       a) Identify the box’s current location (from `(adjacent box location)`).
       b) Compute the shortest robot path distance from current robot location to the box’s adjacent location,
          using the robot movement graph (locations connected by `clear`).
       c) Approximate the number of push actions to move the box from its current `adjacent` location to 
          the goal adjacency state. Since the domain does not specify explicit goal locations, we assume 
          1 push action per box needed as minimal effort heuristic.
    3. Sum over all boxes:
       - Distance for the robot to reach the box.
       - An estimated minimal 1 push to get the box from its current adjacent location to `(adjacent-2 box)`.
    4. Return the total sum as the heuristic estimate.
    5. If the state already satisfies all goal predicates, return 0.

    Notes:
    ------
    - The heuristic uses approximate shortest path length between locations by BFS over static `clear` facts.
      This is efficient and reflects costs of robot movement to boxes.
    - The heuristic is not admissible because it assumes minimal pushes per box without detailed path planning.
    - The heuristic is finite for all solvable states.
    """

    def __init__(self, task):
        """
        Initialize heuristic by extracting relevant static facts and goal information.

        Args:
        -----
        task: Task object representing the planning problem, containing static facts, operators, goals.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract clear edges as a bidirectional graph for robot movement:
        # From facts (clear loc1 loc2), robot can move-2 between loc1 <-> loc2.
        self.robot_graph = dict()  # location -> set(connected_locations)
        for fact in self.static:
            if match(fact, "clear", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.robot_graph.setdefault(loc1, set()).add(loc2)
                self.robot_graph.setdefault(loc2, set()).add(loc1)

        # Extract fixed box positions relative to the robot from static facts:
        # (at-robot box location) means box is associated with location.
        self.box_to_location = {}
        for fact in self.static:
            if match(fact, "at-robot", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                self.box_to_location[box] = loc

        # Extract goal boxes from goal conditions: they must satisfy (adjacent-2 box)
        self.goal_boxes = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "adjacent-2":
                self.goal_boxes.add(parts[1])

    def _bfs_dist(self, start, goal):
        """
        Compute shortest distance between two robot locations via BFS in the robot connectivity graph.

        Args:
        -----
        start: str, starting location name
        goal: str, goal location name

        Returns:
        --------
        int: number of steps between start and goal, or large number if unreachable.
        """
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.robot_graph.get(current, set()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # unreachable implies large cost
        return 1000

    def __call__(self, node):
        """
        Compute the heuristic estimate of required actions to satisfy all box adjacency goals.

        Args:
        -----
        node: Node object containing current state.

        Returns:
        --------
        int: heuristic estimate of required actions to reach goal state.
        """
        state = node.state

        # If goal reached, heuristic is zero.
        # Goal requires `(adjacent-2 box)` for all goal boxes.
        if self.goal_boxes <= state:
            return 0

        # Extract current robot location from `(at-robot-2 loc)`
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot-2", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
            # Defensive: if unknown, assign large heuristic
            return 10000

        # Extract boxes currently adjacent to a location from `(adjacent box location)`
        # This gives box current adjacency locations.
        box_adjacent_locs = {}
        for fact in state:
            if match(fact, "adjacent", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                box_adjacent_locs[box] = loc

        # Extract boxes currently at goal (those with `(adjacent-2 box)`)
        boxes_at_goal = set()
        for fact in state:
            if match(fact, "adjacent-2", "*"):
                boxes_at_goal.add(get_parts(fact)[1])

        # Consider only boxes not yet at goal adjacency
        boxes_to_place = self.goal_boxes - boxes_at_goal

        total_cost = 0

        for box in boxes_to_place:
            # Current adjacency location of the box
            box_loc = box_adjacent_locs.get(box)

            if box_loc is None:
                # If unknown adjacency, assume max cost to find and push
                # Because box might be "at box" only without adjacency info
                total_cost += 10
                continue

            # Estimate robot travel cost from current robot location to box's adjacency location
            dist_to_box = self._bfs_dist(robot_loc, box_loc)

            # Minimal push cost assumed to be 1 per box (at least one push needed)
            push_cost = 1

            # Sum cost for this box: moving robot next to box + one push.
            total_cost += dist_to_box + push_cost

            # Update robot location approximately to box_loc because probably robot will
            # move there after pushing this box (heuristic approximation).
            robot_loc = box_loc

        return total_cost
