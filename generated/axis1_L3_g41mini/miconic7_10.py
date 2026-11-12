from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(adjacent box1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the "secret_domain" sokoban-like domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve the goal condition that all boxes
    are in the "adjacent-2" relation (i.e., pushed to their final positions) from the current state.
    It does so by combining knowledge of:
    - Which boxes are already in goal adjacency (adjacent-2),
    - Which boxes are not yet adjacent-2 but have an "adjacent" location (a target),
    - The estimated cost to bring each box to adjacency-2 by pushing and moving the robot efficiently.

    Assumptions
    -----------
    - The robot moves on locations (tracked by facts "at-robot-2 <loc>").
    - Each box can be pushed by the robot from an adjacent location.
    - A box is considered placed when it satisfies "(adjacent-2 box)" predicate.
    - Movement operators move the robot between locations via "clear" facts.
    - Push operations are needed to change boxes from their initial "adjacent" positions to "adjacent-2".
    - The heuristic does not need to be admissible but should guide the search effectively.

    Heuristic Initialization
    ------------------------
    - Extract all boxes and locations from static facts sets.
    - Map each box to its currently associated adjacent location (from facts "adjacent box loc") from the static facts.
      This acts as a candidate target location for that box.
    - Extract adjacency between locations via "clear loc1 loc2" facts to estimate robot travel cost.
    - Precompute which boxes appear in the goals as needing "(adjacent-2 box)".
    - The heuristic uses these to estimate the robot travel and push costs on-demand.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:

    1. Check if the goal is reached (all boxes have "(adjacent-2 box)" true). If so, return 0.

    2. Identify the current robot location from "(at-robot-2 loc)" fact.

    3. Identify which boxes are already at their goal adjacency (adjacent-2) and which are not.

    4. For each box not yet "adjacent-2":
       - Find its target adjacent location from the static "adjacent box loc" mapping.
       - If the box is currently "at" (the "at box" fact) meaning the box is at the robot's current location,
         then estimate pushing cost as 1 (one push action to reach adjacency-2).
       - Otherwise, estimate:
         a) The cost for the robot to reach the box position (from current robot location to box "at" location).
         b) One push action to push the box to adjacency-2.
       - Sum these costs to get a per-box estimate.

    5. Sum over all boxes not yet adjacent-2 the estimated costs from step 4.

    Notes on robot travel cost:
    - Movement between locations requires moving the robot "at-robot-2 loc" using "move-2" or "push-2".
    - Because we only have "clear loc1 loc2" facts describing connectivity, we perform a simple breadth-first search 
      over locations to estimate minimal moves required for the robot to reach a target location.

    This approach provides a cost estimate combining:
    - Robot distance to boxes,
    - Number of push actions (one per box to get from "at" to "adjacent-2"),
    - Thus guiding the search by how far robot must travel and how many boxes remain to be pushed.

    It is efficient because:
    - It caches adjacency of locations from static facts.
    - It uses simple BFS for distance estimates.
    - It considers per-box minimal cost without complex global optimal matching.

    The heuristic value is always finite (since the domain connectivity is assumed) and 0 iff goals reached.
    """

    def __init__(self, task):
        """
        Initialize the heuristic data structures from static facts and goal conditions.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all boxes mentioned in goals as requiring (adjacent-2 box)
        self.goal_boxes = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.add(parts[1])

        # Extract location adjacency graph from (clear loc1 loc2) facts (symmetric)
        self.location_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "clear" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Extract for each box its target adjacent location from static facts: (adjacent box location)
        self.box_target_adjacent_location = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Only keep target locations for boxes that appear in goal_boxes
                if box in self.goal_boxes:
                    self.box_target_adjacent_location[box] = loc

    def bfs_distance(self, start, goal):
        """
        Compute shortest distance between two locations in the location graph using BFS.
        If unreachable, returns a large number (e.g., 1000).
        """
        if start == goal:
            return 0
        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            neighbors = self.location_graph.get(current, ())
            for n in neighbors:
                if n == goal:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    frontier.append((n, dist + 1))
        return 1000  # Large number if unreachable (should normally not happen)

    def __call__(self, node):
        state = node.state

        # If the goal facts are all present in the state, heuristic is zero
        if self.goals <= state:
            return 0

        # Find robot location from (at-robot-2 loc)
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot-2" and len(parts) == 2:
                robot_loc = parts[1]
                break
        if robot_loc is None:
            # If no robot location found, assume heavily penalizing cost
            return 10000

        # For each box, check if adjacent-2 goal reached
        # We consider only boxes in the goals
        # Track boxes currently adjacent-2 (at goal), and those that are not
        boxes_adjacent2 = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                boxes_adjacent2.add(parts[1])

        boxes_to_place = self.goal_boxes - boxes_adjacent2
        if not boxes_to_place:
            return 0  # All boxes placed

        # For each box, find its location from:
        # - (at box) implies box is at robot location
        # - (at-robot box loc) means robot is at loc pushing box? Actually "at-robot box loc" static means box starts at loc
        # We use "at box" fact to indicate box currently pushed by robot, else box is at location adjacent from static facts.

        # Extract box current location as follows:
        # If "(at box)" in state: box occupies the robot's current location (since "at" has 1 param in domain)
        # Otherwise, we only have static knowledge where the box is adjacent to a location (from static "(adjacent box loc)")
        # To be more precise, since domain has (at ?v1 - box), it's a fact with only box param, so box is "at" robot or pushed?

        # We interpret "(at box)" means box is currently adjacent or carried by robot.
        # For heuristic, assume:

        heuristic_cost = 0
        push_cost_per_box = 1  # one push action needed to get from "at" to "adjacent-2" state

        # Preprocess which boxes have "at" fact in current state
        boxes_at_robot = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 2:
                boxes_at_robot.add(parts[1])

        # For each box to place estimate cost:
        for box in boxes_to_place:
            # Target adjacent location for the box
            target_adj_loc = self.box_target_adjacent_location.get(box)
            if target_adj_loc is None:
                # Defensive: no target adjacency known, assign a high cost
                heuristic_cost += 100
                continue

            # Estimate cost to reach box position and push it
            if box in boxes_at_robot:
                # Box is currently at robot location (being pushed?), cost only 1 push needed
                heuristic_cost += push_cost_per_box
            else:
                # Box not at robot currently:
                # Find box location (from static adjacent facts)
                # Several adjacent facts for box can exist, we consider one from static facts is target_adj_loc;
                # The box may be initially at another location, but for heuristics we approximate location by target adjacent location

                # Estimate robot travel cost from robot_loc to target_adj_loc (the box location)
                dist = self.bfs_distance(robot_loc, target_adj_loc)
                # Add 1 push action to push box from that location to adjacent-2
                heuristic_cost += dist + push_cost_per_box

        return heuristic_cost
