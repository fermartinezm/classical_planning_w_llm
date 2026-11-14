from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the number of actions required to push all boxes
    into their goal locations, guiding a greedy best-first search efficiently.
    It provides a finite estimate for all solvable states and is zero only at goal states.

    Assumptions:
    - The robot must push all boxes onto their goal locations.
    - The heuristic ignores precise robot navigation and potential deadlocks,
      but is improved by estimating minimal box-to-goal distances using adjacency.
    - Moving the robot around to push a box is approximated by shortest path costs on the location graph.
    - Each box's pushing cost is roughly the minimal number of pushes (one per move of the box)
      plus moves for positioning the robot adjacent to the box.
    - The heuristic is non-admissible but efficient and effective in reducing expanded nodes.

    Heuristic Initialization:
    - Extract from static facts the adjacency graph of locations.
    - Precompute for each location a list of adjacent reachable locations.
    - Extract the goal locations of all boxes from the goal conditions.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to find:
       - The robot's current location.
       - Each box's current location.
       - Which locations are clear (unoccupied).
    2. For each box not on its goal:
       - Compute (approximate) shortest distances:
         a) From the box's location to each goal location.
         b) From the robot's location (or clear locations adjacent to the box) to the box location.
       - Estimate minimal pushing distance as:
         push_cost = minimal steps to push box to a goal (using shortest path over locations)
       - Estimate robot positioning cost as:
         min_robot_path = shortest path for the robot to a location adjacent to the box,
                         considering only clear locations (the robot must be adjacent to push)
    3. Sum these estimates for all boxes.
    4. Return zero if all boxes are on their goal locations.
    5. The heuristic value is the sum of these combined costs over all boxes.

    Implementation Details:
    - A breadth-first search (BFS) is used to compute shortest distances.
    - This BFS reuses adjacency and clear information from the current state and static facts.
    - The heuristic minimizes node expansion by guiding robot movements closer to boxes,
      and boxes closer to their targets.

    Note:
    The heuristic may underestimate the true cost in complex deadlock configurations,
    but improves search efficiency in practical Sokoban instances.
    """

    def __init__(self, task):
        """
        Initialize heuristic: extract adjacency graph and goal box locations.
        """
        self.goals = task.goals  # goal facts (e.g., (at box loc_goal))
        self.static = task.static  # static facts including adjacency

        # Build adjacency graph: dict {location: set(adjacent_locations)}
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _dir = get_parts(fact)
                self.adjacency.setdefault(loc1, set()).add(loc2)
                # Note: adjacency in sokoban is bidirectional but given both directions as facts.

        # Extract goal locations per box from the goal facts for quick membership
        self.box_goals = dict()
        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                _, box, loc = get_parts(goal_fact)
                self.box_goals[box] = loc

    def _bfs_distances(self, start, valid_locations):
        """
        BFS to compute shortest distances from start location to all locations in valid_locations.
        Returns a dict: loc -> distance (int).
        Only locations in valid_locations are traversed.
        """
        from collections import deque

        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            current_dist = dist[current]
            for neighbor in self.adjacency.get(current, []):
                if neighbor in valid_locations and neighbor not in dist:
                    dist[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return dist

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.
        """
        state = node.state

        # Quick check if goal reached: heuristic zero if all goals satisfied
        if self.goals <= state:
            return 0

        # Parse current robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        # Parse box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        # Parse clear locations
        clear_locs = set()
        for fact in state:
            if match(fact, "clear", "*"):
                clear_locs.add(get_parts(fact)[1])

        # The robot location is clear, so add it as well to clear places where robot can go
        # (robot occupies that location and can stand there)
        clear_locs.add(robot_loc)

        total_cost = 0

        # Precompute BFS from robot location over clear locations to get dist to all possible places
        robot_reachable_dist = self._bfs_distances(robot_loc, clear_locs)

        for box, box_loc in box_locations.items():
            goal_loc = self.box_goals[box]

            # If box already at goal, no cost for this box
            if box_loc == goal_loc:
                continue

            # Compute shortest pushing distance: min distance box -> goal on all locations (no blocking)
            # Here we approximate by shortest path on location graph (ignoring obstacles)
            # Using BFS over all locations ignoring clears is optimistic but fast.
            # To be more precise, we could limit pushing paths considering boxes as obstacles,
            # but for efficiency this heuristic accepts optimistic estimate.

            # We do BFS from box location over all locations reachable via adjacency:
            # We consider all locations as reachable here for pushing box.

            # For pushing: we consider the full set of locations in adjacency keys.
            all_locations = set(self.adjacency.keys())
            box_to_goal_distances = self._bfs_distances(box_loc, all_locations)
            dist_box_to_goal = box_to_goal_distances.get(goal_loc, None)
            if dist_box_to_goal is None:
                # No path to goal from box_loc (should not happen in well-formed instances)
                dist_box_to_goal = 1000  # Large number

            # Now estimate robot cost to push:
            # Robot must be adjacent to the box to push it.
            # Find robot distance to any location adjacent to box_loc that is clear (robot can stand).
            adjacents = self.adjacency.get(box_loc, set())
            adj_clear = adjacents.intersection(clear_locs)
            if not adj_clear:
                # No place adjacent to box to stand, unreachable -> large cost
                min_robot_dist = 1000
            else:
                # Get minimum distance from robot location to some adjacent location of the box that is clear
                distances = [robot_reachable_dist.get(loc, 1000) for loc in adj_clear]
                min_robot_dist = min(distances)

            # Heuristic for this box: pushing cost + robot cost to get adjacent for pushing
            # Since one push moves box one step closer, dist_box_to_goal approximates pushes.
            # Robot moves estimated by min_robot_dist.

            # We give weight 1 per push, plus 1 per robot move - this can be tuned.
            box_cost = dist_box_to_goal + min_robot_dist

            total_cost += box_cost

        return total_cost
