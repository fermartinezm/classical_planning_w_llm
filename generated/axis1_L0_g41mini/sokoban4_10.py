from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the minimum number of push and move actions
    required to reach a goal state from the current state. It guides a greedy best-first search by
    providing informed estimations that balance precision and computational efficiency.

    Assumptions:
    - The heuristic does not need to be admissible.
    - Moving the robot closer to boxes or target goals has cost 1 per step.
    - Pushing a box closer to its goal advances the state.
    - The robot can move freely around clear locations.
    - The heuristic is zero only when all boxes are at their goal locations.
    - Actions are either 'move' or 'push', and pushing a box requires the robot to be adjacent behind the box.
    - We ignore deadlocks or obstacles aside from occupied locations.
    - The static facts (adjacents) and dynamics (clear locations) are consistent and allow shortest path computations.

    Heuristic Initialization:
    - Extract all adjacency relations from static facts and build a graph of location adjacencies.
    - Extract goal box locations from the task goals.
    - Prepare helper data structures for fast location lookup and pathfinding.
    - Store boxes and robot locations from the current state when called.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract the robot's current location.
    2. Extract each box's current location.
    3. Extract each box's goal location (from goals).
    4. For each box, compute the shortest distance (number of moves) to its goal location.
    5. For the robot, compute the shortest distances to each box location and to each box's pushing position(s):
       - A pushing position is the location behind the box opposite to the direction of push.
    6. Estimate the heuristic cost as the sum over all boxes of:
       - The minimal cost to push that box from its current location to the goal location,
         approximated by the box-to-goal distance plus minimal robot travel distance to reach a pushing position
         for that box.
    7. The heuristic is zero if all boxes are already at their goal locations.
    8. Use efficient BFS shortest path searches over the adjacency graph for distance computations.
    9. The heuristic accounts for moving the robot into position to push boxes and for pushing boxes to goals,
       ignoring complex deadlocks and while assuming open paths.
    10. This heuristic balances accuracy (considering actual locations and adjacency) and efficiency,
        improving guidance over trivial distance heuristics.

    """

    def __init__(self, task):
        # Extract adjacency relations from static facts to build the connectivity graph.
        self.adjacent_map = {}  # loc -> list of (neighbor_loc, direction)
        for fact in task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc_from = parts[1]
                loc_to = parts[2]
                direction = parts[3]
                self.adjacent_map.setdefault(loc_from, []).append((loc_to, direction))
        # Extract box goal locations from task goals.
        self.box_goals = {}
        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box = parts[1]
                goal_loc = parts[2]
                self.box_goals[box] = goal_loc

    def __call__(self, node):
        state = node.state

        # Quick check: if goal reached, heuristic is zero.
        if all(fact in state for fact in node.task.goals):
            return 0

        # Extract robot location from state.
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
            # Defensive fallback: if no robot location found, return high heuristic.
            return 9999

        # Extract boxes and their locations from state.
        boxes = {}  # box -> location
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box = parts[1]
                loc = parts[2]
                boxes[box] = loc

        # If any box missing from goals (should not happen), ignore it.
        box_goal_pairs = []
        for box, goal_loc in self.box_goals.items():
            if box in boxes:
                box_goal_pairs.append((box, boxes[box], goal_loc))

        # Build a quick undirected graph of adjacency ignoring direction (for shortest path).
        graph = {}
        for loc_from, neighbors in self.adjacent_map.items():
            graph.setdefault(loc_from, set())
            for loc_to, _dir in neighbors:
                graph.setdefault(loc_to, set())
                graph[loc_from].add(loc_to)
                graph[loc_to].add(loc_from)

        def bfs_distance(start, goals, blocked_locations=None):
            """
            Compute shortest distance from start to any of the goal locations using BFS.
            - `blocked_locations` is a set of locations considered blocked (cannot traverse through).
            Returns int distance or None if unreachable.
            """
            if blocked_locations is None:
                blocked_locations = set()
            from collections import deque

            if start in goals:
                return 0
            visited = {start}
            queue = deque([(start, 0)])
            while queue:
                current, dist = queue.popleft()
                for neighbor in graph.get(current, []):
                    if neighbor in blocked_locations or neighbor in visited:
                        continue
                    if neighbor in goals:
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
            return None

        # Heuristic estimation sum
        heuristic_sum = 0

        # We treat locations occupied by boxes (except the one we want to move) as blocked for robot movement.
        box_positions = set(loc for _, loc, _ in box_goal_pairs)

        for box, box_loc, goal_loc in box_goal_pairs:
            if box_loc == goal_loc:
                # Box already at goal requires no cost
                continue

            # Distance from box to goal (minimal pushes count)
            dist_box_to_goal = bfs_distance(box_loc, {goal_loc})
            if dist_box_to_goal is None:
                # No path to goal for box, large heuristic penalty
                dist_box_to_goal = 9999

            # For robot movement to pushing position, 
            # the robot must get to a location adjacent to the box, opposite to the push direction.
            # We approximate this by considering the neighbors of the box that are adjacent
            # and for which the box can be pushed into the direction opposite to the robot.

            # Find all possible pushing positions around this box,
            # i.e., locations adjacent to box such that the opposite location is free to push box into.
            pushing_positions = []

            for loc_adj, dir_to_box in self.adjacent_map.get(box_loc, []):
                # Robot would stand at loc_adj and push box from box_loc into the next cell opposite to loc_adj
                # We find where the box would move if pushed from box_loc towards goal along the same direction:

                # Find the location opposite to loc_adj w.r.t. box_loc (i.e. the cell box moves into)
                # According to domain: push needs adjacent loc to box (robot loc), 
                # the box moves forward in same direction
                # For direction from loc_adj->box_loc, box moves from box_loc to floc:
                # We need the static fact adjacent box_loc floc dir_to_box
                # So check neighbors of box_loc on direction dir_to_box to find floc:

                # Get the neighbor of box_loc at direction dir_to_box
                floc_candidates = [loc for loc, d in self.adjacent_map.get(box_loc, []) if d == dir_to_box]
                if not floc_candidates:
                    continue
                floc = floc_candidates[0]

                # The pushing position is loc_adj if floc is clear (not occupied by box)
                # but in heuristic we do not have full info of clear, so we consider box locations blocked.
                if floc in box_positions and floc != box_loc:
                    # Box cannot be pushed into occupied location
                    continue

                pushing_positions.append(loc_adj)

            if not pushing_positions:
                # No pushing positions found (likely deadlock or corner), large penalty
                min_robot_to_push_dist = 9999
            else:
                # Compute minimal robot distance to any pushing position
                # Robot cannot stand on boxes positions
                blocked_for_robot = box_positions.copy()
                # Robot can stand on box_loc only if it is that box's position (pushing that box)
                blocked_for_robot.discard(box_loc)

                min_robot_to_push_dist = bfs_distance(robot_loc, set(pushing_positions), blocked_locations=blocked_for_robot)
                if min_robot_to_push_dist is None:
                    # No path for robot to pushing position
                    min_robot_to_push_dist = 9999

            # Total cost estimate for this box is robot moving to pushing position + number of pushes
            # We add a small overhead of 1 per box to model actual pushing plus moves.
            heuristic_sum += min_robot_to_push_dist + dist_box_to_goal

        return heuristic_sum
