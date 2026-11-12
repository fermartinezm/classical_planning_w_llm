from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact string by removing parentheses 
    and splitting the string.

    Example:
    "(at box1 loc_3_5)" -> ["at", "box1", "loc_3_5"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    Args:
        fact: Complete fact as a string, e.g., "(at box1 loc_3_5)".
        args: Pattern strings to match with wildcards '*' allowed.

    Returns:
        True if `fact` matches pattern `args`, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the number of actions required to push all boxes 
    to their goal locations and then move the robot to the position necessary 
    for the last push(es). The heuristic is not guaranteed to be admissible but aims 
    to provide an informed and computationally efficient estimate to guide the greedy 
    best-first search effectively.

    Assumptions:
    - The domain consists of a grid-like map represented by locations with adjacency.
    - Each box must be pushed onto a unique goal location.
    - The robot can only move onto clear locations and push boxes into clear locations.
    - Heuristic computation focuses on minimal "push distance" from boxes 
      to their goal locations, plus the estimated cost for the robot to reach 
      the required pushing positions.
    - Robot movements without pushing count as 1 action; pushes count as 1 action.

    Heuristic Initialization:
    - Extract static information about location adjacency as a graph.
    - Extract goal box locations from the goals (facts of the form `(at box location)`).
    - Build adjacency data structures from static facts for fast shortest path computations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse current state:
       - Identify the robot's current location.
       - Identify every box's current location.
    2. Identify all goal locations from the goal conditions (for boxes).
    3. For each box, compute:
       a) Minimal push distance to its closest unmatched goal location. 
          - The heuristic assumes the box must be pushed there by navigating the robot 
            around it to the valid pushing position.
       b) This push distance accounts for:
          - Distance from the box to the goal.
          - Distance for the robot to get behind the box for the push.
    4. Assign boxes to goals minimizing the sum of these minimal push distances 
       via a greedy matching (not necessarily optimal, but efficient).
    5. Additionally, compute the minimal distance for the robot to reach pushing positions 
       for the boxes to start pushing.
    6. Sum these distances as the heuristic value.
    7. Return 0 if the state is a goal state.

    Notes:
    - The heuristic uses BFS for shortest path computations on the adjacency graph.
    - Push distances are approximated by Manhattan-like movement on the graph, 
      considering that pushing requires the robot to be behind the box in the 
      opposite direction of the push.
    - This heuristic aims to reduce expanded nodes by guiding the search toward 
      pushing boxes closer to their goals with minimal robot walks.
    """

    def __init__(self, task):
        # Extract static adjacency (undirected) map from static facts.
        # Store it as a dict: {location: {neighbor_location,...}}
        self.adjacency = {}
        for fact in task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = set()
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = set()
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)

        # Extract goal locations for each box from the goals
        self.goal_locations = {}
        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

        # Directions with relative offsets used for computing pushing positions
        # though coordinates are not explicitly given, we simulate direction vectors
        # for adjacency by direction symbol in PDDL:
        # The domain uses directions "up", "down", "left", "right".
        self.directions = ("up", "down", "left", "right")

        # We'll map directions to reverse directions for pushing:
        self.reverse_dir = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left"
        }

    def _bfs_distances(self, start, blocked):
        """
        Compute shortest distances from start to all reachable locations
        ignoring a set of blocked locations.

        Args:
            start: Location from which distances are computed.
            blocked: A set of locations that are considered blocked.

        Returns:
            A dict mapping location -> distance (int).
        """
        from collections import deque

        distances = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in distances and neighbor not in blocked:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def _robot_to_push_pos_dist(self, robot_loc, box_loc, blocked):
        """
        Given robot location and box location, compute minimal distance for the robot 
        to get into a pushing position for the box.

        The robot must occupy a position adjacent to the box in the direction opposite
        to the target push direction.

        Args:
            robot_loc: current robot location.
            box_loc: current box location.
            blocked: set of blocked locations (including other boxes).

        Returns:
            The minimal distance (int) the robot must walk to a pushing position, or a large number if none found.
        """
        # For each direction, check if position opposite to that direction from box_loc is reachable by robot
        # and the box can be pushed in that direction (the destination behind the box is clear).
        min_distance = float("inf")
        # Precompute distances from robot, excluding blocked locations.
        robot_distances = self._bfs_distances(robot_loc, blocked)

        # For each adjacent location to box that is candidate pushing destination:
        # We need to find a location behind box (in reverse direction) the robot can stand on.

        # We'll test each adjacent location of box and see if it's clear (for box's pushed position),
        # and if there's a free location behind the box for the robot.

        for direction in self.reverse_dir:
            # Find candidate robot pushing position:
            for fact in self.adjacency.get(box_loc, []):
                # Here, to identify the direction from box_loc to fact (adjacent location),
                # we check the "adjacent" facts from static in direction-aware form.
                # However, reversely matching adjacency with directions is complicated without coordinate info,
                # so we check all directions by adjacency and static facts.

                # We must find the neighbor positions with known directions.
                # To do this efficiently, iterate static facts again and map adjacency + direction.

                # To speed up, let's build a direction map adjacency when first called.
                # We'll build it once in the constructor.

                # Here, we do the logic after building a direction map.
                pass

        # Since above logic is complicated without coordinate geometry, 
        # we will create a direction map in __init__ to enable this.

        # Return min_distance or large number:
        return min_distance

    def _build_direction_map(self):
        """
        Build mapping: For each location, map adjacent locations and the directions connecting them.

        Returns:
            A dict of dicts:
              direction_map[loc][dir] = neighbor_loc
              reverse_direction_map[loc][neighbor_loc] = direction
        """
        direction_map = {}
        reverse_direction_map = {}
        for fact in self.adjacency_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, dir = get_parts(fact)
                if loc1 not in direction_map:
                    direction_map[loc1] = {}
                direction_map[loc1][dir] = loc2
                if loc1 not in reverse_direction_map:
                    reverse_direction_map[loc1] = {}
                reverse_direction_map[loc1][loc2] = dir
        return direction_map, reverse_direction_map

    def _bfs_min_push_distance(self, box_loc, goal_loc, robot_loc, blocked):
        """
        Estimate minimal number of push moves + robot walks to push a box from box_loc to goal_loc.

        This method performs a simplistic BFS in the joint state space of box location,
        approximating push cost and robot positioning cost, but limited for efficiency.

        Since full joint state BFS is expensive, we approximate as sum of:
        - minimal distance to push box to goal_location (ignoring obstacles).
        - minimal cost for robot to position behind box for pushing.

        Args:
            box_loc: Current box location.
            goal_loc: Goal location to push box to.
            robot_loc: Current robot location.
            blocked: Set of locations currently blocked by other boxes.

        Returns:
            Estimated minimal push distance (int).
        """
        # We approximate the heuristic by:
        # 1) Shortest path distance from box_loc to goal_loc on adjacency graph, counting pushes.
        # 2) Shortest distance for robot to get behind the box at the start position.

        # Use BFS to find shortest box path to goal ignoring blocked except for other boxes.

        from collections import deque

        # Each state: box_pos
        queue = deque()
        queue.append((box_loc, 0))
        visited = {box_loc}

        while queue:
            current_box, cost = queue.popleft()
            if current_box == goal_loc:
                box_push_distance = cost
                break
            # Expanding all possible pushes of the box to neighbors
            for adj_loc in self.adjacency.get(current_box, []):
                # To push the box from current_box to adj_loc,
                # the position behind the box (opposite to (current_box->adj_loc)) must be clear.
                # So, find the direction from current_box to adj_loc:
                # Then robot must stand on the reverse location behind current_box.
                # The reverse position needs to be in adjacency and clear (not blocked).

                # Determine direction from current_box to adj_loc:
                dir_found = None
                for dir, neighbor in self.direction_map.get(current_box, {}).items():
                    if neighbor == adj_loc:
                        dir_found = dir
                        break
                if dir_found is None:
                    continue

                reverse_dir = self.reverse_dir[dir_found]
                # Check if reverse location behind box exists:
                push_pos = self.direction_map.get(current_box, {}).get(reverse_dir)
                if push_pos is None:
                    continue

                # Check if push_pos is clear (not blocked)
                if push_pos in blocked:
                    continue

                # Also the destination adj_loc must be clear:
                if adj_loc in blocked:
                    continue

                if adj_loc not in visited:
                    visited.add(adj_loc)
                    queue.append((adj_loc, cost + 1))
        else:
            # No path found, assume large cost
            box_push_distance = 1000

        # Compute minimal robot distance to a valid push position at start.
        # The robot must be able to get to push_pos with blocked including boxes.
        min_robot_dist = float("inf")
        robot_distances = self._bfs_distances(robot_loc, blocked)

        # For the robot to push initially, some adjacent position behind the box must be reachable.
        for dir, neighbor in self.direction_map.get(box_loc, {}).items():
            reverse_direction = self.reverse_dir[dir]
            push_pos = self.direction_map.get(box_loc, {}).get(reverse_direction)
            if push_pos is None:
                continue
            if push_pos in blocked:
                continue
            dist = robot_distances.get(push_pos, float("inf"))
            min_robot_dist = min(min_robot_dist, dist)

        if min_robot_dist == float("inf"):
            # No reachable push position for robot to start pushing this box.
            min_robot_dist = 1000

        # Returning sum of push distance plus robot walk distance as heuristic.
        return box_push_distance + min_robot_dist

    def __call__(self, node):
        """
        Compute heuristic value estimating the number of actions to solve the current state.

        Args:
            node: search node storing the 'state' attribute as a frozenset of facts.

        Returns:
            An integer heuristic estimate.
        """
        state = node.state

        # If goal state, heuristic is 0
        if self.goal_locations.items() <= {
            # (box, loc) from current state
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in state if match(fact, "at", "*", "*")
        }.items():
            return 0

        # Extract current robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
            # If no robot location found, heuristic unknown but finite:
            return 1000

        # Extract locations of all boxes
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        # If any goal box is missing from current state (should not happen), assign high cost
        if not all(box in box_locations for box in self.goal_locations.keys()):
            return 1000

        # Prepare blocked set: all box locations except the current box we compute for
        all_box_locs = set(box_locations.values())

        total_estimated_cost = 0

        # For direction mapping (build once)
        if not hasattr(self, "direction_map"):
            # Rebuild adjacency facts
            self.adjacency_facts = set()
            for loc in self.adjacency:
                for neighbor in self.adjacency[loc]:
                    # We don't know precise directions from raw adjacency,
                    # We'll approximate that directions can be matched by fact patterns.
                    # Therefore, we will build direction_map from static facts in __init__
                    # To do this, store adjacent facts from static for direction reads.
                    pass
            # Instead, build direction map now from static facts directly:
            self.direction_map = {}
            self.reverse_direction_map = {}
            for fact in self.static:
                if match(fact, "adjacent", "*", "*", "*"):
                    _, loc1, loc2, dir = get_parts(fact)
                    if loc1 not in self.direction_map:
                        self.direction_map[loc1] = {}
                    self.direction_map[loc1][dir] = loc2
                    if loc1 not in self.reverse_direction_map:
                        self.reverse_direction_map[loc1] = {}
                    self.reverse_direction_map[loc1][loc2] = dir

        # Compute heuristic contribution for each box by assigning it to closest unmatched goal location
        # via a greedy matching

        # clone goals list to assign greedily
        remaining_goals = set(self.goal_locations.values())

        # Heuristic: sum of minimal push+robot distances for each box to a unique goal location
        for box, box_loc in box_locations.items():
            min_cost = float("inf")
            selected_goal = None

            for goal_loc in remaining_goals:
                # Block all boxes except current one (allow pushing the box at box_loc)
                blocked = all_box_locs - {box_loc}

                cost = self._bfs_min_push_distance(box_loc, goal_loc, robot_loc, blocked)
                if cost < min_cost:
                    min_cost = cost
                    selected_goal = goal_loc

            if selected_goal is not None:
                remaining_goals.remove(selected_goal)
                total_estimated_cost += min_cost
            else:
                # No reachable goal found for this box: assign large heuristic value
                total_estimated_cost += 1000

        return total_estimated_cost
