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
    A domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to push all boxes
    into their goal adjacency state (adjacent-2 predicate). It calculates a cost
    based on how many boxes are not yet adjacent-2 and the minimal effort to get
    the robot and boxes aligned for pushing.

    Assumptions
    -----------
    - The goal is to have every box satisfy the predicate `(adjacent-2 box)`.
    - Boxes must be pushed by the robot to their goal locations.
    - The robot moves over locations (`at-robot-2`) and manipulates boxes that are adjacent to locations (`adjacent box location`).
    - The robot can push boxes if it is at the same location as the box (`at-robot box location`) and the box is currently at some location (`at box`).
    - Moving the robot or pushing boxes costs roughly one action each.
    - Locations and their clear connections (`clear loc1 loc2`) represent possible robot moves.
    - We ignore obstacles and assume simple heuristics based on distances derived from static facts and current state.
    - The heuristic does not have to be admissible but should guide the search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract the list of all boxes from static facts by collecting all objects typed as boxes.
    - Extract robot initial location(s) from static and current state (using fact `(at-robot-2 loc)`).
    - Build a graph of locations connected by the `clear` predicate from static facts.
    - Extract adjacency relations between boxes and locations.
    - Extract goal boxes that must satisfy `(adjacent-2 box)`.
    - Precompute shortest distances on the location graph for heuristic distance calculations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify boxes that do **not** satisfy `(adjacent-2 box)` yet — these still need pushing.
    2. For each such box:
       - Identify the goal location to which it needs to be adjacent (from the goal or from static `adjacent box location` facts).
       - Estimate the distance from the box's current location to a goal location for adjacency.
    3. Estimate the distance of the robot to get into position to push each non-adjacent box:
       - The robot must be at a location adjacent to the box's current location (`at-robot-2 loc` and `adjacent box loc` in static).
       - Use shortest path distance on the location graph from robot's current location to that push position.
    4. The heuristic sums roughly the minimal distances required to:
       - Move the robot near each box.
       - Push each box to its adjacency goal.
    5. Since the path details are unknown, the heuristic uses the sum of minimal robot moves plus one action per push needed to reach the goal adjacency for each box.
    6. This results in a heuristic value estimating the minimal effort to push all boxes to their goal adjacency.
    7. If state is goal (all boxes adjacent-2), heuristic returns 0.

    This heuristic is efficiently computable by using precomputed distances and simple set checks,
    and it balances guiding the search toward pushing boxes and positioning the robot without full path planning.
    """

    def __init__(self, task):
        # Extract list of boxes and locations from the task facts.
        self.boxes = set()
        self.locations = set()
        for fact in task.facts:
            # Facts like (adjacent box1 location6) -> check object names and types indirectly
            # We do not have type infos directly, infer from domain naming conventions:
            # boxes are objects mentioned in box positions predicates, locations in location predicates.
            # More robust: infer from static facts.
            parts = get_parts(fact)
            # collect box objects from relevant predicates
            if parts[0] == "adjacent":
                # parts = ["adjacent", box, location]
                self.boxes.add(parts[1])
                self.locations.add(parts[2])
            elif parts[0] == "at-robot":
                # (at-robot box location)
                self.boxes.add(parts[1])
                self.locations.add(parts[2])
            elif parts[0] == "clear":
                # (clear loc1 loc2)
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif parts[0] == "adjacent-2":
                # (adjacent-2 box)
                self.boxes.add(parts[1])
            elif parts[0] == "at-robot-2":
                # (at-robot-2 location)
                self.locations.add(parts[1])
            elif parts[0] == "at":
                # (at box)
                # Note: in domain (at ?v1 - box), only box arg, so parts length=2
                if len(parts) == 2:
                    self.boxes.add(parts[1])

        # Keep boxes and locations as sorted lists for deterministic iteration
        self.boxes = sorted(self.boxes)
        self.locations = sorted(self.locations)

        # Store the goal goal adjacency boxes that must satisfy adjacent-2 predicate
        # from the task's goal conditions
        self.goal_boxes = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.add(parts[1])

        # Static facts to build adjacency between locations and boxes
        self.adjacent_box_to_locations = {}
        for fact in task.static:
            if match(fact, "adjacent", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                self.adjacent_box_to_locations.setdefault(box, set()).add(loc)

        # Build graph of locations using clear predicate (edges are between locations)
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in task.static:
            if match(fact, "clear", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                if loc1 in self.location_graph and loc2 in self.location_graph:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1)

        # Precompute all pairs shortest paths (Floyd-Warshall or BFS per location)
        # Locations are few enough for BFS per location.
        self.location_distances = self._compute_all_pairs_shortest_paths(self.location_graph)

    @staticmethod
    def _compute_all_pairs_shortest_paths(graph):
        """Compute shortest path lengths between all locations in the undirected graph."""
        distances = {}
        for start in graph:
            distances[start] = {loc: float('inf') for loc in graph}
            distances[start][start] = 0
            queue = [start]
            idx = 0
            while idx < len(queue):
                current = queue[idx]
                idx += 1
                curr_dist = distances[start][current]
                for neighbor in graph[current]:
                    if distances[start][neighbor] == float('inf'):
                        distances[start][neighbor] = curr_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute heuristic estimate of actions to reach the goal from given node/state.

        Returns 0 if goal reached (all boxes satisfy adjacent-2).
        Otherwise, returns heuristic cost estimating pushing effort plus robot moves.
        """

        state = node.state

        # Quick goal check
        if self.goal_boxes <= state:
            # All goal boxes have adjacent-2 satisfied in state
            return 0

        # Helper: find robot location (at-robot-2 location)
        robot_location = None
        for fact in state:
            if match(fact, "at-robot-2", "*"):
                robot_location = get_parts(fact)[1]
                break
        if robot_location is None:
            # Should not happen, robot must be somewhere
            # Return large heuristic value to discourage this state
            return 1000

        # Determine which boxes are already adjacent-2 (goal satisfied boxes)
        boxes_adjacent2 = set()
        for fact in state:
            if match(fact, "adjacent-2", "*"):
                boxes_adjacent2.add(get_parts(fact)[1])

        # Boxes still needing pushing = goal_boxes - boxes already adjacent-2
        boxes_to_push = self.goal_boxes - boxes_adjacent2

        # Extract current locations of boxes:
        # The predicate (at box) represents the box's current location.
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    box = parts[1]
                    # The box has no explicit location object in predicate (at box),
                    # but domain implies "at box" means box is at its current location.
                    # This domain's 'at' predicate only takes one argument (per domain file),
                    # so "at box" means box is at its current unknown location (likely implicit).
                    # Since 'adjacent box location' gives location info, approximate by any adjacent location of the box.
                    # But we do have (adjacent box location) for boxes in static facts; however static does not reflect current position.
                    # We have (at-robot box location) predicates in static for possible box-location mapping.
                    # The only reliable location info per current state is from `(at-robot box location)` in static? That is static info.
                    # Instead, we look into static facts for `(adjacent box location)` pairs to assume these are possible locations.

                    # Without explicit box location from state, approximate by:
                    # 1) if (adjacent box loc) static contains one unique location, use it
                    # 2) else unset -> meaning box location unknown, assign None.
                    locs = self.adjacent_box_to_locations.get(box)
                    if locs and len(locs) == 1:
                        box_locations[box] = next(iter(locs))
                    else:
                        box_locations[box] = None
            # Also consider if (adjacent box location) is in state? No, adjacent only in static.
            # We can also check the state for (at-robot box location) facts to get the box's current position.
        # If no box location info found via above, fallback to static:
        # To improve, check in state facts for `(at-robot box location)`
        for fact in state:
            if match(fact, "at-robot", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        # For boxes not yet assigned a location, assign None
        for box in boxes_to_push:
            if box not in box_locations:
                box_locations[box] = None

        heuristic = 0

        # For each box that still needs to be pushed to adjacent-2:
        for box in boxes_to_push:
            box_loc = box_locations.get(box)

            # If no location info, count a flat penalty:
            if box_loc is None:
                heuristic += 5  # arbitrary penalty for unknown box location
                continue

            # Find goal locations where this box must be adjacent
            goal_locs = self.adjacent_box_to_locations.get(box, set())
            if not goal_locs:
                # No adjacency goal location info, penalty:
                heuristic += 5
                continue

            # Estimate minimal distance from box_loc to any goal location
            dist_box_to_goal = min(
                self.location_distances.get(box_loc, {}).get(goal_loc, float('inf'))
                for goal_loc in goal_locs
            )
            if dist_box_to_goal == float('inf'):
                dist_box_to_goal = 10  # penalty for unreachable

            # Compute minimal distance for robot to reach a location adjacent to the box to push it:
            # robot must be at some location (loc_r) where (adjacent box loc_r) holds (static fact)
            # find all robot push positions near the box

            robot_push_positions = self.adjacent_box_to_locations.get(box, set())
            if not robot_push_positions:
                # No known push positions, penalty
                dist_robot_to_push = 10
            else:
                # Distance from current robot location to closest push position
                dist_robot_to_push = min(
                    self.location_distances.get(robot_location, {}).get(push_loc, float('inf'))
                    for push_loc in robot_push_positions
                )
                if dist_robot_to_push == float('inf'):
                    dist_robot_to_push = 10  # penalty

            # The heuristic adds:
            #  - robot distance to get into pushing position
            #  - pushes/actions to move box the estimated distance toward adjacency goal
            # We add 1 push action per step box needs to move, plus robot movement

            heuristic += dist_robot_to_push + dist_box_to_goal + 1  # the +1 approximates the push action

        # The heuristic value estimates the minimal sum of actions to push all boxes to adjacency-2 and robot to pushing positions
        # It is never zero unless goal is reached, finite for solvable states.

        return heuristic
