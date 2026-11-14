from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the space-separated components of a PDDL fact string, without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'color' domain (secret_domain).
    The planning problem's goal facts are of the form (painted <tile-2>).
    This heuristic estimates the number of actions required to paint all
    goal tile-2 objects that are not yet painted. The domain's operators
    relevant for painting are:
      - change_color: moves an available-color entry for a robot (color-2)
        along an 'up' edge between tiles;
      - move_down: establishes a robot-at relation between a robot (color-2)
        and a color (color) when both have available-color at the same tile;
      - paint_up: performs the painting of a tile-2 at a tile where
        the robot's available-color, the tile's available-color and a free
        color coincide, also consuming the free color and the down predicate.

    Assumptions
    -----------
    - The heuristic treats painting tasks independently (no attempt to simulate
      full interaction between different painting actions beyond using the
      current state's static information). This gives a good balance between
      informativeness and computational cost for greedy best-first search.
    - The static 'up' relation is directed and change_color moves a robot's
      available-color only in the direction of the up edges (from lower tile
      to the tile indicated by the up edge).
    - There is typically a unique available-color position for each object,
      but the code tolerates multiple occurrences by considering all locations.
    - free-color facts are considered resources (a paint action requires a
      free-color value to be present). The heuristic only uses the current
      availability of free-color to determine whether a color can be used;
      it does not model the dynamic consumption of free-color across multiple
      goal paintings exactly (this keeps computation efficient).

    Heuristic Initialization
    ------------------------
    The constructor extracts:
      - The goal set (the painted tiles to achieve).
      - The static 'up' relation and stores it as a directed adjacency list
        (mapping each tile to the list of tiles reachable in one 'up' step).
    These static edges are used to compute the minimal number of change_color
    actions (the length of the directed path) required to move a robot's
    available-color from its current tile to a target tile.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state, the heuristic proceeds as follows:
      1. If all goal 'painted' facts are already present in the state, return 0.
      2. Parse the state facts into:
         - available_at: mapping from object -> set(locations) where
           (available-color <object> <loc>) holds.
         - loc_to_objects: mapping from location -> set(objects) that have
           available-color at that location.
         - robot_at_pairs: set of (robot, color) pairs derived from (robot-at ...)
         - free_colors: set of color objects for which (free-color ...) is true
         - painted_set: set of already-painted tile-2 objects
      3. For each unpainted goal tile-2 T:
         - Consider every tile-location L for which (available-color T L) holds.
         - For every color C in free_colors that has (available-color C L),
           and for every robot R that has any available-color location:
             - Compute the minimal number of change_color actions to move R's
               available-color from one of its current locations to L
               following the directed 'up' edges (breadth-first search).
             - If the robot-at pair (R, C) is already present in the state,
               no move_down action is needed; otherwise one move_down action
               is required to establish robot-at (R,C).
             - One paint_up action is required to paint T once all preconditions
               are satisfied.
             - Total estimated actions for this plan:
                 distance_moves + (0 or 1 move_down) + 1 paint_up
         - The heuristic picks the minimal-cost plan among all choices of L, C, R.
         - If no feasible plan is found for a goal tile (no matching L/C/R or
           location unreachable along up edges), the heuristic returns a large
           finite value (indicating the goal looks unreachable from current state).
      4. The final heuristic value is the sum of the individual minimal costs
         for all unpainted goal tiles.
      Notes:
        - The heuristic only counts the actions that are needed to establish
          the preconditions and to execute the paint_up itself (change_color
          steps, an optional move_down, and paint_up).
        - The heuristic is intentionally optimistic w.r.t. resource interactions
          (for instance, it does not perfectly model that free-color predicates
          are consumed by paint_up). This keeps computation efficient while
          remaining informative for guiding greedy best-first search.
    """

    def __init__(self, task):
        """Extract static 'up' edges and store the goal painted facts."""
        self.goals = set(task.goals)
        self.static_up = {}  # adjacency list: tile -> list of tiles reachable in one 'up'
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.static_up.setdefault(src, []).append(dst)

    def _bfs_distance(self, start, target):
        """
        Compute directed distance from start to target following up-edges.
        Returns non-negative integer distance, or None if target not reachable.
        """
        if start == target:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nbr in self.static_up.get(node, []):
                if nbr == target:
                    return d + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, d + 1))
        return None

    def __call__(self, node):
        """
        Estimate the number of actions to reach a goal state from the node's state.
        Returns 0 if goal already reached; otherwise returns a finite integer
        estimate (or a large integer if a goal appears unreachable).
        """
        state = node.state

        # Quick goal check: heuristic should be 0 only for goal states.
        if self.goals <= state:
            return 0

        # Parse state facts
        available_at = {}        # object -> set(locations) for (available-color obj loc)
        loc_to_objects = {}      # loc -> set(objects) that have available-color at loc
        robot_at_pairs = set()   # set of (robot, color)
        free_colors = set()      # set of color objects that are free
        painted = set()          # set of painted tile-2 objects

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                available_at.setdefault(obj, set()).add(loc)
                loc_to_objects.setdefault(loc, set()).add(obj)
            elif pred == "robot-at" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robot_at_pairs.add((robot, color))
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])
            elif pred == "painted" and len(parts) >= 2:
                painted.add(parts[1])

        # Goals: consider only painted goals
        goal_tiles = [get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "painted"]

        total_cost = 0
        LARGE = 100000  # large but finite value used if a goal seems unreachable

        for tile in goal_tiles:
            if tile in painted:
                continue  # already done

            best_cost = None

            # All locations L where the tile has available-color L
            candidate_locations = available_at.get(tile, set())
            if not candidate_locations:
                # No location where this tile is present -> cannot paint
                return LARGE

            # For each candidate location, check for usable free colors located there
            for L in candidate_locations:
                # Colors available at L (objects that appear in loc_to_objects and also are free)
                objs_at_L = loc_to_objects.get(L, set())
                candidate_colors = [c for c in objs_at_L if c in free_colors]
                if not candidate_colors:
                    # If there are no free colors at this location, cannot use this location
                    continue

                # For each color candidate and each robot that has available positions
                for C in candidate_colors:
                    # The paint_up precondition requires both (available-color C L) and (available-color tile L).
                    # We already know (available-color tile L); ensure (available-color C L) exists.
                    if C not in available_at or L not in available_at[C]:
                        continue

                    # For each robot (objects that are of type color-2 - we detect those with available_at entries
                    # that are not tile-2 or color; we cannot rely on typing here, so consider all objects that
                    # appear in available_at and are not the tile itself and not the color object C)
                    for robot, robot_locs in available_at.items():
                        # Skip using the tile object itself or the color object as a robot candidate.
                        if robot == tile or robot == C:
                            continue
                        # robot must be a color-2 in domain modeling; if it has no available positions, skip
                        if not robot_locs:
                            continue

                        # For each starting location of the robot, compute distance to L
                        for start_loc in robot_locs:
                            dist = self._bfs_distance(start_loc, L)
                            if dist is None:
                                # This robot location cannot reach L via change_color actions
                                continue

                            # Determine whether move_down is needed to establish robot-at (robot, C)
                            need_move_down = 0 if (robot, C) in robot_at_pairs else 1

                            # One paint_up action required
                            cost = dist + need_move_down + 1

                            if best_cost is None or cost < best_cost:
                                best_cost = cost

            if best_cost is None:
                # Could not find any feasible plan for this goal tile
                return LARGE

            total_cost += best_cost

        return total_cost