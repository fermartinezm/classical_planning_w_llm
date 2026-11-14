from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `secret_domain` (color) domain.
    The domain models painting of "down" tiles (tile-2 objects) by bringing a robot
    (color-2 objects) to the tile above the tile-to-paint, pairing the robot with
    a free paint color (a color object) located at that same tile, performing a
    'move_down' to configure the robot, and then executing 'paint_up' to paint the
    tile-2. There are also operators that let color-2 objects move upwards along
    an `up` graph (change_color). The heuristic estimates the number of actions
    (change_color moves + move_down + paint_up) required to paint all remaining
    target tiles.

    Assumptions
    -----------
    - Goals are only of the form (painted <tile-2>).
    - A tile-2 must be "available" at some tile (via an (available-color <tile-2> <tile>) fact);
      painting requires the tile-2 to be located at the tile above (the same tile where a paint color
      and a robot can be co-located).
    - A paint operation for one tile requires:
        * the tile-2 to be available at a tile X (available-color <tile-2> X),
        * a paint color C that is both free (free-color C) and available at X (available-color C X),
        * a robot (color-2 object) available at some tile Y that can be moved via change_color
          (following up edges) to X (cost = number of change_color moves),
        * then one move_down and one paint_up action.
    - Colors (paint resources) are fixed at tiles and are consumed (free-color removed) when used.
      Robots can be moved upwards along the up graph via change_color; for efficiency the heuristic
      treats any object that currently has an available-color position as a potential mover (this
      is a light relaxation — the actual domain restricts which objects can be moved, but this
      approximation keeps the heuristic fast and informative).
    - The heuristic is not required to be admissible; it aims to be informative and cheap to compute.

    Heuristic Initialization
    ------------------------
    - Store the goal facts passed in via `task.goals` as the set of required painted tiles.
    - Extract static `up` relations from `task.static` and build a directed adjacency map
      from each tile to its upward neighbors (an edge tileA -> tileB exists if (up tileA tileB) is static).
    - Precompute nothing else heavy; distances are computed on demand with BFS.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (with its `state` set of facts), compute the heuristic value as follows:

    1. If all goal facts (painted tiles) are already present in the state, return 0.

    2. Identify which goal tiles are still unpainted:
       - For every goal fact "(painted T)", if "(painted T)" not in current state, T must still be painted.

    3. For each tile-to-paint T:
       a. Find the tile position X where the tile-2 object T is currently located:
          - This is the second argument of a fact "(available-color T X)" in the state.
          - If no such available-color fact exists for T, painting T is considered impossible under current
            assumptions -> assign a large finite penalty (see below).
       b. Verify that there is at least one free paint color C that is available at X:
          - I.e., "(free-color C)" in state and "(available-color C X)" in state.
          - If none, painting T is considered infeasible under current resources -> assign large penalty.
       c. Estimate the minimal number of change_color moves required to bring some mover object
          from its current tile Y to X:
          - Collect candidate start tiles Y from the second argument of ANY "(available-color O Y)" fact
            in the state (this is a relaxed assumption that any available object could serve as a mover).
          - Using BFS on the static up graph, compute the minimal number of up-steps from Y to X.
            The change_color operator moves along an up edge (from v7 to v8), hence the BFS distance
            equals the number of change_color actions required.
          - If no start tile can reach X, painting T is treated as infeasible -> large penalty.
       d. The estimated cost to paint T (under the described relaxation) is:
            cost(T) = (min number of change_color moves to reach X) + 1 (move_down) + 1 (paint_up)

    4. Sum the per-target costs to produce the heuristic estimate. If any target is infeasible under the
       relaxed assumptions (no available tile-2 location, no free color at that location, or no reachable
       mover), the heuristic returns a large finite cost (chosen to be proportional to the number of
       remaining tiles to encourage exploration but remain finite).

    Notes on resource contention:
    - The heuristic checks that each tile's location X has at least one free color available at X.
      It does not globally solve an assignment problem for colors vs. tiles (that would be more expensive
      to compute). This conservative check avoids grossly underestimating when there is no color at all
      for a given tile location.

    Complexity
    ----------
    - BFS on the up graph is typically inexpensive since the number of tiles is moderate. The heuristic
      performs at most one BFS per (start tile, target tile) pair considered, but it caches distances per
      call to avoid repeated exploration for the same source tile.
    """

    INFEASIBLE_COST = 1000  # Large finite penalty for infeasible targets (keeps heuristic finite)

    def __init__(self, task):
        """Extract goal facts and static up-relations from the given task."""
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build adjacency for the up graph: edges u -> v when (up u v) is static.
        self.up_adj = defaultdict(list)
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.up_adj[src].append(dst)

    def _bfs_distance(self, src, dst):
        """
        Compute shortest distance (number of up-steps) from tile `src` to tile `dst`
        following the static up adjacency. Return None if unreachable.
        """
        if src == dst:
            return 0
        visited = {src}
        q = deque([(src, 0)])
        while q:
            cur, d = q.popleft()
            for nb in self.up_adj.get(cur, ()):
                if nb == dst:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate for the given planner node."""
        state = node.state

        # If all goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract currently painted tiles.
        painted = {get_parts(f)[1] for f in state if f.startswith("(painted ")}

        # Determine which goal tiles still need painting (extract tile id from goal facts).
        pending_tiles = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted":
                tile = parts[1]
                if g not in state:
                    pending_tiles.append(tile)

        if not pending_tiles:
            return 0

        # Map available-color facts: obj -> tile (many objects may share same tile)
        # Also map for quick membership tests: available_color_pairs = set of (obj, tile)
        available_color_pairs = set()
        obj_at_tile = defaultdict(list)  # tile -> list of objects available there
        for f in state:
            parts = get_parts(f)
            if parts and parts[0] == "available-color" and len(parts) >= 3:
                obj, tile = parts[1], parts[2]
                available_color_pairs.add((obj, tile))
                obj_at_tile[tile].append(obj)

        # Set of free colors
        free_colors = {get_parts(f)[1] for f in state if f.startswith("(free-color ")}

        # Candidate start tiles for movers: any tile that holds some available-color object.
        start_tiles = set(tile for (_, tile) in available_color_pairs)

        # Cache BFS distances from each start tile to targets during this call
        dist_cache = {}

        total_cost = 0

        for target in pending_tiles:
            # Find the tile X where this tile-2 object is currently located
            loc = None
            for (obj, tile) in available_color_pairs:
                if obj == target:
                    loc = tile
                    break
            if loc is None:
                # No known location for target tile-2 -> infeasible under relaxation
                total_cost += self.INFEASIBLE_COST
                continue

            # Check for at least one free color available at that location
            colors_at_loc = [c for c in obj_at_tile.get(loc, []) if c in free_colors]
            if not colors_at_loc:
                # No free paint color at the tile location -> infeasible
                total_cost += self.INFEASIBLE_COST
                continue

            # Compute minimal number of change_color moves from any start tile to loc
            min_dist = None
            for s in start_tiles:
                # Cache key for BFS
                key = (s, loc)
                if key in dist_cache:
                    d = dist_cache[key]
                else:
                    d = self._bfs_distance(s, loc)
                    dist_cache[key] = d
                if d is not None:
                    if (min_dist is None) or (d < min_dist):
                        min_dist = d

            if min_dist is None:
                # No mover can reach the required tile location -> infeasible
                total_cost += self.INFEASIBLE_COST
                continue

            # Cost to paint this target: change_color moves (min_dist) + move_down + paint_up
            total_cost += (min_dist + 1 + 1)

        return total_cost