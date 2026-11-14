from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Sokoban domain.

    The heuristic estimates the number of actions required to reach the goal
    from a given state by:
      - Counting the minimal number of pushes required to move each box to
        its goal location (computed as the length of a shortest path for the
        box on the location graph).
      - Adding an estimate of the robot's travel cost to reach the starting
        push positions for all boxes. The travel cost is approximated by the
        weight of a minimum spanning tree (MST) over the set of required
        push-start positions together with the robot's current position. Pair
        weights are the shortest path lengths (in robot moves) between locations.

    Assumptions
    -----------
    - Static adjacency facts describe a connected grid/graph of locations.
    - Goals are of the form (at boxX loc_Y) (i.e., each goal ties a *specific*
      box to a specific goal location). The heuristic supports multiple boxes.
    - The heuristic is optimistic with respect to blocking: it ignores other
      boxes/clear preconditions when computing shortest paths for boxes. This
      makes it efficiently computable and informative, but not necessarily
      admissible.
    - Each push moves a box by exactly one graph edge (matching the domain).
      The robot ends up in the box's previous location after a push, which
      allows chained pushes along a path without intervening robot moves.

    Heuristic Initialization
    ------------------------
    - Extract the goal mapping for boxes (box -> goal_location) from task.goals.
    - Parse the static adjacency facts and build:
        * graph: a neighbors mapping for each location (undirected).
        * adj_dir: a mapping (loc1, loc2) -> direction for directed adjacency
                   facts. This is used to find the specific robot location that
                   is required to push from one location to the next along a path.
    - No dynamic precomputation of all-pairs distances is done up-front; BFS
      is used on-demand. The number of boxes (and thus BFS calls) tends to be small.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node (state):

    1. If the state's facts already satisfy all goals, return 0.

    2. Parse the state to obtain:
       - The robot's current location.
       - The current location of each box.

    3. For every box that is not already at its goal:
       a. Compute a shortest path (in number of edges) on the location graph
          from the box's current location to its goal location. This number is
          the minimal number of pushes required for that box (ignoring other
          boxes and clear constraints).
       b. Reconstruct the first step of that path. To perform the first push
          (moving the box from its start to the next node on the path) the
          robot must be in a location r such that r, start, next are
          collinear in the same direction. The code uses the directed
          adjacency facts to identify that required r. If the directed facts
          don't permit identifying a unique r (rare), it falls back to any
          neighbor of the box that is not the immediate target.

       Collect:
         - total_pushes := sum of push counts over boxes,
         - push_start_positions := set of robot locations necessary to begin
           pushing each non-solved box along the chosen shortest path.

    4. Estimate the robot travel cost to visit all required push-start positions:
       - Construct the complete graph over nodes = {robot_position} U push_start_positions,
         with edge weights equal to shortest-path distances (in robot moves)
         between those locations (computed by BFS on the location graph).
       - Compute the weight of a minimum spanning tree (MST) over that complete graph.
         The MST weight is a fast, admissible lower-bound for the cost of visiting
         all those positions starting from the robot. We use it as an optimistic
         estimate of the robot movement cost.

    5. The final heuristic value is:
         heuristic = total_pushes + mst_travel_cost

    Notes on correctness / implementation details
    ---------------------------------------------
    - get_parts() strips parentheses correctly, so object names are extracted
      without surrounding brackets.
    - If any required path between a box and its goal or between two locations
      needed for MST is unreachable, the heuristic returns a large finite
      value (treated as "very costly" so the search will prefer other branches).
    - All used modules are imported (collections.deque and the Heuristic base).
    """

    def __init__(self, task):
        """
        Extract static information and goals from the task.

        - self.goals: set of goal facts (strings)
        - self.box_goals: mapping box_name -> goal_location (string)
        - self.graph: adjacency mapping location -> list of neighboring locations
        - self.adj_dir: mapping (loc1, loc2) -> direction (string) from static facts
        """
        self.goals = task.goals
        static_facts = task.static

        # Map goals: only consider 'at' goals where a box must be at a location.
        self.box_goals = {}
        for g in self.goals:
            parts = get_parts(g)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # We assume that the 'at' goal refers to a box if the object is typed as a box.
                self.box_goals[obj] = loc

        # Build adjacency graph and directed adjacency (with direction labels).
        self.graph = {}  # location -> set(neighbor_locations)
        self.adj_dir = {}  # (loc1, loc2) -> direction

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "adjacent" and len(parts) == 4:
                l1, l2, d = parts[1], parts[2], parts[3]
                # neighbors (we treat the graph as undirected for robot movement
                # and for box movement distances). Still store the directed pair.
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1)
                self.adj_dir[(l1, l2)] = d
                # Note: we keep (l2, l1) entry only if present in static facts
                # (the domain often lists both directions). We do not invent a
                # direction for the reverse edge.
        # Convert neighbor sets to lists for stable iteration
        for loc in list(self.graph.keys()):
            self.graph[loc] = list(self.graph[loc])

    def __call__(self, node):
        """
        Compute the heuristic value for the given node.

        Returns an integer estimate of the number of actions required to reach the goal.
        """
        state = node.state

        # 1) If goal already reached -> zero heuristic.
        if self.goals <= state:
            return 0

        # Parse state: robot location, box positions.
        robot_loc = None
        box_positions = {}  # box -> loc
        # also track set of locations (some tasks might not list isolated locations in static)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at-robot" and len(parts) >= 2:
                robot_loc = parts[1]
            elif pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # treat as a box if it's present in box_goals or if name starts with 'box'
                # (safe fallback)
                if obj in self.box_goals or obj.startswith("box"):
                    box_positions[obj] = loc

        # If robot location unknown in state (shouldn't happen), treat as high cost.
        if robot_loc is None:
            return 10 ** 6

        # Helper: BFS single-source over self.graph, return distances dict and parents for path recon.
        def bfs(source):
            if source not in self.graph:
                return None, None
            dist = {source: 0}
            parent = {source: None}
            q = deque([source])
            while q:
                u = q.popleft()
                for v in self.graph.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return dist, parent

        total_pushes = 0
        push_start_positions = set()  # robot positions required to start pushing each box along its shortest path

        LARGE = 10 ** 6

        # For each box with a goal, compute pushes required and the initial push-start location.
        for box, goal_loc in self.box_goals.items():
            # If box not present in state (weird), treat as costly
            if box not in box_positions:
                return LARGE
            start = box_positions[box]
            if start == goal_loc:
                # already in goal
                continue

            # BFS from start to find shortest path (in box moves) to goal
            dist, parent = bfs(start)
            if dist is None or goal_loc not in dist:
                # unreachable goal for this box (consider very costly)
                return LARGE

            pushes_needed = dist[goal_loc]
            total_pushes += pushes_needed

            # Reconstruct path from start to goal to identify the first step.
            # Build the path list [start, ..., goal]
            path = []
            cur = goal_loc
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()  # now path[0] == start

            if len(path) >= 2:
                s = path[0]      # box start
                n1 = path[1]     # first cell box will move into
                # Determine robot position r such that adj(r, s, d) and adj(s, n1, d)
                required_r = None
                # look for neighbor r of s such that adj_dir[(r,s)] == adj_dir[(s,n1)]
                dir_sn = self.adj_dir.get((s, n1))
                if dir_sn is not None:
                    for r in self.graph.get(s, ()):
                        if r == n1:
                            continue
                        d_rs = self.adj_dir.get((r, s))
                        if d_rs is not None and d_rs == dir_sn:
                            required_r = r
                            break
                # fallback: pick any neighbor of s that is not n1 (the robot must be adjacent, best-effort)
                if required_r is None:
                    for r in self.graph.get(s, ()):
                        if r != n1:
                            required_r = r
                            break
                if required_r is None:
                    # isolated start (shouldn't occur), but treat as costly
                    return LARGE
                push_start_positions.add(required_r)
            else:
                # path length < 2 but start != goal (should not happen), treat as costly
                return LARGE

        # If no pushes needed (all boxes at goal), return 0 (we checked goals earlier, but be safe)
        if total_pushes == 0:
            return 0

        # If there are no push_start_positions (should not happen if total_pushes>0), return pushes only
        if not push_start_positions:
            return total_pushes

        # Build node list for MST: include robot current location and all push_start_positions
        mst_nodes = [robot_loc] + list(push_start_positions)

        # Precompute BFS distances from each mst_node to all locations (we only need distances to other mst_nodes)
        all_dists = {}
        for node_loc in mst_nodes:
            dists, _ = bfs(node_loc)
            if dists is None:
                return LARGE
            all_dists[node_loc] = dists

        # Build complete graph distances matrix and check reachability between mst nodes
        def pair_dist(a, b):
            da = all_dists.get(a)
            if da is None:
                return None
            return da.get(b)

        # If any required pair is unreachable, return large cost
        for a in mst_nodes:
            for b in mst_nodes:
                if a == b:
                    continue
                pd = pair_dist(a, b)
                if pd is None:
                    return LARGE

        # Compute MST cost using Prim's algorithm over the complete graph of mst_nodes
        # Node indices
        nodes = mst_nodes[:]
        in_mst = {nodes[0]}
        remaining = set(nodes[1:])
        mst_cost = 0
        # Precompute pair distances in a dict for speed
        pair_cache = {}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a = nodes[i]
                b = nodes[j]
                pair_cache[(a, b)] = pair_cache[(b, a)] = pair_dist(a, b)

        while remaining:
            best_edge = None
            best_cost = None
            for u in in_mst:
                for v in list(remaining):
                    cost = pair_cache[(u, v)]
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_edge = v
            if best_edge is None:
                # disconnected
                return LARGE
            mst_cost += best_cost
            in_mst.add(best_edge)
            remaining.remove(best_edge)

        # Final heuristic is total pushes plus estimated robot travel (mst_cost).
        # Both pushes and robot moves are unit-cost actions in the domain.
        return int(total_pushes + mst_cost)