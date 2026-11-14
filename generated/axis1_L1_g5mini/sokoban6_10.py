from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    The domain has:
      - zdufuwne <loc> : the robot is at <loc>.
      - ldlu1rwo <box> <loc> : box is at <loc>.
      - qsw2k0xh <loc> : location is currently free (no box).
      - h80z8tvp <from> <to> <label> : a directed, labeled adjacency (static).

    The only actions are:
      - a simple robot step along an h80z8tvp edge if the target node is free (qsw2k0xh),
      - a push action that moves a box from u->v provided the robot stands at some p with
        h80z8tvp(p,u,label) and h80z8tvp(u,v,label) (i.e., two consecutive edges with the same label).

    Assumptions
    -----------
    - We treat the static adjacency facts (h80z8tvp ...) as a directed labeled graph.
    - A single push moves a box one step along the "push-graph" formed by two consecutive edges
      sharing the same label: if (p->u, label) and (u->v, label) exist then the box can move from u->v.
    - Robot moves follow directed edges (h80z8tvp u v label) but only to target nodes that are
      currently free (qsw2k0xh v) in the given state (matching the domain precondition).
    - For efficiency, we ignore dynamic freeing/occupation of tiles that could occur after other
      pushes; the heuristic is allowed to be non-admissible and may therefore be optimistic.

    Heuristic Initialization
    ------------------------
    - Build:
      1. A directed robot-move graph (robot_graph) from h80z8tvp edges (u -> v for any label).
      2. A directed push-graph (push_graph) for boxes: an edge u -> v exists if there are
         static facts (p -> u, L) and (u -> v, L) for the same label L.
      3. For each location u, the set of pushing positions (p) from which the robot can push a box at u.
    - Extract goal locations for boxes from the task.goals set (supports multiple box goals).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are already true in the state, return 0.
    2. Extract current box locations from the state's ldlu1rwo facts.
    3. For each box that is not yet at its goal location(s):
       - Compute the shortest number of pushes (BFS) in the static push_graph from the box's current
         position to one of its goal positions. Each push counts as cost 1 (one push action).
       - If no path exists in the push_graph, we assign a large finite penalty (LARGE_COST) to keep
         the heuristic finite (this typically indicates that the box cannot reach its goal under
         static connectivity constraints).
    4. Compute the robot's minimal number of step moves to reach any valid pushing position for any
       non-goal box:
       - Valid pushing positions for a box at node u are the predecessors p for which there exists
         a same-label pair (p->u, L) and (u->v, L) (already computed in initialization).
       - Robot movement is constrained by qsw2k0xh in the current state: the robot can only step into
         nodes that are currently marked free; however, the robot's starting node is allowed even if
         it is not free.
       - If no pushing position is reachable under the current free-tiles assumption, we treat the
         robot distance as 0 (optimistic).
    5. The returned heuristic is:
         heuristic = (sum of minimal pushes for each non-goal box) + robot_distance_to_first_push
       - This is designed to be informative (captures per-box push effort) while remaining
         efficiently computable (BFS on static graphs).
    """

    LARGE_COST = 1000  # penalty for unreachable boxes in the push-graph

    def __init__(self, task):
        """Precompute static graphs and goal mappings from the task."""
        self.goals = set(task.goals)
        static_facts = task.static

        # Robot move graph: directed edges u -> v (ignoring labels) for robot steps
        self.robot_graph = defaultdict(set)  # node -> set(neighbor_nodes)

        # Adjacency by label: label -> list of (from, to)
        edges_by_label = defaultdict(list)

        # Build adjacency_by_label and robot_graph from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp" and len(parts) >= 4:
                src, dst, label = parts[1], parts[2], parts[3]
                edges_by_label[label].append((src, dst))
                # robot can step from src to dst (if dst is free at runtime)
                self.robot_graph[src].add(dst)

        # Build push_graph: a box can move u -> v if exists (p->u, L) and (u->v, L)
        self.push_graph = defaultdict(set)  # box-location u -> set of reachable box-locations v (one push)
        # Also compute pushing positions: for a box at u, which p can the robot stand on to push it?
        self.pushing_positions = defaultdict(set)  # u -> set(pushing_positions p)

        # For each label, index edges by source and by destination for fast pairing
        for label, edgelist in edges_by_label.items():
            # map from node to outgoing neighbors under this label
            out_by_node = defaultdict(list)
            in_by_node = defaultdict(list)
            for a, b in edgelist:
                out_by_node[a].append(b)
                in_by_node[b].append(a)

            # For any u that has incoming (p->u) and outgoing (u->v) under the same label,
            # we can push box from u to v and robot must be at p to push.
            for u in set(in_by_node.keys()) & set(out_by_node.keys()):
                predecessors = in_by_node[u]
                successors = out_by_node[u]
                for p in predecessors:
                    for v in successors:
                        # push edge u -> v (box moves)
                        self.push_graph[u].add(v)
                        # robot must be at p to do that push
                        self.pushing_positions[u].add(p)

        # Extract goal locations for boxes from task.goals
        # Support multiple goals per box (list)
        self.goal_locations = defaultdict(set)  # box -> set(goal_locations)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.goal_locations[box].add(loc)

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node."""
        state = node.state  # frozenset of fact strings

        # If all goals are satisfied in this state, heuristic is 0.
        if self.goals <= state:
            return 0

        # Collect current box positions from state (ldlu1rwo facts)
        box_positions = {}  # box -> loc
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                box_positions[box] = loc

        # Sum minimal pushes for each box not at goal
        total_pushes = 0
        for box, goal_locs in self.goal_locations.items():
            current_loc = box_positions.get(box)
            # If box not present in state (shouldn't happen), penalize heavily
            if current_loc is None:
                total_pushes += self.LARGE_COST
                continue

            if current_loc in goal_locs:
                # already in a goal location for this box
                continue

            # BFS on push_graph from current_loc to any of goal_locs
            q = deque([(current_loc, 0)])
            visited = {current_loc}
            found_distance = None
            while q:
                u, dist = q.popleft()
                if u in goal_locs:
                    found_distance = dist
                    break
                for v in self.push_graph.get(u, ()):
                    if v not in visited:
                        visited.add(v)
                        q.append((v, dist + 1))

            if found_distance is None:
                # unreachable in the static push graph: assign a large finite cost
                total_pushes += self.LARGE_COST
            else:
                total_pushes += found_distance

        # Find robot location (zdufuwne fact). If none, assume robot distance 0 (optimistic).
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "zdufuwne" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # If there are no boxes needing pushing, robot distance is 0.
        boxes_need_push = [
            box for box, goals in self.goal_locations.items()
            if box_positions.get(box) not in goals
        ]
        if not boxes_need_push:
            return total_pushes  # likely 0 but kept for generality

        # Compute set of target pushing positions that would enable an immediate push for any non-goal box.
        target_pushing_positions = set()
        for box in boxes_need_push:
            cur = box_positions.get(box)
            if cur is None:
                continue
            target_pushing_positions.update(self.pushing_positions.get(cur, ()))

        # If robot location is unknown or there are no pushing positions (static), be optimistic.
        robot_dist = 0
        if robot_loc is not None and target_pushing_positions:
            # Prepare set of currently free nodes (qsw2k0xh facts) for robot move precondition
            free_nodes = {get_parts(f)[1] for f in state if get_parts(f)[0] == "qsw2k0xh" and len(get_parts(f)) >= 2}

            # BFS on robot movement graph: robot can step from u -> v only if v is currently free.
            q = deque([(robot_loc, 0)])
            visited = {robot_loc}
            found = False
            while q and not found:
                u, dist = q.popleft()
                if u in target_pushing_positions:
                    robot_dist = dist
                    found = True
                    break
                # explore neighbors (robot_graph[u]) but only those v that are currently free
                for v in self.robot_graph.get(u, ()):
                    if v not in visited and v in free_nodes:
                        visited.add(v)
                        q.append((v, dist + 1))
            # If not found, keep robot_dist = 0 (optimistic)

        # Final heuristic: pushes (each counted as one push action) plus robot moves to first push
        heuristic_value = total_pushes + robot_dist
        # Ensure finite integer
        return int(heuristic_value)