from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the tokenized parts of a PDDL fact string, stripping surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (a Sokoban-like pushing domain).
    It estimates the number of actions required to place all boxes at their goal
    locations. The estimate combines:
      - a push-distance estimate (number of pushes required for each box to
        reach its goal, computed on a graph of feasible box-moves derived from
        the static adjacency facts),
      - plus a small effort to get the agent to the location required to start
        pushing (minimal number of agent moves to a valid push-position).

    Assumptions
    -----------
    - Facts:
        - (zdufuwne L) : the agent is at location L.
        - (ldlu1rwo BOX L) : box BOX is at location L.
        - (qsw2k0xh L) : location L is "empty" (agent can move into it).
        - (h80z8tvp A B LABEL) : static adjacency from A to B labeled by LABEL.
    - A push (moving a box from A to B) is possible if there exists some
      predecessor P such that both static edges (P -> A, LABEL) and
      (A -> B, LABEL) exist (i.e. two consecutive edges with same LABEL).
      The agent must be at P to execute that push, and B must be empty at the
      moment of the push. For heuristic computation we:
        - construct the abstract "push graph" where A -> B is an edge when a
          label-L chain (P->A, A->B with same L) exists for some P,
        - record the set of possible P positions enabling each A->B push.
    - The heuristic is optimistic regarding dynamic blocking by other boxes:
      individual boxes are routed to their respective goals ignoring temporary
      blocking by other boxes. This keeps the heuristic inexpensive and fast,
      while remaining informative.

    Heuristic Initialization
    ------------------------
    In the constructor we extract from the task static facts:
      - All labeled static adjacency edges (h80z8tvp ...).
      - Build:
          * push_triples: mapping from a box-location A to a list of triples
            (pred_P, A, dest_B) meaning "being at P allows pushing box from A to B".
          * push_graph: adjacency mapping A -> set(dest_B) capturing possible
            box-moves (ignoring which P is used).
          * agent_adjacency: mapping loc -> set(neighbor) used to compute agent
            walking distances (movement edges irrespective of dynamic emptiness).
      - Parse goal facts and remember desired location for each box in the goals.
    These static structures are reused for every state evaluation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner state:
      1. Quick goal check: If all goal facts are already satisfied in the state,
         return 0.
      2. Parse the state to recover:
           - agent_loc: the unique location where (zdufuwne ...) holds.
           - box_locations: map from box -> current location by scanning
             (ldlu1rwo ...) facts.
           - empty_locations: set of locations where (qsw2k0xh ...) holds.
      3. For each box that has a goal location, compute the minimal number of
         pushes required to move the box from its current location to its goal
         by performing a BFS on the precomputed push_graph (each edge counts
         as one push). If a box's goal is unreachable in the push_graph, the
         box contributes a large finite penalty (we cap to a high constant but
         keep it finite).
      4. Sum all minimal push counts across boxes -> pushes_sum.
      5. Compute the agent's shortest walking distances (in terms of x042b0fi
         moves) from the agent's current location to all locations that are
         valid push-predecessors P for some first push of some box. The agent
         walking graph allows stepping only into currently empty locations
         (qsw2k0xh), but starts from the agent's current location even if that
         location itself is not flagged empty. We prefer push triples where
         the push destination B is currently empty, but if none exist we relax
         that requirement (optimistic estimate).
      6. Let agent_start_cost be the minimum walking distance found to any
         qualifying predecessor P (0 if agent already at such P). Add this to
         pushes_sum to obtain the final heuristic value.
    Rationale:
      - Each push corresponds to one yi0kp1zg action.
      - The agent must typically walk to a push precondition location before
        performing pushes; adding the minimal walking distance for the first
        push makes the heuristic more informative without being expensive.
      - We ignore inter-box blocking between pushes to keep the calculation
        fast; the heuristic is therefore optimistic (non-admissibility is
        allowed by the task), but usually effective in guiding greedy search.
    """

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals
        static_facts = task.static

        # Keep map of goals for boxes: box -> goal_location
        self.box_goals = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.box_goals[box] = loc

        # Build label->edges and adjacency structures from h80z8tvp static facts
        edges_by_label = {}     # label -> set of (src, dst)
        agent_adj = {}          # src -> set(dst) (ignoring emptiness; static)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp" and len(parts) >= 4:
                src, dst, label = parts[1], parts[2], parts[3]
                edges_by_label.setdefault(label, set()).add((src, dst))
                agent_adj.setdefault(src, set()).add(dst)

        # Construct push_triples: for each middle node A, list of (P,A,B)
        # whenever (P->A,label) and (A->B,label) exist for same label.
        self.push_triples = {}  # A -> list of (P, A, B)
        self.push_graph = {}    # A -> set(B) (push adjacency ignoring P)
        for label, edges in edges_by_label.items():
            # build edges indexed by source/destination to quickly find chains
            by_src = {}
            by_dst = {}
            for (u, v) in edges:
                by_src.setdefault(u, set()).add(v)
                by_dst.setdefault(v, set()).add(u)
            # for each chain P -> A -> B with same label
            for A in by_dst:
                preds = by_dst.get(A, set())
                succs = by_src.get(A, set())
                if preds and succs:
                    for P in preds:
                        for B in succs:
                            self.push_triples.setdefault(A, []).append((P, A, B))
                            self.push_graph.setdefault(A, set()).add(B)

        # Keep agent adjacency for movement (static)
        self.agent_adjacency = agent_adj

        # Large finite cap for unreachable push distances (keeps heuristic finite)
        self.UNREACHABLE_PUSH_COST = 1000

    def __call__(self, node):
        state = node.state

        # 1) If already at goal, return 0
        if self.goals <= state:
            return 0

        # 2) Parse state: empties, boxes, agent location
        empty_locations = set()
        box_locations = {}  # box -> loc
        agent_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "qsw2k0xh" and len(parts) >= 2:
                empty_locations.add(parts[1])
            elif parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                box_locations[box] = loc
            elif parts[0] == "zdufuwne" and len(parts) >= 2:
                agent_loc = parts[1]

        # If agent location not found, be robust and set to None (treat distances accordingly)
        if agent_loc is None:
            agent_loc = None  # distances will be effectively infinite; but heuristic remains finite

        # 3) For each box with a goal compute minimal pushes (BFS on push_graph)
        total_pushes = 0
        for box, goal_loc in self.box_goals.items():
            start = box_locations.get(box)
            # If box already at goal, no pushes needed
            if start == goal_loc:
                continue
            # BFS over push_graph nodes (locations)
            visited = set()
            dq = deque()
            if start is not None:
                visited.add(start)
                dq.append((start, 0))
            found = False
            while dq:
                curr, dist = dq.popleft()
                # neighbors are possible box destinations after one push
                for nbr in self.push_graph.get(curr, ()):
                    if nbr in visited:
                        continue
                    if nbr == goal_loc:
                        total_pushes += (dist + 1)
                        found = True
                        dq.clear()
                        break
                    visited.add(nbr)
                    dq.append((nbr, dist + 1))
            if not found:
                # If unreachable in the abstract push graph, add a large finite penalty
                total_pushes += self.UNREACHABLE_PUSH_COST

        # 4) Compute agent distances (walking moves) to all locations reachable by walking
        agent_dist = {}  # loc -> distance
        if agent_loc is not None:
            dq = deque()
            agent_dist[agent_loc] = 0
            dq.append(agent_loc)
            while dq:
                u = dq.popleft()
                d = agent_dist[u]
                for v in self.agent_adjacency.get(u, ()):
                    # movement is only possible into currently empty locations
                    if v not in empty_locations:
                        continue
                    if v not in agent_dist:
                        agent_dist[v] = d + 1
                        dq.append(v)

        # 5) Minimum cost to get agent to a valid push-predecessor P for any box's first push
        best_agent_start = None
        # First prefer triples where destination B is currently empty (push immediately feasible)
        for prefer_empty in (True, False):
            best = None
            for box, goal_loc in self.box_goals.items():
                start = box_locations.get(box)
                if start is None:
                    continue
                triples = self.push_triples.get(start, [])
                for (P, A, B) in triples:
                    if prefer_empty and B not in empty_locations:
                        continue
                    # The predecessor P must be reachable by the agent via walking
                    if P == agent_loc:
                        # already at required predecessor
                        best = 0
                        break
                    d = agent_dist.get(P)
                    if d is not None:
                        if best is None or d < best:
                            best = d
                if best == 0:
                    break
            if best is not None:
                best_agent_start = best
                break

        if best_agent_start is None:
            # No reachable push-predecessor found (or agent_loc is None). Optimistically
            # assume agent can teleport to a required predecessor at zero cost,
            # but to maintain some guidance we set it to 0 rather than a huge number.
            best_agent_start = 0

        heuristic_value = total_pushes + best_agent_start
        # return integer heuristic
        return int(heuristic_value)