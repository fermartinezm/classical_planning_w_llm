from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import heapq

def get_parts(fact):
    """Remove surrounding parentheses and split into tokens."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    This heuristic estimates the number of actions required to achieve the goal
    placements of boxes (ldlu1rwo facts). The domain contains:
      - zdufuwne <loc> : the agent is at location <loc>
      - ldlu1rwo <box> <loc> : box is at location <loc>
      - qsw2k0xh <loc> : location <loc> is clear (no box)
      - h80z8tvp <from> <to> <token> : a directed adjacency with an associated token.

    Intuitively, pushing a box one step requires a triple of locations
    Z -> Y -> C where the same token labels both edges (Z->Y and Y->C).
    The push action moves the box from Y to C if the agent stands at Z.
    The heuristic reasons about sequences of such pushes.

    Assumptions
    -----------
    - Goal facts are only of the form (ldlu1rwo <box> <loc>).
    - The static facts (h80z8tvp) describe valid directed edges and tokens.
    - The current state's qsw2k0xh facts indicate which squares the agent can
      move into without pushing a box.
    - We ignore dynamic interactions between simultaneous box moves (we relax
      collisions) — this makes the heuristic fast but not necessarily admissible.
    - If no push path exists for a box to its goal according to the static
      push-graph, a large finite penalty is returned for that box (so the
      heuristic remains finite).

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
      - Extracts the goal mapping (box -> goal_location) from task.goals.
      - Extracts the directed adjacency graph from task.static for predicate
        h80z8tvp. Each adjacency stores (from -> (to, token)).
      - Builds:
          * outgoing[from] = list of (to, token)
          * incoming[to] = list of (from, token)
          * push_successors[loc] = list of tuples (to, set_of_predecessors, token)
            meaning a box at loc can be pushed to 'to' if the agent stands at any
            predecessor in the set_of_predecessors (all triples share the same token).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
      1. If all goal facts are satisfied in the state, return 0.
      2. Parse the state to obtain:
         - agent_loc : the single location with zdufuwne
         - box_locs  : mapping box -> location from ldlu1rwo facts
         - clear_locs: set of locations that currently have qsw2k0xh true
      3. Compute the set of reachable agent positions without pushing any box.
         - The agent can step from A to B if there exists an edge h80z8tvp A B token
           and B is currently clear (qsw2k0xh in state). This yields reachable_set.
      4. For each box that is not at its goal:
         - Compute a minimal-cost push-plan on the relaxed push-graph:
             * Nodes are locations; edges exist from Y -> C if there is at least
               one Z with (Z -> Y token) and (Y -> C token) (same token).
             * The cost of each push edge is 1 (one push).
             * Additionally, to perform the very first push of a box (from its
               current location), the agent must be able to stand on one of the
               predecessors Z for that first push. If none of those predecessors
               are currently reachable by the agent (in reachable_set), we add
               an extra small penalty (1) to the cost of the first push to
               reflect the need to move the agent (approximation).
         - The minimal such cost (pushes + possible initial agent penalty) is
           computed using Dijkstra (small graph).
         - If no path exists in the push-graph, we assign a large finite penalty
           for that box (1000) so the heuristic remains finite.
      5. The heuristic value is the sum of the per-box estimated costs. This
         encourages solving the closest/more accessible boxes first.

    Rationale / Comments
    --------------------
    - The heuristic focuses on push-distance (number of pushes) because pushes
      are the key resource in Sokoban-like domains and they change box locations.
    - Agent movement is approximated: we only check whether the agent can reach
      a valid push-origin (a predecessor) without pushing boxes. If not, we add
      a small penalty. This is cheap to compute and gives reasonable guidance.
    - The heuristic ignores inter-box blocking in the relaxed graph. That speeds
      up computation and tends to underestimate difficulty (i.e., it is
      optimistic) but provides good guidance for greedy search in practice.
    """

    def __init__(self, task):
        # Store goals as a dict: box -> desired_location
        self.goal_locations = {}
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ldlu1rwo":
                # goal fact format: (ldlu1rwo box loc)
                _, box, loc = parts
                self.goal_locations[box] = loc

        # Build adjacency graphs from static h80z8tvp facts
        self.outgoing = defaultdict(list)  # from -> list of (to, token)
        self.incoming = defaultdict(list)  # to   -> list of (from, token)
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "h80z8tvp":
                _, a, b, token = parts
                self.outgoing[a].append((b, token))
                self.incoming[b].append((a, token))

        # Build push-successor structure:
        # for each location Y, for each incoming (Z->Y, token) and outgoing (Y->C, token),
        # record that a box at Y can be pushed to C with predecessor Z (agent at Z).
        # We aggregate predecessors per (Y->C, token).
        self.push_successors = defaultdict(list)
        # temporary structure to aggregate predecessors by (Y, C, token)
        tmp = defaultdict(lambda: defaultdict(set))  # tmp[Y][(C,token)] -> set(predecessors)
        for y, incomings in self.incoming.items():
            # for each incoming (z, token) to y
            for (z, token_in) in incomings:
                # for each outgoing (y->c, token_out) from y
                for (c, token_out) in self.outgoing.get(y, []):
                    if token_in == token_out:
                        tmp[y][(c, token_out)].add(z)
        # convert tmp to push_successors list entries
        for y, mapping in tmp.items():
            for (c, token), preds in mapping.items():
                self.push_successors[y].append((c, frozenset(preds), token))

        # A large finite penalty used if no push path exists for a box.
        self.INF_PENALTY = 1000

    def __call__(self, node):
        state = node.state

        # If goals are already satisfied, heuristic is 0
        if hasattr(self, "goal_locations"):
            # Build goal facts set to check quickly
            goal_facts = set()
            for box, loc in self.goal_locations.items():
                goal_facts.add(f"(ldlu1rwo {box} {loc})")
            if goal_facts <= state:
                return 0

        # Parse state: agent location, box locations, clear locations
        agent_loc = None
        box_locations = {}  # box -> loc
        clear_locations = set()  # locations that currently have qsw2k0xh
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "zdufuwne":
                # (zdufuwne loc)
                if len(parts) >= 2:
                    agent_loc = parts[1]
            elif pred == "ldlu1rwo":
                # (ldlu1rwo box loc)
                if len(parts) >= 3:
                    box, loc = parts[1], parts[2]
                    box_locations[box] = loc
            elif pred == "qsw2k0xh":
                # (qsw2k0xh loc)
                if len(parts) >= 2:
                    clear_locations.add(parts[1])

        # Safety: if agent location missing, return large heuristic
        if agent_loc is None:
            return sum(self.INF_PENALTY for _ in self.goal_locations)

        # Compute reachable agent positions without pushing boxes.
        # Moves allowed: from A to B if there exists h80z8tvp A B token and B is clear.
        reachable = set()
        queue = deque([agent_loc])
        reachable.add(agent_loc)
        while queue:
            u = queue.popleft()
            for (v, token) in self.outgoing.get(u, []):
                # agent can step into v only if v is currently clear
                if v not in reachable and v in clear_locations:
                    reachable.add(v)
                    queue.append(v)

        # For each box that is not at its goal, compute minimal push-cost to its goal
        total_cost = 0
        for box, goal_loc in self.goal_locations.items():
            cur_loc = box_locations.get(box)
            # If box not in state or already at goal
            if cur_loc is None:
                # If box is missing, treat as heavy penalty
                total_cost += self.INF_PENALTY
                continue
            if cur_loc == goal_loc:
                continue

            # Dijkstra on push-graph where:
            # - edges exist from Y to C for each push_successors[Y] entry (C, predecessors, token)
            # - cost for any edge is 1 (one push)
            # - but for the very first edge from the box's current location, if none of the
            #   predecessors for that Y->C edge are reachable by the agent (in 'reachable'),
            #   add an extra penalty 1 for the agent to move to a pushing position (approx).
            visited_cost = {}  # loc -> best_cost found
            heap = []
            # Start at current box location, cost 0
            heapq.heappush(heap, (0, cur_loc))
            found_cost = None
            while heap:
                cost, loc = heapq.heappop(heap)
                if loc in visited_cost and visited_cost[loc] <= cost:
                    continue
                visited_cost[loc] = cost
                if loc == goal_loc:
                    found_cost = cost
                    break
                # expand push edges from loc
                for (c, preds, token) in self.push_successors.get(loc, []):
                    # Determine penalty for this edge:
                    # If we're expanding from the start (loc == cur_loc), check whether any predecessor
                    # is currently reachable. If none is reachable, add +1 penalty.
                    extra = 0
                    if loc == cur_loc:
                        if preds.isdisjoint(reachable):
                            extra = 1
                    new_cost = cost + 1 + extra
                    # push if better
                    if c not in visited_cost or visited_cost[c] > new_cost:
                        heapq.heappush(heap, (new_cost, c))

            if found_cost is None:
                # No push path found; assign a large but finite penalty
                total_cost += self.INF_PENALTY
            else:
                total_cost += found_cost

        return total_cost