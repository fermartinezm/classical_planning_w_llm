from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw) domain.

    The goal facts in this domain are of the form (ne6sl557 <ris>), where <ris>
    is an object of type wqid8mog. To make (ne6sl557 <ris>) true an operator
    f7y6nzq3 must be applied. f7y6nzq3 requires:
      - ndg963qz <du>                 (a du770mtw object <du> with the ndg flag)
      - of6wn8m4 <ris>                (<ris> must currently be "available")
      - qkdx0xe2 <ris> <ts>           (<ris> is at a location <ts> of type mz6r1d30)
      - qkdx0xe2 <pt> <ts>            (a pt4b08k4 object <pt> must be at the same location <ts>)
      - fqyrzl30 <pt> <du>            (<pt> must be linked to the <du> object)
    The link fqyrzl30 is created by jql8txb3 when both <du> and <pt> are at the same
    <ts>. The pt can be moved along static lpmls56r edges using nm6lzp2x; each nm6lzp2x
    application moves a pt one step along a directed lpmls56r edge.

    This heuristic therefore estimates, for every goal <ris> not yet satisfied,
    the cheapest sequence:
      - move some pt along lpmls56r edges until it reaches the location of <ris>,
      - (if necessary) apply jql8txb3 to link that pt to a du located at the same mz
      - apply f7y6nzq3 to make (ne6sl557 <ris>) true.

    Assumptions
    -----------
    - lpmls56r static facts define directed edges between mz6r1d30 nodes.
      Each nm6lzp2x action moves a pt one directed step along such an edge.
    - du objects that can be used to satisfy a goal must have ndg963qz true
      in the current state and must be qkdx0xe2-located at the same mz as the ris.
    - Candidate pts are those objects that currently have a qkdx0xe2 location
      but are not themselves ndg du objects or goal ris objects.
    - If a direct way to satisfy a particular goal cannot be found in the current
      state (e.g., no du with ndg at the goal's location, or no pt can reach
      that location), the heuristic assigns a moderate finite penalty for that goal.
      This keeps the heuristic finite while strongly discouraging those branches.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (task.goals).
    - Parse static facts from the task to build a directed adjacency map
      for mz6r1d30 nodes from all (lpmls56r A B) facts. This is used by BFS to
      estimate how many nm6lzp2x actions are required to move a pt from one mz
      node to another.
    - No other static information is necessary for the heuristic.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given planner node/state:
    1. If all goal facts are already satisfied in the state return 0.
    2. Parse qkdx0xe2 facts to get the current mz location for each object
       that is qkdx0xe2-located.
    3. Determine which du objects are available (those with ndg963qz true)
       and where they are located (via qkdx0xe2).
    4. Identify the set of candidate pt objects as those objects that have a
       qkdx0xe2 location but are neither du objects (ndg) nor goal ris objects.
    5. For each unsatisfied goal (ne6sl557 <ris> not in state):
       a. Determine the mz location ts of <ris> (via qkdx0xe2).
       b. Find du objects that have ndg true and are qkdx0xe2-located at the same ts.
          These du objects are usable locations where f7y6nzq3 could be applied.
       c. For each candidate pt, compute the shortest directed distance (BFS)
          from the pt's current mz to ts using the lpmls56r adjacency graph.
          Each edge corresponds to a single nm6lzp2x action.
       d. If some pt can reach ts in d moves:
          - If (fqyrzl30 pt du) already holds in the state (pt already linked to du)
            estimated actions = d + 1 (d nm moves + 1 f7y6nzq3).
          - Otherwise estimated actions = d + 2 (d nm moves + 1 jql8txb3 + 1 f7y6nzq3).
       e. Choose the minimal estimated actions across all du/pt pairs for that ris.
       f. If ris has no qkdx0xe2 location, or no du at its location, or no pt can reach
          its location, add a finite penalty (e.g. 100) for that ris to keep the estimate finite.
    6. Sum the per-goal estimates to produce the final heuristic value. This sum is
       deliberately non-admissible but tends to correlate with the number of
       required actions and is cheap to compute.
    """

    def __init__(self, task):
        # store the goal set as provided by the planner
        self.goals = set(task.goals)
        # Build directed adjacency for mz nodes from static lpmls56r facts.
        # Each static fact is a string like '(lpmls56r a b)'.
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "lpmls56r" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adjacency.setdefault(a, []).append(b)

    def _bfs_distance(self, start, target):
        """Return shortest directed distance from start mz to target mz using adjacency.
           Return None if target unreachable. start==target -> 0."""
        if start == target:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nbr in self.adjacency.get(node, []):
                if nbr == target:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # 1) Heuristic is zero only if all goal facts are satisfied.
        if self.goals <= state:
            return 0

        # Parse facts from the current state.
        # Build quick-access maps:
        # - qk_map: object -> mz (from qkdx0xe2 facts)
        # - fq_links: set of (pt, du) from fqyrzl30 facts
        # - ndg_set: set of du objects that currently have ndg963qz true
        # - of6_set: set of ris objects that are currently of6wn8m4 true (availability)
        qk_map = {}
        fq_links = set()
        ndg_set = set()
        of6_set = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, mz = parts[1], parts[2]
                qk_map[obj] = mz
            elif pred == "fqyrzl30" and len(parts) >= 3:
                pt, du = parts[1], parts[2]
                fq_links.add((pt, du))
            elif pred == "ndg963qz" and len(parts) >= 2:
                ndg_set.add(parts[1])
            elif pred == "of6wn8m4" and len(parts) >= 2:
                of6_set.add(parts[1])

        # Identify goal objects that must become ne6sl557.
        goal_objs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ne6sl557" and len(parts) >= 2:
                goal_objs.append(parts[1])
            else:
                # Generic fallback: treat any goal fact '(pred obj)' as a single-arg goal
                if len(parts) == 2:
                    goal_objs.append(parts[1])

        # Candidate pts: objects that have qkdx0xe2 but are not ndg du's and not the goals themselves.
        candidates_pt_objs = [o for o in qk_map.keys() if o not in ndg_set and o not in goal_objs]

        total_cost = 0
        LARGE_PENALTY = 100  # finite penalty for goals that look unreachable in current state

        for ris in goal_objs:
            # If this goal is already satisfied in state, skip.
            if f"(ne6sl557 {ris})" in state:
                continue

            # Check ris availability: f7y6nzq3 requires of6wn8m4 ris.
            if ris not in of6_set:
                # If not currently available, penalize but still finite.
                total_cost += LARGE_PENALTY
                continue

            # ris must have a qkdx0xe2 location ts
            ris_mz = qk_map.get(ris)
            if ris_mz is None:
                total_cost += LARGE_PENALTY
                continue

            # find du objects that have ndg and are located at same mz as ris
            usable_du = [du for du in ndg_set if qk_map.get(du) == ris_mz]

            if not usable_du:
                # No du at ris location => cannot apply f7 now; penalize.
                total_cost += LARGE_PENALTY
                continue

            # For each usable du, find minimal cost to produce the f7 preconditions:
            best_for_ris = None
            for du in usable_du:
                # For each candidate pt, compute distance to ris_mz
                for pt in candidates_pt_objs:
                    pt_mz = qk_map.get(pt)
                    if pt_mz is None:
                        continue
                    dist = self._bfs_distance(pt_mz, ris_mz)
                    if dist is None:
                        continue
                    # If fqyrzl30 (pt, du) already exists, jql8txb3 isn't needed.
                    if (pt, du) in fq_links:
                        cost = dist + 1  # nm moves + f7
                    else:
                        cost = dist + 2  # nm moves + jql + f7
                    if best_for_ris is None or cost < best_for_ris:
                        best_for_ris = cost

            if best_for_ris is None:
                # No pt can reach the ris_mz for any usable du.
                total_cost += LARGE_PENALTY
            else:
                total_cost += best_for_ris

        # Heuristic is sum of per-goal estimated actions (finite, integer).
        return int(total_cost)