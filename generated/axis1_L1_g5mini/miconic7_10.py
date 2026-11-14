from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return a list of tokens in a PDDL fact string, without surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z). The domain models a single
    mobile "agent" (represented by the predicate (omggbn9z LOCATION)) that can move across
    locations connected by directed links (acovodrf). The agent can perform two person-oriented
    service actions:
      - diwlmgg9: when at a location that satisfies (kut2ksnk PERSON LOCATION) it produces
                  (f2xo7hb6 PERSON) (a "picked-up" marker).
      - y7nik9q2: when at a location that satisfies (klw11kjf PERSON LOCATION) and the person
                  has (f2xo7hb6 PERSON), it produces (a7sb56om PERSON) (the goal predicate).
    To achieve (a7sb56om PERSON) for a person who does not have (f2xo7hb6), the agent must
    visit a location with kut2ksnk (pick up), then visit a location with klw11kjf (drop).
    If the person already has (f2xo7hb6) the agent only needs to reach a klw11kjf location
    for that person and apply the drop action.

    The heuristic estimates the required number of actions (moves + service actions) to reach
    the goal(s) by computing shortest-path distances on the static location graph (acovodrf)
    and summing a minimal per-person service cost. This is an efficient, informative estimate:
    it uses graph distances (BFS) to count required move actions and counts the two service
    actions (diwlmgg9 and y7nik9q2) explicitly where needed.

    Assumptions
    -----------
    - There is a single mobile agent indicated by exactly one fact of the form (omggbn9z LOCATION)
      in a given state (if none exist the state is considered unreachable).
    - Static facts include directed connectivity facts (acovodrf LOC1 LOC2) representing
      allowed moves of the mobile agent (edges LOC1 -> LOC2).
    - For each PERSON there may be one or more locations listed for kut2ksnk (pickup) and/or
      klw11kjf (drop). The heuristic will choose the best (minimum-cost) pair of pickup/drop
      locations for that person.
    - If a person already has (a7sb56om PERSON) in the given state, no cost for that person is added.
    - If a person is already in the "picked-up" state (f2xo7hb6 PERSON is true) the pickup step
      is not required.
    - If connectivity or required pickup/drop locations are missing/unreachable, a large finite
      penalty is returned (heuristic remains finite).

    Heuristic Initialization
    ------------------------
    - Extract the set of goal person identifiers from task.goals. Each goal is expected to be
      of the form "(a7sb56om PERSON)".
    - Extract the static directed graph of locations from task.static using facts of the form
      "(acovodrf LOC1 LOC2)". Build an adjacency dictionary for BFS shortest-path queries.
    - Store any klw11kjf facts that appear in static (person -> set(locations)). Note: klw11kjf
      may also appear in dynamic state facts; both are considered when computing costs.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node (state), compute the heuristic as follows:

    1. If the node.state already satisfies all goals, return 0.

    2. Find the current location of the mobile agent by locating the (omggbn9z LOC) fact in node.state.
       If no such fact exists, return a large penalty (state considered not reachable by this heuristic).

    3. For each person P that must satisfy (a7sb56om P) but does not yet:
       a. Determine if (f2xo7hb6 P) is already true in node.state (person already "picked up").
       b. Collect possible drop locations: all LOC where (klw11kjf P LOC) appears in the union of
          node.state and static facts.
       c. Collect possible pickup locations: all LOC where (kut2ksnk P LOC) appears in the union of
          node.state and static facts.
       d. Compute minimal action cost for P, assuming the agent starts at the current agent location:
          - If f2xo7hb6 P is true: cost_P = dist(agent_loc, best_drop_loc) + 1
            (move(s) to drop location + one y7nik9q2 action).
          - Else: cost_P = dist(agent_loc, best_pick_loc) + 1   (diwlmgg9)
                             + dist(best_pick_loc, best_drop_loc) + 1  (y7nik9q2)
          - Choose pickup/drop locations that minimize the above cost. Distances are BFS shortest-paths
            on the static adjacency graph. If a required location or path is missing/unreachable, we
            assign a large per-person penalty (but finite).

    4. Sum the per-person minimal costs to obtain the heuristic value. The sum is a straightforward,
       informative approximation. It may overcount agent travel between serving different people
       (i.e., it treats each person as if the agent started at the current agent location), but it
       remains computationally cheap and typically provides good guidance to greedy best-first search.

    Implementation notes
    --------------------
    - BFS is used for shortest-path distances on the static location graph.
    - A large finite penalty (e.g., 1000) is used for unreachable subgoals to ensure finiteness.
    """

    def __init__(self, task):
        # Goals: collect persons that must get (a7sb56om PERSON)
        self.goal_persons = []
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "a7sb56om" and len(parts) >= 2:
                self.goal_persons.append(parts[1])

        # Build adjacency map for locations from static acovodrf facts (directed edges).
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "acovodrf" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, set()).add(dst)

        # Store any klw11kjf facts that are static (person -> set(locations)).
        self.static_klw = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "klw11kjf" and len(parts) >= 3:
                person, loc = parts[1], parts[2]
                self.static_klw.setdefault(person, set()).add(loc)

        # A reasonably large finite penalty for unreachable situations.
        self.INF_COST = 1000

    def bfs_distance(self, start, goal):
        """Return the shortest number of moves (edges) from start to goal using BFS.
           If start == goal return 0. If no path exists, return None.
        """
        if start is None or goal is None:
            return None
        if start == goal:
            return 0
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == goal:
                    return d + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, d + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is 0.
        # task.goal_reached is not accessible here, but compare facts directly:
        all_goals_met = True
        for p in self.goal_persons:
            if f"(a7sb56om {p})" not in state:
                all_goals_met = False
                break
        if all_goals_met:
            return 0

        # Find current agent location from (omggbn9z LOC) in the state.
        agent_locs = [get_parts(fact)[1] for fact in state if get_parts(fact)[0] == "omggbn9z"]
        if not agent_locs:
            # No agent location in state - treat as very costly/unreachable
            return self.INF_COST
        agent_loc = agent_locs[0]

        # For efficient lookup consider static + state for klw facts and kut facts.
        combined_facts = set(self.static_klw.keys())  # just to use variable; we'll build maps below
        # Build maps from PERSON -> set(locations) for klw11kjf and kut2ksnk using both state and static
        klw_map = {}
        # start with static klw
        for person, locs in self.static_klw.items():
            klw_map.setdefault(person, set()).update(locs)
        # add any klw facts present in the state
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "klw11kjf" and len(parts) >= 3:
                klw_map.setdefault(parts[1], set()).add(parts[2])

        kut_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "kut2ksnk" and len(parts) >= 3:
                kut_map.setdefault(parts[1], set()).add(parts[2])
        # Also check static facts for kut2ksnk in case they exist there
        for fact in getattr(node, "static", ()):  # safety; should not be in node, but ignore if missing
            parts = get_parts(fact)
            if parts and parts[0] == "kut2ksnk" and len(parts) >= 3:
                kut_map.setdefault(parts[1], set()).add(parts[2])
        # (Note: task.static was parsed in __init__, but node does not provide static. Most kut facts are
        # in the dynamic state; we already included static klw earlier.)

        total_cost = 0

        for person in self.goal_persons:
            # Skip if already satisfied at this state
            if f"(a7sb56om {person})" in state:
                continue

            # If person already in picked-up state
            picked_fact = f"(f2xo7hb6 {person})"
            picked_up = picked_fact in state

            # Candidate drop locations (where klw holds)
            drop_locs = klw_map.get(person, set())
            # Candidate pickup locations (where kut holds)
            pick_locs = kut_map.get(person, set())

            best_cost = None

            if picked_up:
                # Need only to reach a drop location and perform y7nik9q2 (1 action)
                if not drop_locs:
                    # No drop location known -> unreachable for this person
                    best_cost = self.INF_COST
                else:
                    for dl in drop_locs:
                        dist = self.bfs_distance(agent_loc, dl)
                        if dist is None:
                            continue
                        cost = dist + 1  # moves + y7nik9q2
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                    if best_cost is None:
                        best_cost = self.INF_COST
            else:
                # Need pickup (diwlmgg9) at some pickup location, then travel to drop and y7nik9q2
                if not pick_locs:
                    # If no pickup locations and not already picked, unreachable
                    best_cost = self.INF_COST
                else:
                    if not drop_locs:
                        # No drop locations -> unreachable
                        best_cost = self.INF_COST
                    else:
                        # try all combinations pick_loc x drop_loc and choose minimal cost
                        for pl in pick_locs:
                            d1 = self.bfs_distance(agent_loc, pl)
                            if d1 is None:
                                continue
                            for dl in drop_locs:
                                d2 = self.bfs_distance(pl, dl)
                                if d2 is None:
                                    continue
                                cost = d1 + 1 + d2 + 1  # moves to pick + diwlmgg9 + moves to drop + y7nik9q2
                                if best_cost is None or cost < best_cost:
                                    best_cost = cost
                        if best_cost is None:
                            best_cost = self.INF_COST

            total_cost += best_cost
            # Early exit if cost already extremely large to avoid unnecessary work
            if total_cost >= self.INF_COST:
                return self.INF_COST

        # Return the sum of minimal per-person costs (finite).
        return total_cost