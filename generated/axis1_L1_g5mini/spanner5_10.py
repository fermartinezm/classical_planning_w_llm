from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the tokens of a fact string like '(pred a b)' -> ['pred','a','b']."""
    return fact[1:-1].split()

INF = 10**6

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (e2zu8y93).  The domain
    models an agent (person) that can move along locations, pick up spanners
    and use them to install nuts.  A nut is considered installed when the
    predicate (g59iwyxf <nut>) holds.  Installing a nut consumes a usable
    spanner (predicate g0brvc06) and requires the person to hold that spanner
    and be co-located with the nut.  Movement along locations is given by the
    static relation (kxokb5hx <loc1> <loc2>) and each single edge traversal is
    performed by the move action.

    The heuristic estimates the number of actions (moves, pick-ups, installs)
    remaining to achieve all goal nut installations.

    Assumptions
    -----------
    - Each install of a nut requires exactly one install action (kjsqfni8)
      and one pick-up action (moi6wnfu) unless the person already holds a
      usable spanner.
    - Spanners are usable iff (g0brvc06 <spanner>) is true. After an install
      the used spanner becomes unusable; hence each usable spanner can be used
      at most once.
    - The planner has at least one person-like entity. Persons are detected as
      objects that appear in u4srx1nb but are not classified as spanners or
      nuts. If no such object is detected, one arbitrary empc2xti object is
      chosen as the agent (fallback).
    - Movement cost is measured in the number of directed kxokb5hx edges
      traversed. Each movement along one kxokb5hx edge corresponds to one
      action (n9sbfmrs). Picking up (moi6wnfu) and installing (kjsqfni8) each
      cost one action.
    - The heuristic is not required to be admissible.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - The directed adjacency graph of locations from static facts of the form
      (kxokb5hx loc1 loc2).
    - The set of goal nut objects (from goal atoms (g59iwyxf <nut>)) so we
      know which nuts must be installed.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state we compute an optimistic yet informed sequential cost by
    simulating a greedy schedule for installing the remaining goal nuts:

    1. Parse the current state:
       - Locations of all objects that are described by (u4srx1nb <obj> <loc>).
       - Which spanners are currently usable (g0brvc06).
       - Which nuts are already installed (g59iwyxf) and which are still
         uninstalled (goal nuts that lack g59iwyxf).
       - Which entity (if any) currently holds which spanner via yud7t3vb.

    2. If the state already satisfies all goals, the heuristic returns 0.

    3. Compute shortest-path distances between locations on the directed graph
       defined by kxokb5hx. Distances are computed on demand using BFS; if no
       path exists between two locations, the distance is treated as large.

    4. Greedy assignment simulation:
       - Start from a chosen "agent" (person) location.
       - For each uninstalled goal nut, select a usable spanner (not yet
         assigned) that minimizes the incremental action cost to perform:
            * move sequence (either move agent to spanner then to nut, or
              move spanner to agent then agent to nut), plus
            * pick-up (if the agent does not already hold that spanner), and
            * install.
       - We consider both moving the agent to the spanner (then pick and
         move to nut) and moving the spanner to the agent (then pick and
         move to nut) and choose the cheaper of the two as the cost for that
         nut; if the agent already holds a usable spanner, only the move-to-nut
         + install option is considered for that spanner.
       - After assigning a spanner to a nut we:
            * mark the spanner as used,
            * update the agent's location to the nut's location (the agent
              remains at the nut after installation),
            * accumulate the action costs.
       - If no usable spanner is reachable for a nut (or no spanners remain),
         we still return a finite large estimate (INF-scale) but not infinite
         so the heuristic remains finite for the planner.

    5. The sum of the incremental costs for all remaining nuts is returned.

    Notes on costs modeled:
    - Each kxokb5hx edge traversal by the agent or by a movable object counts
      as one move action.
    - Picking up a spanner (moi6wnfu) costs 1.
    - Installing a nut (kjsqfni8) costs 1.
    - If the agent already holds a usable spanner and is at the nut's
      location, the cost to complete that nut is 1 (install).
    """

    def __init__(self, task):
        # Extract static adjacency graph for locations from kxokb5hx facts.
        static_facts = task.static
        self.adj = {}  # directed adjacency: loc -> set(neighbor_locs)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)

        # Extract goal nuts (objects that must satisfy g59iwyxf).
        self.goal_nuts = set()
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "g59iwyxf" and len(parts) >= 2:
                self.goal_nuts.add(parts[1])

    def _bfs_distance(self, start, goal):
        """Return shortest directed distance (edges) from start to goal or INF."""
        if start == goal:
            return 0
        if start not in self.adj:
            return INF
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, d = dq.popleft()
            for nb in self.adj.get(node, ()):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    dq.append((nb, d + 1))
        return INF

    def __call__(self, node):
        state = node.state

        # If already goal, heuristic is zero.
        if self.goal_nuts <= state:
            return 0

        # Parse state facts.
        loc_of = {}          # object -> location (for objects with u4srx1nb)
        usable_spanners = set()   # spanners with g0brvc06
        installed_nuts = set()    # nuts already installed g59iwyxf
        uninstalled_goal_nuts = set()  # goal nuts not yet installed
        spanner_held_by = {}   # spanner -> holder (yud7t3vb holder spanner)
        subjects_in_u4 = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "u4srx1nb" and len(parts) >= 3:
                subj, loc = parts[1], parts[2]
                loc_of[subj] = loc
                subjects_in_u4.add(subj)
            elif pred == "g0brvc06" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "g59iwyxf" and len(parts) >= 2:
                installed_nuts.add(parts[1])
            elif pred == "yud7t3vb" and len(parts) >= 3:
                holder, sp = parts[1], parts[2]
                spanner_held_by[sp] = holder

        # Determine which goal nuts still need installation.
        for nut in self.goal_nuts:
            if nut not in installed_nuts:
                uninstalled_goal_nuts.add(nut)

        # Identify spanners that are (a) usable and (b) either at a location or held.
        available_spanners = set()
        for s in usable_spanners:
            # Exclude spanners that are usable but somehow not present in world (shouldn't happen).
            # We'll keep spanners usable only if they are either at some location or held.
            if s in loc_of or s in spanner_held_by:
                available_spanners.add(s)

        # Identify person-like entities: those in u4srx1nb subjects but not classified as spanner or nut.
        spanners = set(usable_spanners)  # conservative superset
        nuts = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] in ("bu1om0m1", "g59iwyxf") and len(parts) >= 2:
                nuts.add(parts[1])
        persons = subjects_in_u4 - spanners - nuts
        if not persons:
            # Fallback: pick any subject present in u4srx1nb (may include spanners/nuts
            # but we need an agent to simulate). This is fallback to keep heuristic finite.
            if subjects_in_u4:
                persons = {next(iter(subjects_in_u4))}
            else:
                # No u4srx1nb facts at all (degenerate), return an estimate based on number of nuts.
                return len(uninstalled_goal_nuts) * 3

        # Choose one person to simulate (greedy). If multiple, pick arbitrary (first).
        person = next(iter(persons))
        person_loc = loc_of.get(person, None)

        # If person has a usable spanner held, note it.
        person_held_spanner = None
        for sp, holder in spanner_held_by.items():
            if holder == person and sp in available_spanners:
                person_held_spanner = sp
                break

        total_cost = 0

        # Greedily schedule installs: for each remaining nut choose best spanner currently available.
        # We simulate sequential actions: after installing nut, agent will be at nut's location and the
        # chosen spanner is consumed.
        remaining_spanners = set(available_spanners)
        remaining_nuts = set(uninstalled_goal_nuts)

        # If no usable spanners exist but nuts remain, return a large finite estimate.
        if not remaining_spanners and remaining_nuts:
            # return a pessimistic but finite number proportional to remaining nuts
            return len(remaining_nuts) * INF // 10

        # Iterate until all nuts simulated or no spanners left.
        while remaining_nuts:
            if not remaining_spanners:
                # No spanners left; return large estimate.
                total_cost += len(remaining_nuts) * (INF // 10)
                break

            # For each nut and each candidate spanner compute incremental cost and pick minimal.
            best = None  # tuple (cost, nut, spanner, resulting_person_loc, spanner_was_held_by_person_bool)
            for nut in remaining_nuts:
                nut_loc = loc_of.get(nut, None)
                if nut_loc is None:
                    # If nut has no known location, penalize strongly but keep finite.
                    nut_loc = None

                for sp in list(remaining_spanners):
                    # spanner may be at a location or held by someone.
                    sp_loc = loc_of.get(sp, None)
                    sp_holder = spanner_held_by.get(sp, None)
                    # If spanner is held by someone else (not our simulated person), we conservatively treat it as unreachable
                    # unless that holder is in persons set (we don't simulate moving other agents).
                    if sp_holder and sp_holder != person:
                        # Skip this spanner option (we cannot easily get it).
                        continue

                    # Option 1: agent moves to spanner, pick, move to nut, install.
                    # Distances: person -> sp_loc, sp_loc -> nut_loc
                    d_p_sp = self._bfs_distance(person_loc, sp_loc) if (person_loc is not None and sp_loc is not None) else INF
                    d_sp_n = self._bfs_distance(sp_loc, nut_loc) if (sp_loc is not None and nut_loc is not None) else INF
                    cost_option1 = d_p_sp + 1 + d_sp_n + 1  # moves + pick (1) + install (1)

                    # Option 2: move spanner to person, pick, move to nut, install.
                    # Distances: sp_loc -> person_loc, person_loc -> nut_loc
                    d_sp_p = self._bfs_distance(sp_loc, person_loc) if (sp_loc is not None and person_loc is not None) else INF
                    d_p_n = self._bfs_distance(person_loc, nut_loc) if (person_loc is not None and nut_loc is not None) else INF
                    cost_option2 = d_sp_p + 1 + d_p_n + 1

                    # Option 3: if person already holds this spanner (sp is held by person), cost is just move to nut + install.
                    if sp_holder == person:
                        cost_option3 = d_p_n + 1 if (person_loc is not None and nut_loc is not None) else INF
                    else:
                        cost_option3 = INF

                    # Minimal for this (nut,sp) pair:
                    local_cost = min(cost_option1, cost_option2, cost_option3)

                    # If any distance is INF (unreachable), local_cost will be INF and thus not preferred.
                    if best is None or local_cost < best[0]:
                        # resulting person location after installation will be nut_loc (if reachable) otherwise unchanged
                        resulting_loc = nut_loc if nut_loc is not None else person_loc
                        best = (local_cost, nut, sp, resulting_loc, sp_holder == person)

            if best is None:
                # No feasible assignment found (e.g., all usable spanners held by others or unreachable).
                total_cost += len(remaining_nuts) * (INF // 10)
                break

            cost, chosen_nut, chosen_sp, new_person_loc, sp_was_held = best

            # If the cost is extremely large (unreachable), break and return large estimate.
            if cost >= INF // 2:
                total_cost += len(remaining_nuts) * (INF // 10)
                break

            # Apply chosen action sequence to our simulated world:
            total_cost += int(cost)
            # Update agent location and remove chosen nut and spanner from remaining sets.
            person_loc = new_person_loc
            remaining_nuts.remove(chosen_nut)
            remaining_spanners.discard(chosen_sp)
            # If the chosen spanner was the one the person was holding, clear that flag.
            if sp_was_held:
                person_held_spanner = None

        # Return integer heuristic value.
        return int(total_cost)