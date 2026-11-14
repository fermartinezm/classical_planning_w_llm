from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split fact into parts."""
    return fact[1:-1].split()

def shortest_path_distance(adj, start, goal):
    """
    Compute shortest directed path distance from start to goal using BFS.
    Return integer distance if reachable, else return None.
    """
    if start == goal:
        return 0
    if start not in adj:
        return None
    visited = {start}
    q = deque([(start, 0)])
    while q:
        node, d = q.popleft()
        for nb in adj.get(node, ()):
            if nb == goal:
                return d + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return None

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (nuts & spanners).
    The domain models a person who can walk along directed location links,
    pick up a good spanner when co-located with it, and use a good spanner
    to tighten a nut at the nut's location. Using a spanner consumes its
    "good" property so it cannot be reused.

    The heuristic estimates the number of actions required to achieve all
    remaining g59iwyxf(goal-tightened) facts for nuts. Action costs:
      - move (n9sbfmrs) along one kxokb5hx edge: cost 1
      - pick up spanner (moi6wnfu): cost 1
      - tighten nut (kjsqfni8): cost 1

    Assumptions
    -----------
    - There is (effectively) a single agent (person). If multiple agents
      exist the heuristic uses one representative agent (the one that
      appears in facts but is not a nut or spanner).
    - Spanners that have the g0brvc06 fact are usable; a spanner may be
      currently held (yud7t3vb) or lying at a location (u4srx1nb).
    - Each tightening consumes a good spanner (the spanner loses g0brvc06).
    - The heuristic is not required to be admissible; it uses a greedy
      sequential assignment of spanners to remaining nuts to produce a
      compact estimate that accounts for travel and pick/tighten actions.

    Heuristic Initialization
    ------------------------
    - Extract goal nut symbols from task.goals (goals are g59iwyxf <nut>).
    - Extract static directed adjacency edges from kxokb5hx facts in
      task.static, forming an adjacency map for BFS shortest-path calls.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are already true in the state -> return 0.
    2. Parse the current state into:
       - loc_of: mapping object -> location from u4srx1nb facts.
       - holder_map: mapping person -> spanner from yud7t3vb facts.
       - good_spanners: set of spanners with g0brvc06 in the state.
       - target_nuts: list of nut objects that still need g59iwyxf (present
         in task goals but absent in the current state).
       - nut locations from loc_of.
    3. Build the set/dict of available good spanners and their effective
       locations; a spanner that is held is considered located at the
       holder's location.
    4. Greedy sequential simulation:
       - Start from the agent's current location.
       - While nuts remain:
         - If agent currently holds a usable spanner, compute for every
           remaining nut: cost = distance(agent_loc, nut_loc) + 1 (tighten).
         - Otherwise, for every usable spanner compute the cost to:
             move to spanner + pick (1) + move to nut + tighten (1).
         - Choose the nut (and spanner) that yields the minimal incremental
           cost, add that cost, update agent location to the nut's location,
           and mark the spanner used (it becomes unavailable).
       - If at any point a required movement is impossible (no path) or no
         spanners remain available for some remaining nut, return a large
         finite value to indicate high cost (keeps heuristic finite).
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        # Goals: collect the nut symbols that must have g59iwyxf
        self.goals = task.goals
        # Build adjacency map from static kxokb5hx facts (directed edges)
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx":
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, set()).add(dst)

    def __call__(self, node):
        # If goal already reached, heuristic is zero.
        state = node.state
        # Quick check: if all goal facts are in the state, return 0.
        if self.goals <= state:
            return 0

        # Parse state facts
        loc_of = {}           # object -> location (from u4srx1nb)
        holder_of = {}        # person -> spanner (from yud7t3vb)
        held_by = {}          # spanner -> person (inverse of above)
        good_spanners = set() # spanners with g0brvc06
        tightened = set()     # nuts already tightened (g59iwyxf)
        # Collect all facts we care about
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                loc_of[obj] = loc
            elif pred == "yud7t3vb" and len(parts) >= 3:
                person, spanner = parts[1], parts[2]
                holder_of[person] = spanner
                held_by[spanner] = person
            elif pred == "g0brvc06" and len(parts) >= 2:
                good_spanners.add(parts[1])
            elif pred == "g59iwyxf" and len(parts) >= 2:
                tightened.add(parts[1])

        # Identify the agent (person). Heuristic assumes one main agent.
        # Select an object which appears in loc_of but is not a nut or spanner.
        # Nuts are those appearing in goals (goal predicate is g59iwyxf).
        goal_nuts = [get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "g59iwyxf"]
        nut_set = set(goal_nuts)
        # Spanners are objects that appear in good_spanners or held_by
        spanner_set = set(good_spanners) | set(held_by.keys())
        agent = None
        for obj, loc in loc_of.items():
            if obj not in nut_set and obj not in spanner_set:
                agent = obj
                break
        # Fallback: if not found, but holder_of exists, pick the holder
        if agent is None and holder_of:
            agent = next(iter(holder_of.keys()))
        # If still no agent found, return a large cost to keep heuristic finite
        if agent is None or agent not in loc_of:
            return 100000

        agent_loc = loc_of[agent]

        # Determine which nuts still need tightening (present in goals but not in state)
        remaining_nuts = [n for n in goal_nuts if n not in tightened]
        # If none, cost 0 (should be captured by goal test earlier, but safe)
        if not remaining_nuts:
            return 0

        # Build available good spanners mapping to their effective locations.
        # If a spanner is held, its effective location is the holder's location.
        available_spanners = {}
        for sp in good_spanners:
            if sp in held_by:
                holder = held_by[sp]
                # If holder has a known location
                if holder in loc_of:
                    available_spanners[sp] = loc_of[holder]
                else:
                    # Unknown holder location -> treat as unavailable
                    continue
            else:
                # If spanner lying at a location
                if sp in loc_of:
                    available_spanners[sp] = loc_of[sp]
                else:
                    # No location known for this spanner -> skip
                    continue

        total_cost = 0
        cur_loc = agent_loc
        # If agent currently holds a usable spanner, note it
        agent_holding = holder_of.get(agent, None)
        # If agent holds a spanner that is not good, treat as not holding usable spanner
        if agent_holding is not None and agent_holding not in good_spanners:
            agent_holding = None

        # Greedy sequentially assign spanners to nuts.
        nuts = set(remaining_nuts)
        spanners = dict(available_spanners)  # mutable copy

        # If there are no available spanners and agent is not holding one, impossible
        if not spanners and agent_holding is None:
            return 100000

        while nuts:
            best_choice = None  # (cost, nut, spanner_used, new_agent_loc)
            # For each nut, evaluate cost
            for nut in list(nuts):
                nut_loc = loc_of.get(nut, None)
                if nut_loc is None:
                    # Can't find nut location -> treat as unreachable (high cost)
                    continue
                # If agent holds a good spanner, cost is just move to nut + tighten
                if agent_holding is not None:
                    d = shortest_path_distance(self.adj, cur_loc, nut_loc)
                    if d is None:
                        # unreachable
                        continue
                    cost = d + 1  # move + tighten
                    choice = (cost, nut, agent_holding, nut_loc)
                    if best_choice is None or cost < best_choice[0]:
                        best_choice = choice
                else:
                    # Try all available spanners and pick minimal
                    for sp, sp_loc in spanners.items():
                        # Need to move from cur_loc to sp_loc, pick (1), move to nut, tighten (1)
                        d1 = shortest_path_distance(self.adj, cur_loc, sp_loc)
                        if d1 is None:
                            continue
                        d2 = shortest_path_distance(self.adj, sp_loc, nut_loc)
                        if d2 is None:
                            continue
                        cost = d1 + 1 + d2 + 1  # move to sp + pick + move to nut + tighten
                        choice = (cost, nut, sp, nut_loc)
                        if best_choice is None or cost < best_choice[0]:
                            best_choice = choice
            # If we couldn't find any feasible nut (due to unreachable or no spanners), return large cost
            if best_choice is None:
                return 100000

            cost, chosen_nut, used_spanner, new_loc = best_choice
            total_cost += cost
            # Update state of simulation:
            # agent will be at new_loc (the nut location) after tightening
            cur_loc = new_loc
            # Remove chosen nut
            nuts.remove(chosen_nut)
            # If the used spanner was one from spanners, remove it (consumed)
            if used_spanner in spanners:
                del spanners[used_spanner]
            # If agent was not holding but used a spanner that was currently located at the agent
            # (this would be captured earlier). After using a spanner, agent no longer holds it.
            agent_holding = None

            # If no spanners left but still nuts remain, impossible -> large cost
            if not spanners and agent_holding is None and nuts:
                return 100000

        return total_cost