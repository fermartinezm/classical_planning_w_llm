from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (nuts-and-spanners) domain.

    This heuristic estimates the number of actions required to make all goal nuts
    have predicate (g59iwyxf <nut>) true. Each goal nut must be "operated on"
    by an agent (person) using an intact spanner. Using a spanner requires:
      - a person at the nut's location,
      - the person to have a yud7t3vb relation to an intact spanner (g0brvc06),
      - the nut to be present (bu1om0m1) (precondition of the operator).
    To obtain a spanner a person must move to the spanner's location (using
    n9sbfmrs along kxokb5hx links) and pick it up with moi6wnfu. After using a
    spanner it becomes destroyed (g0brvc06 removed) and cannot be reused.

    Assumptions
    -----------
    - Movement cost: each application of n9sbfmrs (single hop along kxokb5hx) costs 1.
    - Picking a spanner (moi6wnfu) costs 1 action.
    - Using a spanner to fix a nut (kjsqfni8) costs 1 action.
    - There is no "drop" or transfer action for spanners. A spanner carried by a
      person can only be used by that same person.
    - The heuristic assumes actions are executed sequentially by the available
      persons (agents). If multiple persons exist, the heuristic uses a best-effort
      assignment (greedy) to reduce the estimated total number of actions.
    - If there are not enough intact spanners reachable/available to satisfy all
      remaining goal nuts, the heuristic returns a large finite number (representing
      an effectively unreachable estimate).

    Heuristic Initialization
    ------------------------
    - Save goal facts (task.goals).
    - Extract the static adjacency graph from static facts with predicate "kxokb5hx".
      The graph is treated as directed because edges are used as preconditions in
      movement operators.
    - No other static information is necessary for the estimate.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are already present in the current state, return 0.
    2. Parse the current state to extract:
       - Locations of objects recorded via u4srx1nb facts (object -> location).
       - Which spanners are intact (g0brvc06 facts).
       - Which nuts are already fixed (g59iwyxf) and which goal nuts remain.
       - Which nuts are still present (bu1om0m1).
       - Which persons currently hold a spanner (yud7t3vb facts).
    3. Build a directed adjacency mapping from kxokb5hx static facts and use BFS
       to compute shortest-path lengths (in hops) on demand between locations.
    4. For each remaining goal nut, estimate the minimal actions necessary:
       - Consider each person as a candidate performer.
       - If a candidate person already holds an intact spanner:
           cost = distance(person_loc, nut_loc) + 1 (use)
       - Otherwise, the person must go to an intact, free spanner's location:
           cost = distance(person_loc, spanner_loc) + 1 (pick) +
                  distance(spanner_loc, nut_loc) + 1 (use)
       - Spanners currently carried by other persons are NOT available for another
         person (no transfer possible).
    5. Use a greedy assignment loop:
       - Repeatedly choose the remaining nut with the minimal estimated cost
         (considering current agent locations and available spanners),
         add that cost to the total, update the performing agent's location to
         the nut location, and mark the used spanner (if any) as consumed.
       - Repeat until all goal nuts are assigned or no feasible assignment exists.
    6. If some nuts cannot be assigned because no intact spanner is available or
       reachable, return a large finite cost (serves as a "very costly / unlikely"
       estimate).
    """

    def __init__(self, task):
        # Goals that must be achieved (set of goal fact strings)
        self.goals = task.goals
        # Static facts: used to build adjacency graph (kxokb5hx)
        static_facts = task.static

        # Build directed adjacency graph between locations from kxokb5hx facts.
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                frm, to = parts[1], parts[2]
                self.adjacency.setdefault(frm, set()).add(to)

        # A large finite number used when the heuristic wants to indicate
        # a very large cost (e.g., impossible because of lack of spanners).
        self.INF = 10 ** 6

    def shortest_distance(self, start, goal):
        """Return shortest directed distance in hops between start and goal locations.
        If unreachable, return INF. If start == goal, distance is 0.
        """
        if start == goal:
            return 0
        if start not in self.adjacency:
            return self.INF
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            loc, d = queue.popleft()
            for nbr in self.adjacency.get(loc, ()):
                if nbr == goal:
                    return d + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, d + 1))
        return self.INF

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If goals are already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse state facts
        obj_location = {}  # mapping object -> location from u4srx1nb facts
        intact_spanners = set()  # spanner names that have g0brvc06
        nuts_fixed = set()  # nuts that already have g59iwyxf
        nuts_present_flag = set()  # nuts that have bu1om0m1 (present)
        person_holds = {}  # person -> spanner (from yud7t3vb)
        # Also track which objects are spanners or nuts by seeing predicates.
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                obj_location[obj] = loc
            elif pred == "g0brvc06" and len(parts) >= 2:
                intact_spanners.add(parts[1])
            elif pred == "g59iwyxf" and len(parts) >= 2:
                nuts_fixed.add(parts[1])
            elif pred == "bu1om0m1" and len(parts) >= 2:
                nuts_present_flag.add(parts[1])
            elif pred == "yud7t3vb" and len(parts) >= 3:
                person, spanner = parts[1], parts[2]
                person_holds[person] = spanner

        # Determine which nuts are required by the goal but not yet fixed.
        goal_nuts = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "g59iwyxf" and len(parts) >= 2:
                goal_nuts.append(parts[1])
        remaining_nuts = [n for n in goal_nuts if n not in nuts_fixed]

        if not remaining_nuts:
            return 0

        # Identify persons: objects that appear in u4srx1nb but are not spanners or nuts.
        # We consider an object a spanner if it is listed in intact_spanners or appears as a value in person_holds.
        possible_spanners = set(intact_spanners) | set(person_holds.values())
        # Also an object is a nut if it appears in nuts_fixed or nuts_present_flag or as a goal nut.
        possible_nuts = set(nuts_fixed) | set(nuts_present_flag) | set(goal_nuts)

        persons = {}
        for obj, loc in obj_location.items():
            if obj in possible_spanners or obj in possible_nuts:
                continue
            # treat as person
            persons[obj] = {"loc": loc, "holds": None}

        # Include persons that may not appear in obj_location but appear in person_holds
        for p, sp in person_holds.items():
            if p not in persons:
                # If person's location not recorded via u4srx1nb, try to find it in obj_location
                loc = obj_location.get(p, None)
                persons[p] = {"loc": loc, "holds": sp}
            else:
                persons[p]["holds"] = sp

        # For persons we don't have location for, set INF-loc (unreachable)
        for p in list(persons.keys()):
            if persons[p]["loc"] is None:
                # If person has no recorded location, set to a dummy unreachable token.
                persons[p]["loc"] = "__UNKNOWN_LOC__"

        # Build set of free intact spanners (not currently held by a person)
        free_intact_spanners = set()
        for sp in intact_spanners:
            # if any person holds this spanner, skip (not free)
            if sp in person_holds.values():
                continue
            # else check if we have a location for the spanner
            loc = obj_location.get(sp, None)
            if loc is not None:
                free_intact_spanners.add(sp)
        # Map spanner -> location for free spanners
        spanner_location = {sp: obj_location[sp] for sp in free_intact_spanners if sp in obj_location}

        # Also keep track of spanners that are carried and intact (usable but only by holder)
        carried_intact_spanners = {p: sp for p, sp in person_holds.items() if sp in intact_spanners}

        total_cost = 0

        # Greedy assignment: repeatedly pick the remaining nut with minimal estimated cost
        remaining = set(remaining_nuts)
        # Copy persons info so we can update locations and held spanners
        agents = {p: {"loc": info["loc"], "holds": info["holds"]} for p, info in persons.items()}
        # spanners available at locations (not carried). We'll remove items as consumed.
        sp_at_loc = dict(spanner_location)  # sp -> loc

        # If there are no agents (persons), this is unsolvable; return large cost.
        if not agents:
            return self.INF

        while remaining:
            best_choice = None
            best_cost = self.INF
            best_details = None  # (nut, agent, spanner_used_or_None, used_from_carried_flag)

            # Evaluate each nut
            for nut in remaining:
                # Where is the nut? must exist in obj_location to be fixable (precondition)
                nut_loc = obj_location.get(nut)
                if nut_loc is None:
                    # If nut has no recorded location, we cannot reach it in this model.
                    continue

                # For each agent consider:
                for agent_name, info in agents.items():
                    agent_loc = info["loc"]
                    # If agent holds an intact spanner:
                    if info["holds"] and info["holds"] in intact_spanners:
                        # cost: move to nut + use
                        dist = self.shortest_distance(agent_loc, nut_loc)
                        if dist >= self.INF:
                            continue
                        cost = dist + 1  # +1 for kjsqfni8 (use)
                        if cost < best_cost:
                            best_cost = cost
                            best_choice = nut
                            best_details = (agent_name, info["holds"], True)  # used carried spanner
                    # Otherwise, try free spanners
                    for sp, sp_loc in sp_at_loc.items():
                        dist_to_sp = self.shortest_distance(agent_loc, sp_loc)
                        if dist_to_sp >= self.INF:
                            continue
                        dist_sp_to_nut = self.shortest_distance(sp_loc, nut_loc)
                        if dist_sp_to_nut >= self.INF:
                            continue
                        # cost: move to sp + pick + move to nut + use
                        cost = dist_to_sp + 1 + dist_sp_to_nut + 1
                        if cost < best_cost:
                            best_cost = cost
                            best_choice = nut
                            best_details = (agent_name, sp, False)  # used free spanner at sp_loc

            if best_choice is None or best_cost >= self.INF:
                # No feasible assignment for some remaining nut (not enough spanners or unreachable)
                return self.INF

            # Apply the best assignment
            chosen_nut = best_choice
            agent_name, sp_used, used_carried = best_details

            # Update total cost and state of agent and spanner pool
            total_cost += best_cost

            # Agent moves to nut location
            agents[agent_name]["loc"] = obj_location.get(chosen_nut)

            # If spanner was carried by the agent, it becomes destroyed and agent no longer holds it
            if used_carried:
                # Remove the spanner from agent's holds (destroyed)
                agents[agent_name]["holds"] = None
                # Also mark spanner as no longer intact (remove from intact_spanners if present)
                if sp_used in intact_spanners:
                    intact_spanners.remove(sp_used)
            else:
                # Spanner was taken from sp_at_loc and consumed
                if sp_used in sp_at_loc:
                    del sp_at_loc[sp_used]
                if sp_used in intact_spanners:
                    intact_spanners.remove(sp_used)
                # After picking, agent no longer holds it after using, so no need to set holds.

            # The nut is now considered fixed (would result in g59iwyxf in real plan)
            remaining.remove(chosen_nut)

        return total_cost