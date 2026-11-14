from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(u4srx1nb spanner1 location4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    =========
    The heuristic estimates the minimum number of actions needed to reach a goal
    state by focusing on progressing the "nuts" (objects of type rupjajag) from 
    their current locations to the goal condition (predicate g59iwyxf nutX). 
    The heuristic models the sequence of required actions to achieve this, 
    taking into account that nuts must be unlocked/unblocked before they can be 
    "activated" (g59iwyxf true) via the kjsqfni8 action, which requires:
      - The nut to be at a location where bu1om0m1 holds (nut is "locked" initially).
      - The corresponding spanner (e2zu8y93) to be at some location and "enabled"
        (g0brvc06 predicate).
      - A person (ilfxqhcc) related to a spanner via predicate yud7t3vb.
      - Locks can be manipulated by shifting objects of type ilfxqhcc along spatial
        connections (kxokb5hx) with the n9sbfmrs action.

    The heuristic draws from the structure of the domain and its three actions:
      1. n9sbfmrs: Moves a person (ilfxqhcc) between adjacent locations (i6azntdy).
      2. moi6wnfu: Places a person at an e2zu8y93 object if the person and object 
         share the same location.
      3. kjsqfni8: Unlocks (activates) a nut if preconditions about location and 
         spanner state hold.

    Assumptions:
    ============
    - The goal is to satisfy all g59iwyxf predicates for the nuts given in the goal.
    - The map of locations and their adjacency is static and represented by kxokb5hx relations.
    - The positions of persons (ilfxqhcc) can be moved along adjacent locations.
    - Spanners (e2zu8y93) must be enabled (g0brvc06 true) and related by yud7t3vb to a person.
    - Nuts (rupjajag) start locked by bu1om0m1, must be at gate locations reachable via 
      kxokb5hx paths, and finally activated via g59iwyxf.
    - We do not consider action costs explicitly; each action counts as cost 1.
    - The heuristic is not guaranteed admissible but aims to give a meaningful estimate 
      of minimal action steps.

    Heuristic Initialization:
    =========================
    - Extract from static facts the adjacency graph between locations (kxokb5hx).
    - Extract from goals the nuts that must be activated (g59iwyxf nut).
    - Store mappings of which nuts are locked (bu1om0m1) and their current location.
    - Extract the initial and current location of persons (ilfxqhcc).
    - Track enabled spanners (g0brvc06).
    - Map persons to spanners if yud7t3vb holds.
    - Build a graph of locations (i6azntdy) connected by kxokb5hx for shortest path calculations.

    Step-By-Step Thinking for Computing Heuristic:
    ==============================================
    1. Identify all nuts that are not yet activated (missing g59iwyxf nut).
    2. For each such nut, determine whether it is still locked (bu1om0m1) and its current location.
    3. Identify the persons (ilfxqhcc) and their current locations.
    4. Identify spanners (e2zu8y93) that are enabled (g0brvc06) and which person is assigned to which spanner (yud7t3vb).
    5. For each nut to be activated, the heuristic estimates:
       a) Move a person (assigned to a spanner) from their current location to the nut's current location
          using the shortest path on the adjacency graph.
       b) Apply moi6wnfu to place the person on the spanner (if not already done).
       c) Apply kjsqfni8 to activate the nut (removing locks and setting g59iwyxf).
    6. Also consider that persons may need to move along the chain of locations via n9sbfmrs.
    7. Sum over all nuts the estimated minimal actions:
       - minimal moves for the person to reach the nut location,
       - 1 action for moi6wnfu if needed,
       - 1 action for kjsqfni8.
    8. For persons assigned to spanners but not at the spanner location, count moves needed to get there 
       as well, since preconditions demand the person to be at the same location as the spanner.
    9. The heuristic takes the minimum cost over available persons for each nut to find the least cost person
       to activate it.
    10. The heuristic returns zero if all nuts are activated (i.e., goal reached).

    Note:
    =====
    The heuristic ignores possible interleavings and concurrency, focusing on minimal distances and required 
    action steps to achieve the goal predicates.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build location adjacency graph from static facts (kxokb5hx)
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "kxokb5hx", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)  # Assuming undirected adjacency

        # Extract nuts that must be activated (g59iwyxf nut)
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "g59iwyxf", "*"):
                nut = get_parts(goal)[1]
                self.goal_nuts.add(nut)

    def shortest_path_length(self, start, goal):
        """Find shortest path length between two locations in the location graph using BFS.
        Returns a large number (1000) if not reachable."""
        if start == goal:
            return 0
        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in self.location_graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        return 1000  # Large value if no path

    def __call__(self, node):
        state = node.state

        # If goal fully reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Extract persons and their current locations
        persons = set()
        person_locations = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                # ilfxqhcc persons: those that appear as ?v7he5 in u4srx1nb facts and are of type ilfxqhcc
                # We must recognize persons by their names appearing in yud7t3vb or by domain knowledge.
                # Since type is not explicit here, we identify persons as those also appearing in yud7t3vb predicate.
                persons.add(obj)
                person_locations[obj] = loc

        # Extract spanners enabled and their locations (g0brvc06) and track relation yud7t3vb person <-> spanner
        spanner_enabled = set()
        spanner_locations = {}
        for fact in state:
            if match(fact, "g0brvc06", "*"):
                spanner = get_parts(fact)[1]
                spanner_enabled.add(spanner)

        # Spanner locations from u4srx1nb facts
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                # Among objects, spanners are those of type e2zu8y93, but type is unknown here, so recognize 
                # spanners that are enabled or linked to persons
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                if obj in spanner_enabled:
                    spanner_locations[obj] = loc

        # Persons assigned to spanners (yud7t3vb person spanner)
        person_to_spanner = {}
        spanner_to_person = {}
        for fact in state:
            if match(fact, "yud7t3vb", "*", "*"):
                parts = get_parts(fact)
                person = parts[1]
                spanner = parts[2]
                person_to_spanner[person] = spanner
                spanner_to_person[spanner] = person

        # Extract nuts currently locked (bu1om0m1 nut) and their locations
        locked_nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "bu1om0m1", "*"):
                nut = get_parts(fact)[1]
                locked_nuts.add(nut)
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                if obj in locked_nuts or obj in self.goal_nuts:
                    nut_locations[obj] = loc

        # Nuts that are not yet activated:
        nuts_not_activated = self.goal_nuts - {get_parts(f)[1] for f in state if match(f, "g59iwyxf", "*")}

        # If no nuts left to activate, heuristic is zero
        if not nuts_not_activated:
            return 0

        total_cost = 0

        # For each nut to activate, estimate minimal cost:
        # Cost: 
        # - move person from current location to nut location (shortest path)
        # - if person not at spanner location, move to spanner location also counted
        # - 1 action for moi6wnfu (placing person on spanner)
        # - 1 action for kjsqfni8 (activating nut)
        # Also, persons can move along adjacency.

        for nut in nuts_not_activated:
            nut_loc = nut_locations.get(nut, None)
            if nut_loc is None:
                # Nut location unknown, assign large cost to discourage searching
                total_cost += 1000
                continue

            best_cost_for_nut = 10000  # large initial cost

            for person in persons:
                spanner = person_to_spanner.get(person, None)
                if spanner is None:
                    # Person is not assigned to spanner, can't activate nut, skip
                    continue

                spanner_loc = spanner_locations.get(spanner, None)
                if spanner_loc is None:
                    # Spanner location unknown, assign large cost
                    continue

                person_loc = person_locations.get(person, None)
                if person_loc is None:
                    continue

                # Cost for person to move to nut location to unlock (n9sbfmrs moves)
                dist_person_to_nut = self.shortest_path_length(person_loc, nut_loc)

                # Cost for person to move from nut location to spanner location (for moi6wnfu)
                dist_nut_to_spanner = self.shortest_path_length(nut_loc, spanner_loc)

                # We count the moves as separate segments since person must move there for unlock,
                # then be at spanner location for moi6wnfu.

                # Total moves:
                # Move person from current to nut location + move nut to spanner location (actually moving person)
                moves = dist_person_to_nut + dist_nut_to_spanner

                # Add 1 action for moi6wnfu, 1 for kjsqfni8
                moves += 2

                if moves < best_cost_for_nut:
                    best_cost_for_nut = moves

            # If no person found to operate the nut, assign high cost
            if best_cost_for_nut == 10000:
                best_cost_for_nut = 1000

            total_cost += best_cost_for_nut

        return total_cost
