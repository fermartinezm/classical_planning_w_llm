from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts from a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with optional wildcards.

    :param fact: fact string, e.g. "(tightened nut1 locatable1)"
    :param args: pattern parts, e.g. "tightened", "*", "locatable1"
    :return: True if the fact matches pattern, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (spanner) domain.

    Summary
    -------
    This heuristic estimates the number of remaining "tightening" moves needed
    to achieve the goal conditions of the form (tightened-2 <locatable> <location>).
    It uses the difference between the current locations of locatables with respect 
    to nut tightening and their goal targets. It also considers the locations of nuts
    and loose/tightened states to approximate the minimal number of actions.

    Assumptions
    -----------
    - Each goal fact specifies a locatable tightened in a certain location (tightened-2).
    - Movement of tightening along locatables is done by actions that move tightening states along links.
    - Nuts must be tightened from a loose-2 location to the locations given by the goal.
    - The set of locatables, their linking (loose, at, carrying, link, usable) defines possible routes.
    - One action can move tightening from one locatable to another if preconditions hold.
    - The heuristic does *not* attempt to be admissible but aims for good guidance by counting minimal distance to tightenings and nuts locations.
    - Efficiency in computation is important.

    Heuristic Initialization
    ------------------------
    - Extract all goal (tightened-2 locatable location) facts.
    - Extract all static facts representing links between locatables.
    - Build a graph over locatables using "link" facts as nodes and "loose", "at", "carrying", "usable" relations to determine reachable neighbors.
    - Extract the initial positions of nuts and tightened nuts.
    - Store the initial loose-2 locations of nuts.
    - Prepare data structures for fast lookup of neighbors and facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goal facts of the form (tightened-2 locatable location).
       Each such fact means a locatable must be tightened at a certain location.
    
    2. For each goal locatable:
       - Find its current tightened-2 location in the state (if any).
       - If location matches goal location, no cost for that locatable.
       - Otherwise estimate minimal number of "walk" or "tighten_nut" actions needed to move tightening from current locatable to goal location.
         This is approximated by shortest path length on the locatable graph between current locatable and goal locatable, plus the cost related to tightening nuts if necessary.

    3. For each nut:
       - Determine its current loose-2 location and tightened locatable in the state.
       - If the nut is not yet loose-2 at the goal location or tightened at the correct locatable, estimate actions needed for relocating and tightening it.

    4. To estimate travel cost between locatables and locations:
       - Use the preprocessed graph of locatables and locations (from static facts) to compute shortest distances.
    
    5. Aggregate the costs for all nuts and locatables that need tightening.
       - The heuristic sums a count of actions roughly proportional to the number of nuts/locatables that must be moved plus the distance (number of moves) to correct positions.
    
    Note: The heuristic does not consider all complex precondition details fully but uses an approximation of distances and mismatches inferred from facts,
    suitable for guiding greedy best-first search efficiently.

    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse goal tightened-2 facts: map locatable -> location
        self.goal_tightened_2 = {}  # {locatable: location}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "tightened-2" and len(parts) == 3:
                locatable, location = parts[1], parts[2]
                self.goal_tightened_2[locatable] = location

        # Parse static information:
        # Collect all locatables that are linked (have (link ?locatable))
        self.linked_locatables = set()
        for fact in self.static:
            if fact.startswith("(link "):
                parts = get_parts(fact)
                # parts[0] == link, parts[1] = locatable
                self.linked_locatables.add(parts[1])

        # Build adjacency by examining static facts for loose, at, carrying, usable connections
        # We create a graph where nodes are locatables, edges mean possible moves/uses
        # This graph represents the underlying topology for moving tightening, nuts, spanners, etc.

        # We consider these static relation types as edges: loose, at, carrying, usable (bidirectional where meaningful)

        # Adjacency as locatable -> set of locatables reachable by an action step
        self.adj = {loc: set() for loc in self.linked_locatables}

        def add_edge(a, b):
            if a in self.linked_locatables and b in self.linked_locatables:
                self.adj[a].add(b)

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            # Relations with two locatables connected
            if pred == "loose" and len(parts) == 3:
                add_edge(parts[1], parts[2])
                add_edge(parts[2], parts[1])
            elif pred == "at" and len(parts) == 3:
                add_edge(parts[1], parts[2])
                add_edge(parts[2], parts[1])
            elif pred == "carrying" and len(parts) == 3:
                add_edge(parts[1], parts[2])
                # carrying direction might matter; add reverse as well for heuristic path search
                add_edge(parts[2], parts[1])
            elif pred == "usable" and len(parts) == 3:
                add_edge(parts[1], parts[2])
                add_edge(parts[2], parts[1])

        # Extract nuts and their loose-2 initial locations from static, to know all nuts known
        self.nuts = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "loose-2" and len(parts) == 3:
                nut = parts[1]
                self.nuts.add(nut)

    def __call__(self, node):
        state = node.state

        # Shortcut: goal test
        if self.goal_tightened_2 <= state:
            return 0

        # Extract current tightened-2 facts: locatable -> location
        current_tightened_2 = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "tightened-2" and len(parts) == 3:
                locatable, location = parts[1], parts[2]
                current_tightened_2[locatable] = location

        # Extract current tightened facts: nut -> locatable
        current_tightened = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "tightened" and len(parts) == 3:
                nut, locatable = parts[1], parts[2]
                current_tightened[nut] = locatable

        # Extract current loose-2 facts for nuts: nut -> location
        current_loose_2 = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "loose-2" and len(parts) == 3:
                nut, location = parts[1], parts[2]
                current_loose_2[nut] = location

        # Helper: BFS to get shortest path length between two locatables
        # Return number of steps or a large number if unreachable
        def shortest_path_len(start, target):
            if start == target:
                return 0
            if start not in self.adj or target not in self.adj:
                return 1000  # unreachable large penalty
            from collections import deque
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                current, dist = queue.popleft()
                for neigh in self.adj[current]:
                    if neigh == target:
                        return dist + 1
                    if neigh not in visited:
                        visited.add(neigh)
                        queue.append((neigh, dist + 1))
            return 1000  # unreachable large penalty

        heuristic_cost = 0

        # We estimate cost based on:
        # For each locatable in goal tightened-2:
        #  - cost to bring tightening-2 from current locatable position to goal locatable
        # For each nut:
        #  - cost to get nut from current loose-2 position to goal locatable's location and tighten.

        # 1) For each goal tightened-2 locatable:
        for locatable_goal, loc_goal_loc in self.goal_tightened_2.items():
            # Current tightened-2 location for this locatable
            loc_current_loc = current_tightened_2.get(locatable_goal)
            if loc_current_loc == loc_goal_loc:
                # Already at goal location, cost 0 for this locatable tightening-2
                continue

            # If no current tightened-2 for this locatable, we must move tightening here from somewhere else
            # We approximate distance by path length from current_loc to goal locatable
            # For heuristic, approximate cost as path length + 1 for tightening action
            min_cost = 1000
            # Search minimal distance from any tightened-2 locatable to this goal locatable:
            if loc_current_loc:  # if present
                dist = shortest_path_len(loc_current_loc, locatable_goal)
                min_cost = min(min_cost, dist)

            # Also consider that we might need to start from some nut's tightened locatable if no tightened-2 on locatable_goal exists:
            for nut, nut_tight_loc in current_tightened.items():
                dist = shortest_path_len(nut_tight_loc, locatable_goal)
                if dist < min_cost:
                    min_cost = dist

            # Cost: moves along locatable graph + 1 tightening_nut action to finalize tightening-2
            if min_cost == 1000:
                # Unreachable, large penalty
                heuristic_cost += 15
            else:
                heuristic_cost += min_cost + 1

            # Add cost for moving tightening-2 from old location to goal location if different location (location here means a location object)
            # If loc_current_loc and loc_goal_loc differ, additional cost for walk or pickup_spanner type actions
            if loc_current_loc and loc_current_loc != loc_goal_loc:
                heuristic_cost += 1  # estimate 1 action for moving tightened-2 location

        # 2) For each nut:
        for nut in self.nuts:
            nut_loose_loc = current_loose_2.get(nut)
            nut_tight_loc = current_tightened.get(nut)

            # Find the goal locatable(s) related to this nut:
            # Since goal mentions tightened-2 locatable at location,
            # and nuts are tightened at locatables, we try to match goal tightened-2 to nut tightening.

            # Find goal tightened-2 locatables whose location matches loose-2 location or nut tightening location:
            # To simplify, we assume nut should be tightened at some locatable with tightened-2 goal
            # We attempt to find closest goal tightened-2 locatable to current nut tightening locatable.

            if not nut_loose_loc and not nut_tight_loc:
                # Nut is not loose-2 nor tightened: this is a strange partial state, count penalty
                heuristic_cost += 10
                continue

            # Try to find associated goal locatable for this nut by minimal distance among goal tightened-2 locatables
            min_dist = 1000
            associated_goal_loc = None
            for locatable_goal, loc_goal_loc in self.goal_tightened_2.items():
                # Distance between nut’s tightened locatable and goal locatable
                if nut_tight_loc:
                    dist = shortest_path_len(nut_tight_loc, locatable_goal)
                    if dist < min_dist:
                        min_dist = dist
                        associated_goal_loc = locatable_goal

            # If no tightening locatable for nut, try to measure distance from loose-2 loc location
            if nut_tight_loc is None and nut_loose_loc:
                # We guess cost to get nut tightened at a goal locatable:
                for locatable_goal in self.goal_tightened_2.keys():
                    dist = shortest_path_len(locatable_goal, locatable_goal)  # zero, just pick goal locatable
                    if dist < min_dist:
                        min_dist = dist
                        associated_goal_loc = locatable_goal

            # If no association found, assign penalty and continue
            if associated_goal_loc is None:
                heuristic_cost += 8
                continue

            # Cost to move nut from loose location to goal locatable (tightening location)
            # Consider:
            # If nut is loose at a location different from goal location, we need to move it there:
            # Approximate by distance from loose-2 location to goal location + 1 tighten_nut action.
            if nut_loose_loc and nut_loose_loc != self.goal_tightened_2[associated_goal_loc]:
                heuristic_cost += 1  # Cost for tighten_nut action to move nut's loose-2 from old location to goal tightening location
                heuristic_cost += 2  # Add penalty for relocation and tightening actions

            # If nut is tightened at wrong locatable, estimate cost to walk tightening to goal locatable
            if nut_tight_loc and nut_tight_loc != associated_goal_loc:
                dist = shortest_path_len(nut_tight_loc, associated_goal_loc)
                heuristic_cost += dist + 1  # travel + tighten_nut-2 actions

            # If nut is not tightened at all but loose-2 already at goal location, estimate 1 action to tighten nut
            if nut_tight_loc is None and nut_loose_loc == self.goal_tightened_2[associated_goal_loc]:
                heuristic_cost += 1  # one tighten action to finish

        # Bound heuristic to at least 1 if it is positive, to avoid zero for non-goal states
        if heuristic_cost == 0:
            heuristic_cost = 1

        return heuristic_cost
