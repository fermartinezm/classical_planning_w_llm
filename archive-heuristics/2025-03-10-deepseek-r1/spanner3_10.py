from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class spanner3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions required to tighten all goal nuts by calculating the minimal path for each nut using the closest available spanner. Each spanner can be used once.

    # Assumptions
    - Each nut requires a unique spanner.
    - The man can carry multiple spanners but must pick up each spanner once.
    - Links form a directed graph; shortest paths are computed using BFS.

    # Heuristic Initialization
    - Extracts static links to build a graph for shortest path calculations.
    - Identifies goal nuts from the task's goals.

    # Step-By-Step Thinking
    1. Determine the man's current location and carried spanners.
    2. Identify all loose nuts that are part of the goal.
    3. For each loose nut:
        a. Calculate the cost using each available spanner (carried or not).
        b. Select the spanner with the minimal cost.
        c. Sum costs, ensuring each spanner is used once.
    """

    def __init__(self, task):
        self.goal_nuts = set()
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        self.links = {}
        for fact in task.static:
            if fact.startswith('(link '):
                parts = fact[1:-1].split()
                if parts[0] == 'link' and len(parts) == 3:
                    from_loc, to_loc = parts[1], parts[2]
                    self.links.setdefault(from_loc, []).append(to_loc)

    def _shortest_path(self, start, end):
        if start == end:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.links.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state
        man, man_loc = None, None

        # Identify the man
        for fact in state:
            if fact.startswith('(carrying '):
                parts = fact[1:-1].split()
                man = parts[1]
                break
        if not man:
            for fact in state:
                if fact.startswith('(at '):
                    parts = fact[1:-1].split()
                    if parts[0] == 'at' and parts[1] not in self.goal_nuts and 'spanner' not in parts[1]:
                        man, man_loc = parts[1], parts[2]
                        break
        else:
            for fact in state:
                if fact.startswith('(at ') and parts[1] == man:
                    parts = fact[1:-1].split()
                    man_loc = parts[2]
                    break

        if not man or not man_loc:
            return 0

        # Collect loose nuts in goal
        loose_nuts = [nut for nut in self.goal_nuts if f'(loose {nut})' in state]

        # Get nut locations
        nut_locs = {}
        for nut in loose_nuts:
            for fact in state:
                if fact.startswith('(at ') and parts[1] == nut:
                    parts = fact[1:-1].split()
                    nut_locs[nut] = parts[2]
                    break

        # Usable spanners
        usable_spanners = [parts[1] for fact in state if fact.startswith('(usable ') and (parts := fact[1:-1].split())]

        # Carried spanners
        carried = {parts[2] for fact in state if fact.startswith('(carrying ') and (parts := fact[1:-1].split()) and parts[1] == man}

        # Spanner locations
        spanner_locs = {}
        for s in usable_spanners:
            if s in carried:
                spanner_locs[s] = man_loc
            else:
                for fact in state:
                    if fact.startswith('(at ') and (parts := fact[1:-1].split()) and parts[1] == s:
                        spanner_locs[s] = parts[2]
                        break

        available_spanners = set(usable_spanners)
        total_cost = 0

        for nut in loose_nuts:
            nl = nut_locs.get(nut)
            if not nl:
                continue

            min_cost, chosen = float('inf'), None
            for s in list(available_spanners):
                sl = spanner_locs.get(s)
                if not sl:
                    continue

                if s in carried:
                    cost = self._shortest_path(man_loc, nl) + 1
                else:
                    d1 = self._shortest_path(man_loc, sl)
                    d2 = self._shortest_path(sl, nl)
                    cost = d1 + d2 + 2

                if cost < min_cost:
                    min_cost, chosen = cost, s

            if chosen:
                total_cost += min_cost
                available_spanners.discard(chosen)

        return total_cost
