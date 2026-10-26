from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    return fact[1:-1].split()


def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts by considering the shortest paths for collecting usable spanners and moving to nut locations.

    # Assumptions
    - The man can carry multiple spanners, but each can be used only once.
    - The links between locations form an undirected graph.
    - The problem instances are solvable (enough spanners available for the nuts).

    # Heuristic Initialization
    - Extract static link facts to build a graph for shortest path calculations.
    - The graph is used to compute the minimal steps between any two locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. For each loose nut, determine its location.
    3. For each loose nut, compute the minimal steps required to tighten it:
        a. If the man is carrying a usable spanner, steps are the distance to the nut plus tightening.
        b. Otherwise, find the closest usable spanner (either in the world or carried), compute steps to collect it, move to the nut, and tighten.
    4. Sum the minimal steps for all loose nuts to get the heuristic value.
    """

    def __init__(self, task):
        self.static_graph = {}
        self.link_facts = [fact for fact in task.static if fact.startswith('(link ')]
        
        # Build adjacency list for locations
        self.static_graph = {}
        for fact in self.link_facts:
            parts = get_parts(fact)
            l1, l2 = parts[1], parts[2]
            if l1 not in self.static_graph:
                self.static_graph[l1] = []
            self.static_graph[l1].append(l2)
            if l2 not in self.static_graph:
                self.static_graph[l2] = []
            self.static_graph[l2].append(l1)

    def __call__(self, node):
        state = node.state
        man_name = None
        # Determine the man's name
        for fact in state:
            if fact.startswith('(carrying '):
                parts = get_parts(fact)
                man_name = parts[1]
                break
        if not man_name:
            for fact in state:
                if fact.startswith('(at '):
                    parts = get_parts(fact)
                    obj = parts[1]
                    # Check if obj is a spanner or nut
                    is_spanner = any(f == '(usable {})'.format(obj) for f in state)
                    is_nut = any(f.startswith('(loose {}'.format(obj)) or any(f.startswith('(tightened {}'.format(obj)) for f in state)
                    if not is_spanner and not is_nut:
                        man_name = obj
                        break
        if not man_name:
            return 0  # Fallback if no man found (should not happen in valid states)

        # Get man's location
        man_location = None
        for fact in state:
            if fact.startswith('(at ') and get_parts(fact)[1] == man_name:
                man_location = get_parts(fact)[2]
                break
        if not man_location:
            return 0  # Fallback

        # Collect loose nuts and their locations
        loose_nuts = [get_parts(fact)[1] for fact in state if fact.startswith('(loose ')]
        nut_locations = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in loose_nuts:
                    nut_locations[obj] = loc

        # Collect usable spanners and their locations (carried or in world)
        usable_spanners = [get_parts(fact)[1] for fact in state if fact.startswith('(usable ')]
        carried_spanners = [get_parts(fact)[2] for fact in state if fact.startswith('(carrying {}'.format(man_name))]
        spanner_locations = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in usable_spanners and obj not in carried_spanners:
                    spanner_locations[obj] = loc
        for s in carried_spanners:
            spanner_locations[s] = man_location

        # Function to compute shortest path between two locations
        def shortest_path(start, end):
            if start == end:
                return 0
            visited = set()
            queue = deque([(start, 0)])
            while queue:
                node, dist = queue.popleft()
                if node == end:
                    return dist
                if node not in visited:
                    visited.add(node)
                    for neighbor in self.static_graph.get(node, []):
                        queue.append((neighbor, dist + 1))
            return float('inf')  # No path

        total = 0
        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut, None)
            if not nut_loc:
                continue  # Nut not found (should not happen)
            min_steps = float('inf')

            # Check carried usable spanners
            for s in carried_spanners:
                if s not in usable_spanners:
                    continue
                distance = shortest_path(man_location, nut_loc)
                steps = distance + 1  # tighten
                if steps < min_steps:
                    min_steps = steps

            # Check usable spanners in the world
            for s in usable_spanners:
                if s in carried_spanners:
                    continue
                s_loc = spanner_locations.get(s, None)
                if not s_loc:
                    continue
                to_spanner = shortest_path(man_location, s_loc)
                if to_spanner == float('inf'):
                    continue
                to_nut = shortest_path(s_loc, nut_loc)
                steps = to_spanner + 1 + to_nut + 1  # pickup + tighten
                if steps < min_steps:
                    min_steps = steps

            total += min_steps if min_steps != float('inf') else 0  # Handle unsolvable as 0

        return total
