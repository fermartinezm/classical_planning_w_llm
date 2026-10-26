from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    return fact[1:-1].split()


def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class Spanner22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions needed to tighten all loose nuts. For each nut, it considers the minimal steps required to either use a carried usable spanner or fetch a new one.

    # Assumptions
    - The man can carry multiple spanners, but each spanner can be used once.
    - The shortest path between locations is computed using directed links.
    - The problem is solvable, with enough usable spanners for all loose nuts.

    # Heuristic Initialization
    - Extracts static links to build a directed graph and precompute shortest paths between all locations using BFS.

    # Step-By-Step Thinking
    1. Determine the man's current location.
    2. For each loose nut:
        a. Calculate the cost to tighten using a carried spanner (if available).
        b. Calculate the cost to fetch the closest usable spanner and then tighten.
        c. Sum the minimal cost from the above options.
    3. Return the total estimated actions.
    """

    def __init__(self, task):
        self.static = task.static
        self.goals = task.goals

        # Build directed graph from static links
        links = []
        for fact in self.static:
            if match(fact, 'link', '*', '*'):
                parts = get_parts(fact)
                from_loc, to_loc = parts[1], parts[2]
                links.append((from_loc, to_loc))

        self.graph = defaultdict(list)
        for from_loc, to_loc in links:
            self.graph[from_loc].append(to_loc)

        # Precompute shortest paths using BFS
        self.shortest_paths = {}
        for start in self.graph:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for loc, dist in visited.items():
                self.shortest_paths[(start, loc)] = dist

    def __call__(self, node):
        state = node.state
        man = None
        man_location = None

        # Determine the man's name from carrying facts
        for fact in state:
            if match(fact, 'carrying', '*', '*'):
                man = get_parts(fact)[1]
                break

        # Fallback to find man's name from 'at' fact (not a spanner/nut)
        if man is None:
            for fact in state:
                if match(fact, 'at', '*', '*'):
                    obj = get_parts(fact)[1]
                    if not (fnmatch(obj, 'spanner*') or fnmatch(obj, 'nut*')):
                        man = obj
                        break

        # Find man's current location
        for fact in state:
            if match(fact, 'at', man, '*'):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
            return float('inf')

        # Collect loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if match(fact, 'loose', '*'):
                nut = get_parts(fact)[1]
                for f in state:
                    if match(f, 'at', nut, '*'):
                        nut_loc = get_parts(f)[2]
                        loose_nuts.append((nut, nut_loc))
                        break

        # Collect usable spanners (carried and in-world)
        carried_spanners = []
        usable_spanners = []
        for fact in state:
            if match(fact, 'usable', '*'):
                spanner = get_parts(fact)[1]
                # Check if carried
                carried = any(match(f, 'carrying', man, spanner) for f in state)
                if carried:
                    carried_spanners.append(spanner)
                else:
                    # Check if in world
                    for f in state:
                        if match(f, 'at', spanner, '*'):
                            spanner_loc = get_parts(f)[2]
                            usable_spanners.append((spanner, spanner_loc))
                            break

        total = 0
        for nut, nut_loc in loose_nuts:
            option1 = None
            if carried_spanners:
                distance = self.shortest_paths.get((man_location, nut_loc), float('inf'))
                option1 = distance + 1  # Tighten action

            # Calculate option2: fetch and use a spanner
            option2 = float('inf')
            for s, s_loc in usable_spanners:
                d1 = self.shortest_paths.get((man_location, s_loc), float('inf'))
                d2 = self.shortest_paths.get((s_loc, nut_loc), float('inf'))
                total_d = d1 + d2
                if total_d < option2:
                    option2 = total_d
            option2 += 2  # Pickup and tighten

            # Choose minimal option
            if option1 is not None:
                total += min(option1, option2)
            else:
                total += option2

        return total
