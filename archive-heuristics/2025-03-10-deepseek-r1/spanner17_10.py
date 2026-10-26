from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions required to tighten all loose nuts by:
    - Walking to pick up necessary spanners.
    - Walking to each nut's location to tighten them.
    - Counting pickup and tighten actions.

    # Assumptions
    - The man can carry multiple spanners.
    - Each spanner can be used only once.
    - The shortest path between locations is used for movement steps.

    # Heuristic Initialization
    - Precomputes shortest paths between all locations using static link facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Find all loose nuts and group them by location.
    3. Count usable spanners carried and available in the world.
    4. Calculate the number of spanners to pickup.
    5. For each needed pickup, select the closest usable spanner and add steps.
    6. Add steps to travel to each nut location and perform tighten actions.
    """

    def __init__(self, task):
        self.static_links = set()
        self.shortest_paths = dict()

        # Extract static links from the task's static facts
        for fact in task.static:
            if fnmatch(fact, '(link * *)'):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                self.static_links.add((l1, l2))
                self.static_links.add((l2, l1))  # Assuming bidirectional links

        # Build the graph for shortest paths
        graph = defaultdict(list)
        for l1, l2 in self.static_links:
            graph[l1].append(l2)
            graph[l2].append(l1)

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = defaultdict(dict)
        locations = set(graph.keys())
        for loc in locations:
            visited = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            self.shortest_paths[loc] = visited

    def __call__(self, node):
        state = node.state

        # Extract man's name and current location
        man = None
        for fact in state:
            if fnmatch(fact, '(carrying * *)'):
                parts = fact[1:-1].split()
                man = parts[1]
                break
        if not man:  # Fallback to 'at' predicate if not carrying
            for fact in state:
                if fnmatch(fact, '(at * *)'):
                    parts = fact[1:-1].split()
                    obj = parts[1]
                    if not any(fnmatch(obj, pattern) for pattern in ['spanner*', 'nut*']):
                        man = obj
                        break
        if not man:
            return 0

        man_location = None
        for fact in state:
            if fnmatch(fact, '(at %s *)' % man):
                parts = fact[1:-1].split()
                man_location = parts[2]
                break
        if not man_location:
            return 0

        # Collect loose nuts and group by location
        loose_nuts = []
        for fact in state:
            if fnmatch(fact, '(loose *)'):
                nut = fact[1:-1].split()[1]
                loose_nuts.append(nut)

        nut_locations = defaultdict(int)
        for nut in loose_nuts:
            for fact in state:
                if fnmatch(fact, '(at %s *)' % nut):
                    loc = fact[1:-1].split()[2]
                    nut_locations[loc] += 1
                    break

        # Count usable carried spanners
        carried_usable = 0
        for fact in state:
            if fnmatch(fact, '(carrying %s *)' % man):
                spanner = fact[1:-1].split()[2]
                if '(usable %s)' % spanner in state:
                    carried_usable += 1

        # Collect usable spanners in the world
        world_spanners = []
        for fact in state:
            if fnmatch(fact, '(at * *)'):
                parts = fact[1:-1].split()
                obj, loc = parts[1], parts[2]
                if fnmatch(obj, 'spanner*') and '(usable %s)' % obj in state:
                    world_spanners.append((obj, loc))

        # Calculate needed pickups
        num_loose = len(loose_nuts)
        num_pickups = max(0, num_loose - carried_usable)
        steps = 0
        current_loc = man_location
        remaining_spanners = world_spanners.copy()

        # Handle spanner pickups
        for _ in range(num_pickups):
            if not remaining_spanners:
                break
            # Find closest spanner
            min_dist = float('inf')
            closest = None
            for (spanner, loc) in remaining_spanners:
                if current_loc in self.shortest_paths and loc in self.shortest_paths[current_loc]:
                    dist = self.shortest_paths[current_loc][loc]
                    if dist < min_dist:
                        min_dist = dist
                        closest = (spanner, loc)
            if closest:
                steps += min_dist  # Walk to spanner
                steps += 1  # Pickup action
                current_loc = closest[1]
                remaining_spanners.remove(closest)

        # Handle nut tightening
        for loc, count in nut_locations.items():
            if current_loc in self.shortest_paths and loc in self.shortest_paths[current_loc]:
                steps += self.shortest_paths[current_loc][loc]  # Walk to nut location
                steps += count  # Tighten actions
                current_loc = loc  # Update current location

        # Add actions for pickups and tightens
        steps += num_pickups + num_loose
        return steps
