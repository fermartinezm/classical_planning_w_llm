from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions required to tighten all loose nuts by assigning each to a usable spanner. The heuristic considers movement between locations, picking up spanners, and tightening actions. It uses a greedy approach to assign the closest usable spanner to each nut, ensuring each spanner is used at most once.

    # Assumptions
    - The man can carry multiple spanners but each spanner can be used only once.
    - Movement between linked locations is bidirectional and costs 1 action per step.
    - The problem is solvable (enough spanners exist), but a penalty is added for missing spanners.

    # Heuristic Initialization
    - Precomputes shortest paths between all locations using BFS based on static link facts.
    - Extracts goal conditions to check if the state is a goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state meets all goals; return 0 if true.
    2. Extract the man's current location.
    3. Identify all loose nuts and their locations.
    4. Gather all usable spanners, noting if they are carried or on the ground.
    5. For each loose nut, greedily assign the closest usable spanner (minimizing total steps and actions).
    6. Sum the costs of movement, pickup, and tighten actions for assigned spanners.
    7. Add a penalty if there are more loose nuts than usable spanners.
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        # Build shortest paths between locations using BFS
        self.shortest_paths = {}
        links = set()

        # Extract link facts from static information
        for fact in static:
            if match(fact, 'link', '*', '*'):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                links.add((l1, l2))
                links.add((l2, l1))  # links are bidirectional

        # Build adjacency list
        graph = defaultdict(list)
        for l1, l2 in links:
            graph[l1].append(l2)
            graph[l2].append(l1)

        # Collect all unique locations
        locations = set()
        for l1, l2 in links:
            locations.add(l1)
            locations.add(l2)
        locations = list(locations)

        # Precompute shortest paths from each location to all others
        for start in locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                current_dist = visited[current]
                for neighbor in graph[current]:
                    if neighbor not in visited or current_dist + 1 < visited.get(neighbor, float('inf')):
                        visited[neighbor] = current_dist + 1
                        queue.append(neighbor)
            # Update shortest_paths
            for end in locations:
                self.shortest_paths[(start, end)] = visited.get(end, float('inf'))

    def __call__(self, node):
        state = node.state

        # Check if all goals are satisfied
        if self.goals <= state:
            return 0

        # Extract man's current location
        man_location = None
        for fact in state:
            if match(fact, 'at', 'bob', '*'):
                man_location = get_parts(fact)[2]
                break
        if not man_location:
            return float('inf')  # Invalid state

        # Extract loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if match(fact, 'loose', '*'):
                nut = get_parts(fact)[1]
                # Find nut's location
                for f in state:
                    if match(f, 'at', nut, '*'):
                        nut_loc = get_parts(f)[2]
                        loose_nuts.append((nut, nut_loc))
                        break

        # Extract usable spanners and their locations/carried status
        usable_spanners = []
        for fact in state:
            if match(fact, 'usable', '*'):
                spanner = get_parts(fact)[1]
                carried = False
                # Check if spanner is carried
                for f in state:
                    if match(f, 'carrying', 'bob', spanner):
                        carried = True
                        break
                if carried:
                    usable_spanners.append((spanner, man_location, True))
                else:
                    # Find spanner's location
                    for f in state:
                        if match(f, 'at', spanner, '*'):
                            spanner_loc = get_parts(f)[2]
                            usable_spanners.append((spanner, spanner_loc, False))
                            break

        total_cost = 0
        remaining_spanners = usable_spanners.copy()
        penalty_per_missing = 1000  # Arbitrary large penalty for missing spanners

        for nut, nut_loc in loose_nuts:
            if not remaining_spanners:
                # No more spanners, add penalty for remaining nuts
                total_cost += penalty_per_missing * (len(loose_nuts) - len(usable_spanners))
                break

            min_cost = float('inf')
            best_spanner = None
            for s, s_loc, carried in remaining_spanners:
                if carried:
                    steps = self.shortest_paths.get((man_location, nut_loc), float('inf'))
                    cost = steps + 1  # walk steps + tighten
                else:
                    steps_to_spanner = self.shortest_paths.get((man_location, s_loc), float('inf'))
                    steps_to_nut = self.shortest_paths.get((s_loc, nut_loc), float('inf'))
                    cost = steps_to_spanner + 1 + steps_to_nut + 1  # walk, pickup, walk, tighten
                if cost < min_cost:
                    min_cost = cost
                    best_spanner = (s, s_loc, carried)

            if best_spanner is None:
                return float('inf')  # No path found

            total_cost += min_cost
            remaining_spanners.remove(best_spanner)

        # Add penalty for any remaining nuts if there are not enough spanners
        if len(loose_nuts) > len(usable_spanners):
            total_cost += penalty_per_missing * (len(loose_nuts) - len(usable_spanners))

        return total_cost
