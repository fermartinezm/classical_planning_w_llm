from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts in the goal.
    The heuristic considers:
    - The man's current location and whether he's carrying usable spanners.
    - The locations of loose nuts and available spanners.
    - The path required to collect spanners and reach nuts.

    # Assumptions:
    - The man can carry multiple spanners at once.
    - Each spanner can be used only once (after which it becomes unusable).
    - The man must be at the same location as a nut to tighten it.
    - The man must be carrying a usable spanner to tighten a nut.

    # Heuristic Initialization
    - Extract static information about location links to compute shortest paths.
    - Identify goal nuts that need to be tightened.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts that still need to be tightened (not in goal state).
    2. For each loose nut:
       a. If the man is not at the nut's location:
          - Compute the shortest path from current location to nut's location.
          - Add the number of walk actions needed (path length).
       b. If the man is not carrying any usable spanner:
          - Find the nearest usable spanner to the man's current location.
          - Compute the path to the spanner and back to the nut's location.
          - Add pickup action and path length.
       c. Add 1 action for tightening the nut (if conditions are met).
    3. Sum all actions required for all loose nuts.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build a graph of location links for pathfinding
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)

        # Precompute goal nuts (all nuts that need to be tightened)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def _shortest_path_length(self, start, end):
        """Compute the shortest path length between two locations using BFS."""
        if start == end:
            return 0

        visited = set()
        queue = [(start, 0)]
        while queue:
            current, distance = queue.pop(0)
            if current == end:
                return distance
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.links.get(current, set()):
                queue.append((neighbor, distance + 1))
        return float('inf')  # No path exists

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Extract current state information
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        nut_locations = {}
        spanner_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("nut"):
                    nut_locations[obj] = loc
                elif obj.startswith("spanner"):
                    spanner_locations[obj] = loc

        total_cost = 0

        # Process each nut that needs to be tightened
        for nut in self.goal_nuts:
            if match(f"(tightened {nut})", *state):
                continue  # Already tightened

            nut_loc = nut_locations.get(nut)
            if nut_loc is None:
                continue  # Nut not found in state (shouldn't happen)

            # Check if we have a usable spanner (either carried or available)
            usable_carried = [s for s in carried_spanners if s in usable_spanners]
            available_spanners = [s for s in spanner_locations if s in usable_spanners]

            # Cost to get to nut location
            if man_location != nut_loc:
                path_length = self._shortest_path_length(man_location, nut_loc)
                if path_length == float('inf'):
                    continue  # No path exists
                total_cost += path_length

            # Cost to get a usable spanner if needed
            if not usable_carried:
                if not available_spanners:
                    continue  # No usable spanners available

                # Find nearest spanner
                min_cost = float('inf')
                for spanner in available_spanners:
                    spanner_loc = spanner_locations[spanner]
                    # Cost to go to spanner and then to nut
                    cost_to_spanner = self._shortest_path_length(man_location, spanner_loc)
                    cost_to_nut = self._shortest_path_length(spanner_loc, nut_loc)
                    total_spanner_cost = cost_to_spanner + cost_to_nut + 1  # +1 for pickup
                    if total_spanner_cost < min_cost:
                        min_cost = total_spanner_cost

                if min_cost != float('inf'):
                    total_cost += min_cost
                else:
                    continue  # No path to spanner

            # Cost to tighten the nut
            total_cost += 1

        return total_cost
