from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at nut1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the
    number of walk, pickup_spanner, and tighten_nut actions required.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A usable spanner must be carried to tighten a nut.
    - The heuristic assumes the shortest path between locations.

    # Heuristic Initialization
    - Extract the locations of all objects (man, spanners, nuts) from the initial state.
    - Identify the links between locations from the static facts.
    - Store the usable spanners.
    - Store the loose nuts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify the loose nuts and their locations.
    3. For each loose nut:
       a. Find a usable spanner.
       b. If the man is not carrying a usable spanner, estimate the cost to:
          i. Walk to a location with a usable spanner.
          ii. Pick up the spanner.
       c. Estimate the cost to:
          i. Walk to the nut's location.
          ii. Tighten the nut.
    4. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information for pathfinding.
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 not in self.links:
                    self.links[l1] = []
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l1].append(l2)
                self.links[l2].append(l1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the goal is already reached.
        if self.goal_reached(state):
            return 0

        # Extract information from the current state.
        man_location = None
        carried_spanner = None
        usable_spanners = set()
        loose_nuts = set()
        nut_locations = {}
        spanner_locations = {}
        man_location = None

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                if obj == 'bob':
                    man_location = location
                elif any(s in obj for s in ['spanner1', 'spanner2', 'spanner3', 'spanner4', 'spanner5', 'spanner6', 'spanner7', 'spanner8', 'spanner9', 'spanner10']):
                    spanner_locations[obj] = location
            elif match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                if any(n in obj for n in ['nut1', 'nut2', 'nut3', 'nut4', 'nut5']):
                    nut_locations[obj] = location
            elif match(fact, "carrying", "*", "*"):
                carried_spanner = get_parts(fact)[2]
            elif match(fact, "usable", "*"):
                usable_spanners.add(get_parts(fact)[1])
            elif match(fact, "loose", "*"):
                loose_nuts.add(get_parts(fact)[1])

        # If no loose nuts, return 0
        if not loose_nuts:
            return 0

        # Calculate the heuristic cost.
        total_cost = 0
        for nut in loose_nuts:
            nut_location = nut_locations[nut]
            # Find a usable spanner.
            found_spanner = False
            for spanner in usable_spanners:
                if spanner in spanner_locations:
                    found_spanner = True
                    break

            if not found_spanner:
                continue

            if carried_spanner not in usable_spanners:
                # Walk to a location with a usable spanner and pick it up.
                best_spanner_location = None
                min_dist = float('inf')
                for spanner in usable_spanners:
                    if spanner in spanner_locations:
                        spanner_location = spanner_locations[spanner]
                        dist = self.shortest_path(man_location, spanner_location)
                        if dist < min_dist:
                            min_dist = dist
                            best_spanner_location = spanner_location

                if best_spanner_location is not None:
                    total_cost += min_dist + 1  # Walk + pickup
                    man_location = best_spanner_location
                    carried_spanner = spanner

            # Walk to the nut's location and tighten it.
            dist = self.shortest_path(man_location, nut_location)
            total_cost += dist + 1  # Walk + tighten

        return total_cost

    def goal_reached(self, state):
        """Check if all goal conditions are satisfied in the given state."""
        return self.goals <= state

    def shortest_path(self, start, end):
        """Compute the shortest path between two locations using BFS."""
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)
            if location == end:
                return distance

            if location in self.links:
                for neighbor in self.links[location]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

        return float('inf')  # No path found
