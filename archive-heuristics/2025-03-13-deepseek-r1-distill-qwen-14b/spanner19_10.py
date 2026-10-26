from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts. Each nut requires the man to be at its location with a usable spanner. The heuristic calculates the minimal steps needed to achieve this for all nuts.

    # Assumptions:
    - The man can carry at most one spanner at a time.
    - The man starts at the shed location if not specified otherwise.
    - The goal is to have all nuts tightened.
    - The static facts include the links between locations.

    # Heuristic Initialization
    - Extracts static facts (location links) and goal conditions (all nuts must be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the man and all spanners.
    2. For each loose nut:
        a. Determine the distance (number of steps) between the man's current location and the nut's location.
        b. If the man is not carrying a spanner, add the steps to pick one up.
        c. Add one step to tighten the nut.
    3. Sum the steps for all nuts, considering that after each tightening, the spanner becomes unusable.
    4. If multiple nuts are in the same location, minimize redundant movements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals  # Goal conditions (all nuts must be tightened)
        static_facts = task.static  # Static facts (location links)

        # Extract location links from static facts
        self.location_links = {}
        for fact in static_facts:
            if fact.startswith('(link'):
                parts = fact[1:-1].split()
                loc1, loc2 = parts[1], parts[2]
                if loc1 not in self.location_links:
                    self.location_links[loc1] = []
                self.location_links[loc1].append(loc2)
                if loc2 not in self.location_links:
                    self.location_links[loc2] = []
                self.location_links[loc2].append(loc1)

        # Precompute the man's initial location if possible
        self.man_initial_location = None
        for fact in static_facts:
            if fact.startswith('(at bob'):
                self.man_initial_location = fact[1:-1].split()[1]

    def __call__(self, node):
        """Estimate the minimum number of actions to tighten all loose nuts."""
        state = node.state

        def match(fact, *args):
            """Check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current state information
        man_location = None
        carried_spanner = None
        nut_info = {}
        for fact in state:
            if match(fact, 'at', 'bob', '*'):
                man_location = fact[1:-1].split()[2]
            elif match(fact, 'carrying', 'bob', '*'):
                carried_spanner = fact[1:-1].split()[2]
            elif match(fact, 'at', '*', 'nut'):
                nut = fact[1:-1].split()[1]
                loc = fact[1:-1].split()[2]
                nut_info[nut] = loc
            elif match(fact, 'loose', '*'):
                nut = fact[1:-1].split()[1]
                if nut not in nut_info:
                    nut_info[nut] = None  # Nut exists but location unknown

        # If man's location is not found, assume initial location
        if man_location is None:
            man_location = self.man_initial_location

        total_actions = 0

        # For each loose nut, calculate required actions
        for nut, loc in nut_info.items():
            if not match(f'(tightened {nut})', '*'):
                if loc is None:
                    continue  # Nut location unknown, cannot estimate

                # Calculate distance between man and nut
                distance = 0
                current = man_location
                path = []
                visited = set()
                queue = [(current, 0)]
                found = False

                while queue:
                    current_loc, dist = queue.pop(0)
                    if current_loc == loc:
                        distance = dist
                        found = True
                        break
                    if current_loc in visited:
                        continue
                    visited.add(current_loc)
                    for neighbor in self.location_links.get(current_loc, []):
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))

                if not found:
                    continue  # Cannot reach nut, heuristic is infinity

                # Add walking actions
                total_actions += distance

                # If not carrying a spanner, add pickup and carry action
                if carried_spanner is None:
                    total_actions += 1  # Pickup spanner
                    carried_spanner = True  # Assume spanner is picked up

                # Add tightening action
                total_actions += 1

        return total_actions
