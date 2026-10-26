from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts. Each nut requires the man to be at its location with a usable spanner. The heuristic calculates the minimal steps needed to achieve this for all nuts.

    # Assumptions:
    - The man can carry at most one spanner at a time.
    - The man starts at the shed location.
    - The goal is to have all specified nuts tightened.
    - The shortest path between locations is used to estimate movement costs.

    # Heuristic Initialization
    - Extracts goal conditions for each nut.
    - Constructs a graph of locations from static facts to compute distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each nut that needs to be tightened:
       a. If the nut is already tightened, skip it.
       b. Determine if the man has a usable spanner. If not, add actions to pick one up.
       c. Calculate the shortest path from the man's current location to the nut's location.
       d. Add actions for moving to the nut's location, picking up the spanner (if needed), and tightening the nut.
    2. Sum the actions for all nuts to get the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (location links)

        # Build location graph from static facts
        self.location_links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = fact[5:-1].split()
                if loc1 not in self.location_links:
                    self.location_links[loc1] = []
                if loc2 not in self.location_links[loc1]:
                    self.location_links[loc1].append(loc2)
                if loc2 not in self.location_links:
                    self.location_links[loc2] = []
                if loc1 not in self.location_links[loc2]:
                    self.location_links[loc2].append(loc1)

        # Precompute all possible shortest paths between locations using BFS
        self.location_dist = {}
        for loc in self.location_links:
            self.location_dist[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = {loc}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.location_links[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.location_dist[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        goal_nuts = [fact[6:-1].split()[0] for fact in self.goals if fact.startswith("(tightened ")]

        # If all nuts are already tightened, return 0
        if all(fact in state for fact in self.goals):
            return 0

        # Extract current locations
        current_locations = {}
        for fact in state:
            if fact.startswith("(at "):
                obj, loc = fact[4:-1].split()
                current_locations[obj] = loc
            elif fact.startswith("(carrying "):
                man, spanner = fact[8:-1].split()
                current_locations[man] = spanner

        # If the man is not in the state, assume he's at shed (default)
        man = "bob"
        if man not in current_locations:
            current_locations[man] = "shed"

        total_actions = 0

        # For each nut that needs to be tightened
        for nut in goal_nuts:
            nut_fact = f"(loose {nut})"
            if nut_fact not in state:
                continue  # Already tightened

            nut_loc = None
            for fact in state:
                if fact.startswith(f"(at {nut})"):
                    nut_loc = fact[4:-1].split()[1]
                    break

            if nut_loc is None:
                continue  # Nut location not found (should not happen)

            # Check if the man has a usable spanner
            has_spanner = any(fact.startswith(f"(carrying {man} ") and "usable" in fact for fact in state)
            if not has_spanner:
                # Find nearest spanner location
                spanner_locs = [fact[4:-1].split()[1] for fact in state if fact.startswith("(at spanner") and "usable" in fact]
                if not spanner_locs:
                    continue  # No spanners available (should not happen)
                
                # Find the closest spanner location to the man's current location
                min_dist = float('inf')
                closest_spanner = None
                man_loc = current_locations[man]
                for spanner_loc in spanner_locs:
                    if spanner_loc in self.location_dist.get(man_loc, {}):
                        dist = self.location_dist[man_loc][spanner_loc]
                        if dist < min_dist:
                            min_dist = dist
                            closest_spanner = spanner_loc
                if closest_spanner is None:
                    continue  # Should not happen as spanner_locs is non-empty

                # Move to spanner
                total_actions += min_dist + 1  # Move + pick up spanner
                man_loc = closest_spanner
                has_spanner = True

            # Move to nut's location
            if man_loc != nut_loc:
                if nut_loc in self.location_dist.get(man_loc, {}):
                    dist = self.location_dist[man_loc][nut_loc]
                    total_actions += dist  # Walking actions
                else:
                    # If path not found, assume it's unreachable (should not happen)
                    continue

            # Tighten the nut
            total_actions += 1

        return total_actions

    @staticmethod
    def get_parts(fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    @staticmethod
    def match(fact, *args):
        """
        Check if a PDDL fact matches a given pattern.
        - `fact`: The complete fact as a string, e.g., "(at bob shed)".
        - `args`: The expected pattern (wildcards `*` allowed).
        - Returns `True` if the fact matches the pattern, else `False`.
        """
        parts = SpannerHeuristic.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))
