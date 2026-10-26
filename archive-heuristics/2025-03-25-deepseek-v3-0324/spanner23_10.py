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
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the following aspects:
    - The man's current location and whether he needs to move to pick up spanners
    - The locations of loose nuts and whether they need to be tightened
    - The locations of usable spanners and whether they need to be picked up
    - Whether the man is currently carrying usable spanners

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can be used to tighten exactly one nut
    - The man must be at the same location as a nut to tighten it
    - The man must be at the same location as a spanner to pick it up

    # Heuristic Initialization
    - Extract the goal conditions (which nuts need to be tightened)
    - Extract static information about location links
    - Create a mapping of locations to their connected neighbors

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many loose nuts still need to be tightened (from goal conditions)
    2. Check if the man is carrying any usable spanners
    3. For each loose nut:
       a. Calculate the distance from the man's current location to the nut's location
       b. If no usable spanners are carried, find the nearest usable spanner
       c. Add the distance to get the spanner and then go to the nut
    4. For each required spanner pickup, add 1 action (pickup_spanner)
    5. For each nut to be tightened, add 1 action (tighten_nut)
    6. Sum all movement actions (walk), pickup actions, and tightening actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Build a graph of location connections
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Extract goal nuts that need to be tightened
        goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                goal_nuts.add(get_parts(goal)[1])
        
        # Find current state of nuts (loose or tightened)
        loose_nuts = set()
        tightened_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.add(get_parts(fact)[1])
            elif match(fact, "tightened", "*"):
                tightened_nuts.add(get_parts(fact)[1])
        
        # Nuts that still need to be tightened
        remaining_nuts = goal_nuts - tightened_nuts
        
        # If all nuts are tightened, heuristic is 0
        if not remaining_nuts:
            return 0
        
        # Find man's current location
        man_location = None
        carrying_spanners = set()
        usable_spanners = set()
        spanner_locations = {}
        nut_locations = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carrying_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "at", "*", "*"):
                obj = parts[1]
                loc = parts[2]
                if obj.startswith("spanner"):
                    spanner_locations[obj] = loc
                elif obj.startswith("nut"):
                    nut_locations[obj] = loc
        
        # Calculate heuristic value
        total_cost = 0
        
        # Find usable spanners the man is carrying
        available_spanners = [s for s in carrying_spanners if s in usable_spanners]
        
        for nut in remaining_nuts:
            nut_loc = nut_locations[nut]
            
            # If we have a usable spanner, just go to the nut
            if available_spanners:
                distance = self._shortest_path_distance(man_location, nut_loc)
                total_cost += distance + 1  # +1 for tighten action
                man_location = nut_loc  # Update man's location after moving
                available_spanners.pop()  # Use up one spanner
            else:
                # Need to find nearest usable spanner
                min_spanner_dist = float('inf')
                best_spanner = None
                
                for spanner in usable_spanners:
                    if spanner in spanner_locations:
                        spanner_loc = spanner_locations[spanner]
                        distance = self._shortest_path_distance(man_location, spanner_loc)
                        if distance < min_spanner_dist:
                            min_spanner_dist = distance
                            best_spanner = spanner
                
                if best_spanner:
                    # Cost to get spanner and go to nut
                    spanner_loc = spanner_locations[best_spanner]
                    to_spanner = min_spanner_dist
                    spanner_to_nut = self._shortest_path_distance(spanner_loc, nut_loc)
                    total_cost += to_spanner + 1 + spanner_to_nut + 1  # +1 for pickup and +1 for tighten
                    man_location = nut_loc  # Update man's location
                    usable_spanners.remove(best_spanner)  # Mark spanner as used
                else:
                    # No usable spanners left - problem unsolvable from this state
                    return float('inf')
        
        return total_cost

    def _shortest_path_distance(self, start, end):
        """Calculate the shortest path distance between two locations using BFS."""
        if start == end:
            return 0
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            if current == end:
                return distance
                
            if current not in visited:
                visited.add(current)
                for neighbor in self.location_graph.get(current, []):
                    queue.append((neighbor, distance + 1))
        
        # No path exists (shouldn't happen in valid problems)
        return float('inf')
