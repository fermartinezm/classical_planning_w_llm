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
    It considers the man's current location, carried spanners, and the locations of
    loose nuts and usable spanners.

    # Assumptions:
    - The man can carry multiple spanners at once.
    - Each spanner can only be used once (becomes unusable after tightening a nut).
    - The man must be at the nut's location to tighten it.
    - The man must be at a spanner's location to pick it up.

    # Heuristic Initialization
    - Extract the link information between locations from static facts.
    - Identify goal nuts that need to be tightened.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (from goals).
    2. Check if the man is carrying any usable spanners:
       - If not, find the nearest usable spanner and add walking/pickup actions.
    3. For each loose nut:
       - Add walking actions to reach the nut's location.
       - If no usable spanner is available at the nut's location, add actions to:
         - Walk to a usable spanner's location.
         - Pick up the spanner.
         - Walk back to the nut's location.
       - Add the tighten action.
    4. Optimize by considering carried spanners and their usability.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build a graph of connected locations from static 'link' facts
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Extract goal nuts that need to be tightened
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        total_cost = 0

        # Get man's current location
        man_loc = None
        carrying_spanners = set()
        usable_spanners = set()
        loose_nuts = set()
        spanner_locations = {}
        nut_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_loc = parts[2]
            elif match(fact, "carrying", "*", "*"):
                carrying_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "loose", "*"):
                loose_nuts.add(parts[1])
            elif match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("spanner"):
                    spanner_locations[obj] = loc
                elif obj.startswith("nut"):
                    nut_locations[obj] = loc

        # Only consider nuts that are both loose and in goals
        relevant_nuts = loose_nuts & self.goal_nuts
        if not relevant_nuts:
            return 0  # All goal nuts are already tightened

        # Find usable spanners the man is carrying
        available_spanners = [s for s in carrying_spanners if s in usable_spanners]

        # If no usable spanner is carried, find the nearest one
        if not available_spanners:
            # Find all usable spanners not carried
            unused_spanners = [s for s in usable_spanners if s not in carrying_spanners]
            if unused_spanners:
                # Get the closest spanner (simplification: just pick first one)
                closest_spanner = unused_spanners[0]
                spanner_loc = spanner_locations[closest_spanner]
                
                # Add cost to walk to spanner and pick it up
                if man_loc != spanner_loc:
                    total_cost += 1  # Simplified: assume 1 walk action
                total_cost += 1  # pickup_spanner action
                available_spanners.append(closest_spanner)
            else:
                # No usable spanners left - problem unsolvable from this state
                return float('inf')

        # For each relevant nut, add actions to tighten it
        for nut in relevant_nuts:
            nut_loc = nut_locations[nut]
            
            # Walk to nut location if not already there
            if man_loc != nut_loc:
                total_cost += 1  # Simplified: assume 1 walk action
            
            # Use a spanner to tighten the nut
            total_cost += 1  # tighten_nut action
            if available_spanners:
                available_spanners.pop()  # Spanner becomes unusable
            
            # If we ran out of spanners, get another one
            if not available_spanners:
                unused_spanners = [s for s in usable_spanners 
                                 if s not in carrying_spanners and s in spanner_locations]
                if unused_spanners:
                    closest_spanner = unused_spanners[0]
                    spanner_loc = spanner_locations[closest_spanner]
                    
                    # Walk to spanner and pick it up
                    if nut_loc != spanner_loc:
                        total_cost += 1  # walk action
                    total_cost += 1  # pickup_spanner action
                    available_spanners.append(closest_spanner)
                else:
                    # No more usable spanners
                    return float('inf')

        return total_cost
