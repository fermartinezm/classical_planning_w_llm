from fnmatch import fnmatch
from collections import deque
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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) < len(args) or (len(parts) > len(args) and args[-1] != '*'):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It considers the cost of movement,
    picking up spanners, and tightening nuts. It simulates a greedy process
    where the man, if not carrying a spanner, goes to the closest available
    usable spanner on the ground, picks it up, and then goes to the closest
    remaining loose goal nut to tighten it.

    # Assumptions
    - The man can only carry one spanner at a time.
    - Each usable spanner can tighten exactly one nut.
    - There are enough usable spanners available (initially or on the ground)
      to tighten all loose goal nuts in solvable instances.
    - The locations form a connected graph via 'link' predicates, and all
      relevant objects (man, nuts, spanners) are located within this graph.
    - Nuts remain at their initial locations.
    - There is exactly one man object in the problem.

    # Heuristic Initialization
    - Extracts all locations from 'link' and initial 'at' facts to build the location graph.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Stores the goal conditions (specifically, which nuts need to be tightened).
    - Identifies the name of the man object.
    - Stores the static locations of the goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic simulates a greedy process to tighten all loose goal nuts:
    1. Identify the set of loose nuts that are part of the goal in the current state.
    2. If this set is empty, the heuristic is 0.
    3. Identify the man's current location in the state.
    4. Identify usable spanners available in the current state (either carried or on the ground).
    5. Check if the number of loose goal nuts exceeds the number of usable spanners available in the state. If so, return infinity (unsolvable state).
    6. Initialize total cost to 0.
    7. Initialize simulation variables: current man location, set of remaining loose goal nuts, set of usable spanners available on the ground, and whether the man is currently carrying a usable spanner, based on the current state.
    8. While there are still remaining loose goal nuts in the simulation:
       a. If the man is currently carrying a usable spanner in the simulation:
          - Find the remaining loose goal nut closest (by walk distance) to the man's current location in the simulation.
          - Add the walk distance to that nut's location to the total cost.
          - Update the man's current location in the simulation to the nut's location.
          - Add 1 (for the 'tighten_nut' action) to the total cost.
          - Remove the nut from the set of remaining loose goal nuts in the simulation.
          - The man is no longer carrying a usable spanner in the simulation (it was used).
       b. If the man is not currently carrying a usable spanner in the simulation:
          - Find the usable spanner on the ground (that hasn't been picked up in the simulation) closest (by walk distance) to the man's current location in the simulation.
          - If no such spanner exists (which should not happen if the initial check passed), return infinity.
          - Add the walk distance to the spanner's location to the total cost.
          - Update the man's current location in the simulation to the spanner's location.
          - Add 1 (for the 'pickup_spanner' action) to the total cost.
          - Remove the spanner from the set of available spanners on the ground in the simulation.
          - The man is now carrying a usable spanner in the simulation.
    9. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by pre-calculating distances and identifying static elements."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Store static locations of goal nuts (nuts don't move)
        self.nut_locations = {}
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj in self.goal_nuts:
                     self.nut_locations[obj] = loc

        # Identify the man object name
        self.man_name = None
        initial_objects = set()
        initial_spanners = set()
        initial_nuts = set()

        # Collect all objects mentioned in the initial state and categorize potential spanners/nuts
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                initial_objects.add(parts[1])
            elif parts[0] == 'carrying':
                 initial_objects.add(parts[1]) # Man is carrying
                 initial_objects.add(parts[2]) # Spanner is carried
            elif parts[0] == 'usable':
                 initial_objects.add(parts[1]) # Usable object (spanner)
                 initial_spanners.add(parts[1])
            elif parts[0] == 'loose':
                 initial_objects.add(parts[1]) # Loose object (nut)
                 initial_nuts.add(parts[1])
            elif parts[0] == 'tightened':
                 initial_objects.add(parts[1]) # Tightened object (nut)
                 initial_nuts.add(parts[1])

        # The man is the object that is not a spanner and not a nut among initial objects
        potential_men = initial_objects - initial_spanners - initial_nuts
        if len(potential_men) == 1:
            self.man_name = list(potential_men)[0]
        elif 'bob' in initial_objects: # Fallback for common man name
             self.man_name = 'bob'
        else:
            # This case indicates a problem with the problem definition or assumptions
            # The heuristic will likely return infinity later if man_name is None
            print("Warning: Could not uniquely identify the man object in __init__.")


        # Extract all locations from link facts and initial 'at' facts
        locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc) # Add location from initial 'at' fact

        self.locations = list(locations) # Store as list

        # Build graph from link facts
        self.graph = {loc: [] for loc in self.locations}
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in self.graph and loc2 in self.graph: # Ensure locations are in our collected set
                     self.graph[loc1].append(loc2)
                     self.graph[loc2].append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                for neighbor in self.graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Get the pre-calculated shortest distance between two locations."""
        # Return infinity if locations are not in the pre-calculated distances (e.g., isolated)
        return self.distances.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}

        # 2. If all goal nuts are tightened, the heuristic is 0.
        if not loose_goal_nuts:
            return 0

        # 3. Identify man's current location
        man_loc = None
        if self.man_name:
            for fact in state:
                if match(fact, "at", self.man_name, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None:
             # Man's location not found in state (e.g., not 'at' anywhere?)
             # This indicates an invalid state.
             return float('inf')

        # 4. Identify usable spanners available in the current state (carried or on ground)
        usable_spanners_in_state = set() # Stores (spanner_name, location or 'carried')
        for fact in state:
            if match(fact, "usable", "*"):
                spanner_name = get_parts(fact)[1]
                # Find where this usable spanner is
                found_loc = None
                for fact2 in state:
                    if match(fact2, "at", spanner_name, "*"):
                        found_loc = get_parts(fact2)[2]
                        usable_spanners_in_state.add((spanner_name, found_loc))
                        break
                    if self.man_name and match(fact2, "carrying", self.man_name, spanner_name):
                        found_loc = 'carried'
                        usable_spanners_in_state.add((spanner_name, found_loc))
                        break
                # If found_loc is None, the usable spanner is not located anywhere - invalid state?
                # Assuming valid states where usable spanners are either at a location or carried.


        # 5. Check if enough usable spanners exist for remaining nuts
        num_nuts_to_tighten = len(loose_goal_nuts)
        usable_spanners_count_in_state = len(usable_spanners_in_state)

        if num_nuts_to_tighten > usable_spanners_count_in_state:
             # Not enough usable spanners exist in the current state to meet the goal.
             return float('inf') # Unsolvable state with current resources

        # 6. Initialize simulation variables
        total_cost = 0
        current_man_loc_sim = man_loc
        remaining_loose_nuts_sim = set(loose_goal_nuts)
        # Track usable spanners available for pickup in the simulation.
        # Start with usable spanners that are currently on the ground in the state.
        sim_available_spanners_on_ground = {(s_name, s_loc) for s_name, s_loc in usable_spanners_in_state if s_loc != 'carried'}
        # Track if the man starts carrying a usable spanner in the simulation.
        sim_man_carrying_usable = any(s_loc == 'carried' for s_name, s_loc in usable_spanners_in_state)


        # 8. Simulate the greedy process
        while remaining_loose_nuts_sim:
            if sim_man_carrying_usable:
                # Man has a spanner, go tighten the closest remaining loose nut
                closest_nut = None
                min_dist = float('inf')
                nut_loc = None

                for nut in remaining_loose_nuts_sim:
                    # Get the nut's static location
                    current_nut_loc = self.nut_locations.get(nut)
                    if current_nut_loc is None:
                         # Nut location not found (should have been in initial state)
                         return float('inf') # Problem definition issue?

                    dist = self.get_distance(current_man_loc_sim, current_nut_loc)
                    if dist < min_dist:
                        min_dist = dist
                        closest_nut = nut
                        nut_loc = current_nut_loc

                if closest_nut is None:
                     # Should not happen if remaining_loose_nuts_sim is not empty
                     return float('inf') # Error state

                # Cost to walk to the nut
                total_cost += min_dist
                current_man_loc_sim = nut_loc

                # Cost to tighten the nut
                total_cost += 1 # tighten_nut action
                remaining_loose_nuts_sim.remove(closest_nut)
                sim_man_carrying_usable = False # Spanner is used

            else:
                # Man needs a spanner, go pick up the closest usable one on the ground
                closest_spanner_info = None # (spanner_name, location)
                min_dist = float('inf')

                # Find the closest spanner among those available for pickup in the simulation
                for s_name, s_loc in sim_available_spanners_on_ground:
                    dist = self.get_distance(current_man_loc_sim, s_loc)
                    if dist < min_dist:
                        min_dist = dist
                        closest_spanner_info = (s_name, s_loc)

                if closest_spanner_info is None:
                    # No usable spanners left on the ground in the simulation, but nuts remain.
                    # This should have been caught by the initial count check, but as a safeguard:
                    return float('inf') # Unsolvable state

                spanner_name, spanner_loc = closest_spanner_info

                # Cost to walk to the spanner
                total_cost += min_dist
                current_man_loc_sim = spanner_loc

                # Cost to pick up the spanner
                total_cost += 1 # pickup_spanner action
                sim_available_spanners_on_ground.remove((spanner_name, spanner_loc)) # Spanner is no longer on ground
                sim_man_carrying_usable = True # Man is now carrying it

        # 9. Return the total accumulated cost.
        return total_cost

