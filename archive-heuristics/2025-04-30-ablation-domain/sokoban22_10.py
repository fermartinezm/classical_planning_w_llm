# Add necessary imports
from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # If the number of parts doesn't match the number of args, it's not a match
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of pushes required to move each box
    from its current location to its goal location. It calculates the shortest
    path distance for each box independently, ignoring the robot's position,
    other boxes, and obstacles (walls/non-clear locations). This is a lower
    bound on the number of pushes needed for the boxes themselves, but ignores
    the robot's movement and coordination costs.

    # Assumptions
    - The locations form an undirected graph defined by `adjacent` predicates.
    - The cost of moving a box is related to the shortest path distance in this graph.
    - The heuristic sums the individual box-to-goal distances.
    - All locations mentioned in `adjacent` facts are valid nodes in the graph.
    - All boxes mentioned in the goal are present in the initial state and subsequent states.
    - Goal conditions are only of the form `(at ?box ?location)` or an `(and ...)` of such facts.

    # Heuristic Initialization
    - Builds an undirected graph of locations based on `adjacent` facts.
    - Stores the goal location for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value to 0.
    2. Identify the current location of every box in the state by iterating
       through the state facts and finding facts matching `(at ?box ?location)`
       where `?box` starts with "box".
    3. For each box that is specified in the goal conditions (stored during initialization):
        a. Retrieve the box's current location from the identified locations. If the box
           is not found in the current state facts, it implies an unreachable or
           malformed state for this goal, so return infinity.
        b. Retrieve the box's goal location (stored during initialization).
        c. If the box's current location is not the same as its goal location:
            i. Calculate the shortest path distance between the box's current location
               and its goal location using Breadth-First Search (BFS) on the pre-built
               location graph. This distance represents the minimum number of steps
               the box would need to move if it could move freely along the graph edges.
            ii. If no path exists between the current and goal location for the box
                (BFS returns infinity), the state is unsolvable for this box,
                so return infinity for the overall heuristic value.
            iii. Add the calculated distance to the total heuristic value.
    4. Return the total heuristic value. If all goal boxes were already at their
       goal locations, the sum will be 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and storing goal locations.

        @param task: The planning task object containing initial state, goals, operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the undirected graph of locations from adjacent facts
        self.location_graph = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent" and len(args) == 3:
                loc1, loc2, direction = args
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = set()
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = set()
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Assuming adjacency is symmetric for pathfinding

        # Store goal locations for each box
        self.goal_locations = {}
        # The task.goals attribute is a frozenset of goal facts.
        # If the goal is (and (fact1) (fact2)...), the frozenset contains a single string like '(and (fact1) (fact2)...)'
        # If the goal is a single fact, the frozenset contains that single fact string.
        goal_facts_list = []
        if isinstance(self.goals, frozenset) and len(self.goals) == 1:
             single_goal_str = next(iter(self.goals))
             if single_goal_str.startswith('(and '):
                 # Simple parsing for (and (fact1) (fact2)...)
                 # Assumes facts inside are simple like (predicate arg1 arg2)
                 content = single_goal_str[5:-1].strip() # Remove "(and " and ")"
                 # Split by ') (' to get individual fact strings (with outer parens missing)
                 fact_strings = content.split(') (')
                 for fs in fact_strings:
                     if fs: # Avoid empty strings
                         goal_facts_list.append('(' + fs + ')') # Add back parentheses
             elif single_goal_str.startswith('('):
                 # Single fact goal
                 goal_facts_list.append(single_goal_str)
             else:
                 # Unexpected format
                 print(f"Warning: Unexpected single goal format: {single_goal_str}")
        elif isinstance(self.goals, frozenset):
             # Multiple goals not wrapped in (and ...) - less common but possible
             goal_facts_list = list(self.goals)
        else:
             # Unexpected type for self.goals
             print(f"Warning: Unexpected type for task.goals: {type(self.goals)}")


        # Process the extracted goal facts
        for goal in goal_facts_list:
            predicate, *args = get_parts(goal)
            # Assuming goal facts for boxes are always (at ?box ?location)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location
            # Ignore other potential goal predicates if any

    def get_distance(self, start_loc, end_loc):
        """
        Calculates the shortest path distance between two locations
        using BFS on the location graph. Returns infinity if no path exists.

        @param start_loc: The starting location name (string).
        @param end_loc: The target location name (string).
        @return: The shortest distance (integer) or float('inf') if unreachable.
        """
        if start_loc == end_loc:
            return 0
        # Ensure locations exist in the graph before starting BFS
        if start_loc not in self.location_graph or end_loc not in self.location_graph:
             # This indicates a problem with the PDDL definition or state,
             # but returning infinity is a safe heuristic value for unreachable goals.
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Check if current_loc has neighbors in the graph
            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # If the queue is empty and the end_loc was not reached
        return float('inf') # No path found

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions (pushes).

        @param node: The search node containing the current state.
        @return: The estimated heuristic cost (integer or float('inf')).
        """
        state = node.state

        # Find current location of each box
        current_box_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            # Assuming facts like "(at box1 loc_R_C)"
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                 box, loc = args
                 current_box_locations[box] = loc

        total_heuristic = 0

        # For each box that has a goal location
        for box, goal_loc in self.goal_locations.items():
            # Check if the box exists in the current state
            if box in current_box_locations:
                current_loc = current_box_locations[box]
                # If the box is not yet at its goal
                if current_loc != goal_loc:
                    # Calculate shortest path distance for the box
                    dist = self.get_distance(current_loc, goal_loc)
                    # If the goal is unreachable for this box, the state is unsolvable
                    if dist == float('inf'):
                        return float('inf')
                    total_heuristic += dist
            else:
                 # If a box required by the goal is not found in the state,
                 # this state is likely invalid or represents an impossible scenario.
                 # Return infinity to prune this branch.
                 return float('inf')

        # The total heuristic is the sum of distances for all boxes not at their goals.
        # If all goal boxes are at their goals, the loop finishes and total_heuristic is 0.
        return total_heuristic
