# Assume Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_3_5)" -> ["at", "box1", "loc_3_5"]
    return fact[1:-1].split()

def get_coords(loc_str):
    """Parse location string 'loc_R_C' into (R, C) tuple of integers."""
    # Example: "loc_3_5" -> (3, 5)
    parts = loc_str.split('_')
    # Basic check for expected format
    if len(parts) == 3 and parts[0] == 'loc' and parts[1].isdigit() and parts[2].isdigit():
        return (int(parts[1]), int(parts[2]))
    else:
        # Handle unexpected format - returning (0,0) as a fallback.
        # In a real system, robust error handling or logging might be needed.
        # Assuming valid 'loc_R_C' format based on problem description examples.
        return (0, 0)

def manhattan_distance(loc1_str, loc2_str):
    """Calculate Manhattan distance between two locations."""
    r1, c1 = get_coords(loc1_str)
    r2, c2 = get_coords(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)

def sign(x):
    """Return the sign of a number."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push each box
    to its goal location, considering the robot's position and the need
    to reposition the robot between pushes when the box changes direction.
    It sums the estimated costs for each box independently.

    # Assumptions
    - The locations are on a grid and named 'loc_R_C'. Manhattan distance
      is used as an estimate for shortest path distance for both box pushes
      and robot movements through clear cells.
    - Each box is assigned to a unique goal location (inferred from goal state).
    - The cost of moving the robot through clear cells is approximated by
      Manhattan distance, ignoring obstacles (other boxes, walls implicitly
      defined by lack of adjacent facts/clear predicates). This is a relaxation.
    - A minimal path for a box from its current location to its goal involves
      at most one change in direction (e.g., all vertical moves then all
      horizontal moves, or vice versa).

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
      Assumes goal conditions are of the form `(at <box> <location>)`.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of the estimated costs for
    each box that is not yet at its goal location. The cost for a single
    box is estimated as follows:

    Let the box be at B=(rb, cb), its goal at G=(rg, cg), and the robot at R=(rr, cr).

    1.  **Pushes Cost:** The minimum number of push actions required for the box
        to reach its goal is the Manhattan distance `n = |rb - rg| + |cb - cg|`.
        Each push costs 1 action. This contributes `n` to the box's cost.

    2.  **Robot Movement Cost:** The robot needs to move to a position adjacent
        to the box to perform each push.
        -   **Initial Approach:** To perform the *first* push (from B towards G),
            the robot must be at a specific location P_1, adjacent to B and
            opposite the direction of the first step.
            -   If the box needs to move vertically (rb != rg), the first vertical
                step is towards (rb + sign(rg-rb), cb). The required robot location
                P_v is (rb - sign(rg-rb), cb).
            -   If the box needs to move horizontally (cb != cg), the first horizontal
                step is towards (rb, cb + sign(cg-cb)). The required robot location
                P_h is (rb, cb - sign(cg-cb)).
            The robot must move from its current location R to the required position
            for the first push.
            -   If only vertical movement is needed (cb == cg), the robot must reach P_v.
                The initial approach cost is `dist(R, P_v)`.
            -   If only horizontal movement is needed (rb == rg), the robot must reach P_h.
                The initial approach cost is `dist(R, P_h)`.
            -   If both vertical and horizontal movement are needed (rb != rg and cb != cg),
                the box can move either vertically or horizontally first. The robot must
                reach either P_v or P_h. The minimum robot approach cost is
                `min(dist(R, P_v), dist(R, P_h))`. This cost is added to the box's total.

        -   **Repositioning Between Pushes:** After a push, the robot ends up at the
            box's location *before* that push. To perform the *next* push, the robot
            might need to move again. If the box moves consecutively in the same
            direction (e.g., down then down), the robot is already in the correct
            relative position for the next push (repositioning cost is 0). If the
            box changes direction (e.g., down then right), the robot needs to move
            to get into the correct position for the next push. Assuming a minimal
            path for the box involves at most one direction change, this single
            change incurs a robot repositioning cost. A simple estimate for this
            cost is 2 robot moves (e.g., moving around the corner of the box).
            This cost of 2 is added *only once* per box if it requires both
            vertical and horizontal movement to reach the goal.

    The total heuristic is the sum of `pushes_cost + initial_approach_cost + repositioning_cost`
    for each box not at its goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goal conditions are always (at <box> <location>) for boxes
            if predicate == "at" and len(args) == 2:
                 obj, location = args
                 # We only care about objects of type 'box'
                 # PDDL type info is not directly in facts, rely on naming convention 'boxN'
                 if obj.startswith('box'):
                      self.goal_locations[obj] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        loc_r_str = None
        # Iterating through the frozenset to find the robot location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                loc_r_str = parts[1]
                break

        # If robot location is not found, something is wrong with the state representation
        # or it's an unreachable state in practice. Return infinity.
        if loc_r_str is None:
             return float('inf')

        # Find current box locations
        box_locations = {}
        # Iterating through the frozenset to find box locations
        for fact in state:
            parts = get_parts(fact)
            # Assuming box locations are represented as (at <box> <location>)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith('box'):
                 box_locations[parts[1]] = parts[2]

        total_heuristic = 0

        # Calculate cost for each box not at its goal
        for box, goal_loc_str in self.goal_locations.items():
            current_loc_str = box_locations.get(box)

            # If box is not found in the state or is already at the goal, cost is 0 for this box
            if current_loc_str is None or current_loc_str == goal_loc_str:
                continue

            # Get coordinates
            rb, cb = get_coords(current_loc_str)
            rg, cg = get_coords(goal_loc_str)
            rr, cr = get_coords(loc_r_str)

            # 1. Cost for pushes (Manhattan distance)
            pushes_cost = manhattan_distance(current_loc_str, goal_loc_str)
            box_cost = pushes_cost

            # 2. Cost for robot approach and repositioning
            initial_approach_cost = 0
            repositioning_cost = 0

            needs_vertical = (rb != rg)
            needs_horizontal = (cb != cg)

            if needs_vertical and not needs_horizontal:
                # Only vertical movement needed for the box
                s = sign(rg - rb)
                P_v_coords = (rb - s, cb)
                P_v_str = f'loc_{P_v_coords[0]}_{P_v_coords[1]}'
                initial_approach_cost = manhattan_distance(loc_r_str, P_v_str)

            elif needs_horizontal and not needs_vertical:
                # Only horizontal movement needed for the box
                t = sign(cg - cb)
                P_h_coords = (rb, cb - t)
                P_h_str = f'loc_{P_h_coords[0]}_{P_h_coords[1]}'
                initial_approach_cost = manhattan_distance(loc_r_str, P_h_str)

            elif needs_vertical and needs_horizontal:
                # Both vertical and horizontal movement needed for the box
                s = sign(rg - rb)
                t = sign(cg - cb)
                P_v_coords = (rb - s, cb)
                P_h_coords = (rb, cb - t)
                P_v_str = f'loc_{P_v_coords[0]}_{P_v_coords[1]}'
                P_h_str = f'loc_{P_h_coords[0]}_{P_h_coords[1]}'

                # Robot needs to reach the required position for the first push
                initial_approach_cost = min(
                    manhattan_distance(loc_r_str, P_v_str),
                    manhattan_distance(loc_r_str, P_h_str)
                )
                # Add cost for repositioning when the box changes direction (once)
                repositioning_cost = 2

            box_cost += initial_approach_cost + repositioning_cost
            total_heuristic += box_cost

        return total_heuristic
