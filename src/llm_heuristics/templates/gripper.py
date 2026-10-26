from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class GripperHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Gripper domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all balls
    from `rooma` to `roomb`.

    # Assumptions:
    - The robot has two grippers, allowing it to carry up to two balls per trip.
    - The robot must return to rooma after each trip, except for the final trip.
    - If the robot starts in roomb, it must move to rooma first.

    # Heuristic Initialization
    - Implicitly assume that all balls must be in `roomb` at the end.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of balls still in `rooma` that need to be transported.
    2. Determine if the robot is currently carrying balls (it may start with 1 or 2 already).
    3. Check whether the robot is in `rooma` or `roomb`:
       - If in room B, it may need to drop the carried balls first before moving to A.
       - If in room A, it can immediately begin planning the transport.
    4. Handle the case where the robot starts with balls in the grippers:
       - If carrying 2 balls, it should move to B, drop them, and return to A.
       - If carrying 1 ball and an odd number remains in `rooma`, it may pick up another ball before moving.
       - If carrying 1 ball and an even number remains, it transports the single ball first.
    5. Compute the number of full two-ball trips needed:
       - This is `balls_in_rooma // 2` (since up to 2 balls are moved per trip).
       - Each full two-ball trip costs 6 actions (except for the last trip).
    6. Handle the last remaining ball (if the total number of balls is odd):
       - If one ball is left, the robot moves to A, picks it up, moves to B, and drops it.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The set of facts that must hold in goal states. We assume that all balls must be in `roomb` at the end.
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        static_facts = task.static

    def __call__(self, node):
        """Estimate the minimum cost to transport all remaining balls from room A to room B."""
        state = node.state

        def match(fact, *args):
            """
            Utility function to check if a PDDL fact matches a given pattern.
            - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
            - `args`: The pattern to match (e.g., "at", "*", "rooma").
            - Returns `True` if the fact matches the pattern, `False` otherwise.
            """
            parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Count how many balls are currently in room A.
        balls_in_room_a = sum(1 for fact in state if match(fact, "at", "*", "rooma"))

        # Count the number of balls currently held by the robot.
        balls_in_grippers = sum(1 for fact in state if match(fact, "carry", "*", "*"))

        # Check if the robot is in room A.
        robot_in_room_a = "(at-robby rooma)" in state

        # Define the cost of each individual action for readability.
        move_cost = 1  # Moving between rooms.
        pick_cost = 1  # Picking up a ball.
        drop_cost = 1  # Dropping a ball.

        total_cost = 0  # Initialize the heuristic cost.

        # Handle cases where the robot is already carrying balls.
        if robot_in_room_a:
            if balls_in_grippers == 2:
                # Both grippers are full, so move to room B and drop both balls.
                total_cost += move_cost + 2 * drop_cost
            elif balls_in_grippers == 1 and balls_in_room_a % 2 == 1:
                # Pick one more ball to fill both grippers, then move and drop both.
                total_cost += pick_cost + move_cost + 2 * drop_cost
                balls_in_room_a -= 1  # Since we moved one extra ball.
            elif balls_in_grippers == 1 and balls_in_room_a % 2 == 0:
                # Move with one ball and drop it, leaving an even number of balls.
                total_cost += move_cost + drop_cost
        else:
            # If the robot is in room B, it must drop any carried balls.
            total_cost += balls_in_grippers * drop_cost

        if balls_in_room_a > 0:
            # Move back to room A to continue transporting balls.
            total_cost += move_cost

            # Compute the number of trips with two balls.
            num_two_ball_trips = balls_in_room_a // 2

            # Each trip includes: 2 picks, 1 move to B, 2 drops and 1 move back to A (except for the last trip).
            total_cost += num_two_ball_trips * (2 * pick_cost + move_cost + 2 * drop_cost + move_cost) - 1

            # If there's a single ball left after the two-ball trips, go back to A and move the ball by itself.
            if balls_in_room_a % 2 == 1:
                total_cost += move_cost + pick_cost + move_cost + drop_cost

        # Return the estimated cost to goal state.
        return total_cost
