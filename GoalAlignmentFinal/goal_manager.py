import uuid

# Dynamic in-memory storage for finalized goals
_goal_index = {}

def store_goal(goal_text: str) -> str:
    """
    Dynamically stores the provided goal_text and returns a unique goal ID.
    """
    if not goal_text or not isinstance(goal_text, str):
        raise ValueError("Goal text must be a non-empty string.")
    goal_id = str(uuid.uuid4())
    _goal_index[goal_id] = goal_text
    return goal_id

def update_goal(goal_id: str, new_goal_text: str) -> None:
    """
    Updates an existing stored goal with new_goal_text.
    """
    if goal_id in _goal_index and new_goal_text and isinstance(new_goal_text, str):
        _goal_index[goal_id] = new_goal_text
    else:
        raise KeyError("Goal ID not found or new goal text is invalid.")

def get_goal(goal_id: str) -> str:
    """
    Retrieves the goal text associated with the given goal ID.
    """
    return _goal_index.get(goal_id)
