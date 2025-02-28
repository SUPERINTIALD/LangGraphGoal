from langchain_core.tools import tool

#####################
#   Example Tools   #
#####################

@tool
def dummy_tool():
    """A dummy tool that just returns a random comment."""
    return random.choice(["Interesting idea!", "Needs more detail!", "Could be improved.", "This goal is valid"])
