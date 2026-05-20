"""Research path strategy — quality is mediocre, agent needs more context/search."""

from tools.files import read_file
from tools.search import grep_search

def handle(cpu_model, args, root, state, step_index, path, score, quality) -> dict:
    """ Triggered when score is mediocre (e.g. 5-6). Proactively gathers more context. """
    print(f"   🔬 Researching: quality {score}/10. Finding more context...")
    
    # Gemma's quality check might mention missing context.
    issues = quality.get("issues", [])
    relevant_search = ""
    for issue in issues:
        if "context" in issue.lower() or "missing" in issue.lower() or "unknown" in issue.lower():
            # Try to extract a relevant search term or just use the issue text
            relevant_search = issue
            break
            
    if relevant_search:
        print(f"     🔍 Searching for clues: {relevant_search[:50]}...")
        # Use grep_search to find relevant parts of the codebase
        search_results = grep_search(root, relevant_search.split()[-1]) # Just a simple heuristic for now
        if search_results.get("ok"):
            # Inject this into the step's dynamic context for the next retry
            state.structured_plan[step_index]["dynamic_context"] = f"SEARCH RESULTS for '{relevant_search}':\n{search_results['matches']}"
    
    return {
        "ok": False, # Do not accept yet, force a retry with the new context
        "summary": f"Low quality ({score}/10). Initiated research path to gather more context.",
        "retry": True
    }
