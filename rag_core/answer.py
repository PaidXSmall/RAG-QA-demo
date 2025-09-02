from typing import List, Dict

def compose_extractive_answer(query: str, hits: List[Dict]) -> str:
    """Naive extractive "answer" assembled from top evidence.

    In v1, new text not generated. Just presents the most relevant
    chunks to the user. This keeps the whole app local + free.

    Can be replaced with a summarizer that reads the `hits` and
    produces a concise paragraph answer.
    """
    lines = [f"Q: {query}", "", "Top evidence:"]
    for i, h in enumerate(hits, 1):
        # Truncate for display; the full text is still visible in the UI expander.
        snippet = h["text"][:600].strip().replace("\n", " ")
        lines.append(f"{i}. [{h['source']}] {snippet}…")
    lines += ["", "Answer (based on evidence above):",
              "• See highlighted snippets. (Abstractive step optional in v2)"]
    return "\n".join(lines)
