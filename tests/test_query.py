from backend.rag.query import run_query

# Basic functionality
def test_run_query_returns_answer():
    result = run_query("What is tuition?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert isinstance(result["answer"], str)

# Empty input
def test_run_query_empty_string():
    result = run_query("")
    assert "answer" in result
    assert isinstance(result["answer"], str)

# Non-text input
def test_run_query_numeric_input():
    result = run_query("12345")
    assert "answer" in result
