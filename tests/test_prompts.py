from harness.prompts import PromptHistory, PromptQueue


def test_history_walks_back_newest_first_and_forward_to_draft() -> None:
    h = PromptHistory()
    for p in ["a", "b", "c"]:
        h.push(p)
    assert h.previous("draft") == "c"
    assert h.previous("c") == "b"
    assert h.previous("b") == "a"
    assert h.previous("a") is None  # stays on the oldest
    assert h.next() == "b"
    assert h.next() == "c"
    assert h.next() == "draft"
    assert h.next() is None


def test_history_ignores_empty_and_consecutive_duplicates() -> None:
    h = PromptHistory()
    for p in ["a", "", "  ", "a", "b", "a", "a"]:
        h.push(p)
    assert [h.previous(""), h.previous(""), h.previous("")] == ["a", "b", "a"]
    assert h.previous("") is None


def test_queue_is_fifo_and_pause_holds() -> None:
    q = PromptQueue()
    for p in "123":
        q.put(p)
    assert q.pop() == "1"
    q.pause()
    assert q.pop() is None and q.pending == ("2", "3")
    q.resume()
    assert [q.pop(), q.pop(), q.pop()] == ["2", "3", None]


def test_queue_clear_unpauses() -> None:
    q = PromptQueue()
    q.put("x")
    q.pause()
    assert q.clear() == 1 and not q.paused and len(q) == 0
