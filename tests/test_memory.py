from app.memory import LocalMemory

def test_extracts_python_and_response_style_preferences(tmp_path):
    memory = LocalMemory(str(tmp_path / "memories.json"))
    memory.add("Hello! I work in Python and prefer brief, code-only responses.", "usr_99", "s1")
    facts = [x["memory"] for x in memory.get_all("usr_99")]
    assert any("Python" in x for x in facts)
    assert any("brief" in x for x in facts)
    assert any("code-only" in x for x in facts)

def test_memory_persists_between_instances(tmp_path):
    path = str(tmp_path / "memories.json")
    LocalMemory(path).add("I prefer concise responses", "u1", "s1")
    assert LocalMemory(path).get_all("u1")
