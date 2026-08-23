from vtuber_subtitle.glossary import format_glossary, load_glossary


def test_load_mapping(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text('{"ぺこら": "佩克拉"}', encoding="utf-8")
    entries = load_glossary(path)
    assert entries == [{"source": "ぺこら", "translation": "佩克拉"}]
    assert "佩克拉" in format_glossary(entries)
