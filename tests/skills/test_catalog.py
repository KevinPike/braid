"""F14 gates on the Skill catalog: discovery order, bad files, the index, its cap, and the file reads (fixture trees only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.skills.catalog import SkillRoot, discover, skill_roots
from tests.skills.helpers import write_skill


def trees(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project, home = tmp_path / "project", tmp_path / "home"
    return project, home, project / ".braid" / "skills", home / ".braid" / "skills"


def test_the_project_skill_shadows_the_user_one_and_claude_roots_appear_only_when_asked(tmp_path: Path) -> None:
    project, home, mine, theirs = trees(tmp_path)
    write_skill(mine, "kata-report", "the project's")
    write_skill(theirs, "kata-report", "the user's")
    write_skill(theirs, "notes", "user-only")
    write_skill(project / ".claude" / "skills", "from-claude", "claude's")
    write_skill(home / ".claude" / "skills", "notes", "claude's notes")
    cfg = [".braid/skills", "~/.braid/skills"]

    plain = discover(skill_roots(cfg, project, home, claude=False))
    assert [(e.name, e.tier, e.description) for e in plain.index()] == [
        ("kata-report", "project", "the project's"), ("notes", "user", "user-only"),
    ]
    shadowed = [(e.name, e.tier, e.shadowed_by) for e in plain.all_entries() if e.shadowed]
    assert shadowed == [("kata-report", "user", "project")]
    assert plain.body("kata-report") == "Step one.\nStep two."

    with_claude = discover(skill_roots(cfg, project, home, claude=True))
    assert [e.name for e in with_claude.index()] == ["kata-report", "notes", "from-claude"]
    claude_roots = [r for r in with_claude.roots if r.tier == "claude"]
    assert claude_roots and not any(r.writable for r in claude_roots)  # braid never writes under .claude/
    assert {(e.name, e.shadowed_by) for e in with_claude.all_entries() if e.shadowed} == {("kata-report", "project"), ("notes", "user")}


def test_a_bad_skill_file_becomes_a_note_naming_it_and_its_siblings_load(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "good", "fine")
    (root / "no-frontmatter").mkdir()
    (root / "no-frontmatter" / "SKILL.md").write_text("just text, no frontmatter\n")
    write_skill(root, "Bad_Name", "uppercase and an underscore")
    (root / "other-dir").mkdir()
    (root / "other-dir" / "SKILL.md").write_text("---\nname: wrong-name\ndescription: x\n---\nbody\n")
    (root / "no-skill-here").mkdir()  # a folder without a SKILL.md is not a skill and not an error
    catalog = discover([SkillRoot(root, "project")])

    assert [e.name for e in catalog.index()] == ["good"]
    notes = "\n".join(catalog.notes)
    assert len(catalog.notes) == 3
    assert str(root / "no-frontmatter" / "SKILL.md") in notes and "frontmatter" in notes
    assert str(root / "Bad_Name" / "SKILL.md") in notes
    assert str(root / "other-dir" / "SKILL.md") in notes and "does not match its directory" in notes


def test_the_index_has_names_and_descriptions_and_neither_bodies_nor_paths(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "alpha", "Does alpha.", body="SECRET-BODY-TEXT", files={"scripts/run.py": "print(1)"})
    write_skill(root, "beta", "x" * 500)
    block = discover([SkillRoot(root, "project")]).index_block()
    assert "- alpha: Does alpha." in block
    assert "SECRET-BODY-TEXT" not in block and str(tmp_path) not in block and "run.py" not in block
    beta = next(line for line in block.splitlines() if line.startswith("- beta:"))
    assert len(beta) < 320 and beta.endswith("…")  # clipped to 300 characters of description


def test_over_max_index_tokens_the_block_is_capped_and_a_note_names_what_was_left_out(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    for name in ("a-one", "b-two", "c-three", "d-four"):
        write_skill(root, name, "word " * 60)
    full = discover([SkillRoot(root, "project")]).index_block()
    capped = discover([SkillRoot(root, "project")], max_index_tokens=330).index_block()
    assert "[" not in full.split("</available_skills>")[0].splitlines()[-1]
    assert len(capped) < len(full)
    note = next(line for line in capped.splitlines() if line.startswith("["))
    listed = [n for n in ("a-one", "b-two", "c-three", "d-four") if f"- {n}:" in capped]
    left_out = [n for n in ("a-one", "b-two", "c-three", "d-four") if n in note]
    assert listed and left_out and sorted(listed + left_out) == ["a-one", "b-two", "c-three", "d-four"]
    assert "index cap" in note


def test_skill_files_are_read_only_inside_the_skill_directory(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill = write_skill(root, "reader", files={"references/guide.md": "the guide", "scripts/go.sh": "echo go"})
    (tmp_path / "secret.txt").write_text("outside")
    (skill / "references" / "leak.md").symlink_to(tmp_path / "secret.txt")
    catalog = discover([SkillRoot(root, "project")])

    assert catalog.resource("reader", "references/guide.md") == "the guide"
    assert catalog.resources("reader") == ["scripts/go.sh", "references/guide.md"]  # the symlink out is not listed
    for bad in ("../../.zshrc", "/etc/passwd", "references/leak.md", "../secret.txt", "references"):
        with pytest.raises((ValueError, FileNotFoundError)):
            catalog.resource("reader", bad)
    with pytest.raises(KeyError):
        catalog.resource("nobody", "references/guide.md")


def test_reload_picks_up_a_new_skill_and_the_version_moves_only_then(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "first")
    catalog = discover([SkillRoot(root, "project")])
    version = catalog.version
    catalog.reload()
    assert catalog.version == version
    write_skill(root, "second")
    catalog.reload()
    assert catalog.version != version and [e.name for e in catalog.index()] == ["first", "second"]
