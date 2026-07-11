"""Tests for src/codes/loader.py: building CodeSystems from data-only JSON
files, and directory-based discovery (including the $EXTRA_CODE_SYSTEMS_DIR
override) that lets a contributor or operator add a coding standard with
no Python code.
"""

from __future__ import annotations

import json

import pytest

from src.codes import loader, registry

_VALID_SYSTEM = {
    "key": "demo-loader-system",
    "name": "Demo Loader System",
    "kind": "diagnostic",
    "specialty_field": "specialty",
    "type_field": "admission_type",
    "default_specialty": "General Medicine",
    "chapter_map": {"D": "Demo Specialty"},
    "codes": {
        "D01": {
            "description": "Demo condition",
            "specialty": "Demo Specialty",
            "admission_type": "emergency",
            "typical_los_days": [1, 2],
        }
    },
}


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """Restore the registry to its pre-test state.

    Not just "pop anything added" - a test overriding an existing key (e.g.
    'icd10') would otherwise leave that override in place for every test
    that runs afterward, since the registry is process-wide.
    """
    before = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(before)


class TestLoadCodeSystemFile:
    def test_loads_well_formed_file(self, tmp_path):
        path = tmp_path / "demo.json"
        path.write_text(json.dumps(_VALID_SYSTEM))

        system = loader.load_code_system_file(path)

        assert system.key == "demo-loader-system"
        assert system.name == "Demo Loader System"
        assert system.kind == "diagnostic"
        assert system.codes["D01"]["description"] == "Demo condition"
        assert system.chapter_map == {"D": "Demo Specialty"}

    def test_applies_defaults_for_optional_fields(self, tmp_path):
        minimal = {
            "key": "demo-minimal",
            "name": "Demo Minimal",
            "kind": "procedure",
            "codes": {},
        }
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps(minimal))

        system = loader.load_code_system_file(path)

        assert system.specialty_field == "specialty"
        assert system.type_field is None
        assert system.default_specialty == "General Medicine"
        assert system.chapter_map == {}
        assert system.codes == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(loader.CodeSystemFileError):
            loader.load_code_system_file(tmp_path / "does-not-exist.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json")

        with pytest.raises(loader.CodeSystemFileError, match="invalid JSON"):
            loader.load_code_system_file(path)

    def test_non_object_top_level_raises(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text(json.dumps(["not", "an", "object"]))

        with pytest.raises(loader.CodeSystemFileError, match="top-level"):
            loader.load_code_system_file(path)

    @pytest.mark.parametrize("missing_field", ["key", "name", "kind", "codes"])
    def test_missing_required_field_raises(self, tmp_path, missing_field):
        data = dict(_VALID_SYSTEM)
        del data[missing_field]
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps(data))

        with pytest.raises(loader.CodeSystemFileError, match=missing_field):
            loader.load_code_system_file(path)

    def test_invalid_kind_raises(self, tmp_path):
        data = dict(_VALID_SYSTEM, kind="not-a-real-kind")
        path = tmp_path / "bad-kind.json"
        path.write_text(json.dumps(data))

        with pytest.raises(loader.CodeSystemFileError, match="kind"):
            loader.load_code_system_file(path)

    def test_non_dict_codes_raises(self, tmp_path):
        data = dict(_VALID_SYSTEM, codes=["not", "a", "dict"])
        path = tmp_path / "bad-codes.json"
        path.write_text(json.dumps(data))

        with pytest.raises(loader.CodeSystemFileError, match="codes"):
            loader.load_code_system_file(path)


class TestDiscoverCodeSystems:
    def test_loads_and_registers_every_json_file_in_dir(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(dict(_VALID_SYSTEM, key="demo-a")))
        (tmp_path / "b.json").write_text(
            json.dumps(dict(_VALID_SYSTEM, key="demo-b", kind="procedure"))
        )

        systems = loader.discover_code_systems(tmp_path)

        assert {s.key for s in systems} == {"demo-a", "demo-b"}
        assert registry.get_code_system("demo-a").key == "demo-a"
        assert registry.get_code_system("demo-b").kind == "procedure"

    def test_register_false_does_not_touch_registry(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(dict(_VALID_SYSTEM, key="demo-unregistered")))

        systems = loader.discover_code_systems(tmp_path, register=False)

        assert len(systems) == 1
        with pytest.raises(KeyError):
            registry.get_code_system("demo-unregistered")

    def test_missing_directory_is_skipped_silently(self, tmp_path):
        assert loader.discover_code_systems(tmp_path / "does-not-exist") == []

    def test_malformed_file_is_skipped_not_fatal(self, tmp_path, caplog):
        (tmp_path / "good.json").write_text(json.dumps(dict(_VALID_SYSTEM, key="demo-good")))
        (tmp_path / "bad.json").write_text("{not valid json")

        with caplog.at_level("WARNING"):
            systems = loader.discover_code_systems(tmp_path)

        assert [s.key for s in systems] == ["demo-good"]
        assert any("bad.json" in record.message for record in caplog.records)

    def test_later_directory_overrides_earlier_by_key(self, tmp_path):
        base_dir = tmp_path / "base"
        override_dir = tmp_path / "override"
        base_dir.mkdir()
        override_dir.mkdir()
        (base_dir / "demo.json").write_text(
            json.dumps(dict(_VALID_SYSTEM, key="demo-override", name="Base Version"))
        )
        (override_dir / "demo.json").write_text(
            json.dumps(dict(_VALID_SYSTEM, key="demo-override", name="Override Version"))
        )

        loader.discover_code_systems(base_dir, override_dir)

        assert registry.get_code_system("demo-override").name == "Override Version"


class TestDefaultCodeSystemDirs:
    def test_includes_repo_code_systems_dir(self):
        dirs = loader.default_code_system_dirs()
        assert any(d.name == "code_systems" for d in dirs)

    def test_includes_extra_dir_when_env_var_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXTRA_CODE_SYSTEMS_DIR", str(tmp_path))
        dirs = loader.default_code_system_dirs()
        assert tmp_path in dirs

    def test_omits_extra_dir_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("EXTRA_CODE_SYSTEMS_DIR", raising=False)
        dirs = loader.default_code_system_dirs()
        assert len(dirs) == 1


class TestBootstrapDefaultCodeSystems:
    def test_registers_builtin_icd10_and_opcs4(self):
        keys = loader.bootstrap_default_code_systems()

        assert "icd10" in keys
        assert "opcs4" in keys
        assert registry.get_code_system("icd10").kind == "diagnostic"
        assert registry.get_code_system("opcs4").kind == "procedure"

    def test_extra_dir_can_override_a_builtin_system(self, tmp_path, monkeypatch):
        override = dict(_VALID_SYSTEM, key="icd10", name="Overridden ICD-10")
        (tmp_path / "icd10.json").write_text(json.dumps(override))
        monkeypatch.setenv("EXTRA_CODE_SYSTEMS_DIR", str(tmp_path))

        loader.bootstrap_default_code_systems()

        assert registry.get_code_system("icd10").name == "Overridden ICD-10"


class TestBuiltInDataFiles:
    """Guards against data loss/corruption in code_systems/icd10.json -
    spot-checks a known entry (verified against the NHS's official ICD-10
    5th Edition tabular list). code_systems/opcs4.json is intentionally
    empty: an audit against the NHS's official OPCS-4 tabular list found
    most of the original curated entries pointed at the wrong procedure
    entirely, so the bad data was removed pending proper curation rather
    than left in place - see the registry docstring on empty `codes`
    dicts being a valid, fully-supported uncurated standard."""

    def test_icd10_json_has_expected_sample_entry(self):
        from src.codes import icd10

        info = icd10.lookup_code("I21.0")
        assert info is not None
        assert info["description"] == (
            "Acute transmural myocardial infarction of anterior wall (STEMI)"
        )
        assert info["specialty"] == "Cardiology"
        assert info["typical_los_days"] == [4, 7]

    def test_icd10_code_count_matches_original(self):
        system = registry.get_code_system("icd10")
        assert len(system.codes) == 75

    def test_opcs4_is_registered_with_no_curated_codes(self):
        system = registry.get_code_system("opcs4")
        assert system.kind == "procedure"
        assert system.codes == {}
