import json
import tempfile
import unittest
from pathlib import Path

from trnscrb import settings


class SettingsTests(unittest.TestCase):
    def test_defaults_include_enrich_provider_profiles(self):
        self.assertIn("enrich", settings._DEFAULTS)
        self.assertEqual(settings._DEFAULTS["enrich"]["provider"], "llama_cpp")
        profiles = settings._DEFAULTS["enrich"]["profiles"]
        self.assertEqual(profiles["ollama"]["endpoint"], "http://127.0.0.1:11434")
        self.assertEqual(profiles["llama_cpp"]["endpoint"], "http://127.0.0.1:8080")
        self.assertEqual(profiles["lmstudio"]["endpoint"], "http://127.0.0.1:1234")
        self.assertEqual(profiles["anthropic"]["endpoint"], "https://api.anthropic.com")
        self.assertEqual(profiles["openai"]["endpoint"], "https://api.openai.com/v1")

    def test_load_backfills_nested_enrich_defaults_without_overwriting_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_settings = Path(tmpdir) / "settings.json"
            tmp_settings.write_text(
                json.dumps(
                    {
                        "auto_record": False,
                        "enrich": {
                            "provider": "openai",
                            "profiles": {
                                "openai": {
                                    "endpoint": "https://api.openai.com",
                                    "api_key": "abc123",
                                    "model": "gpt-4o-mini",
                                    "models": ["gpt-4o-mini"],
                                }
                            },
                        },
                    }
                )
            )
            original = settings._SETTINGS_FILE
            settings._SETTINGS_FILE = tmp_settings
            try:
                loaded = settings.load()
            finally:
                settings._SETTINGS_FILE = original

        self.assertFalse(loaded["auto_record"])
        self.assertEqual(loaded["enrich"]["provider"], "openai")
        self.assertEqual(loaded["enrich"]["profiles"]["openai"]["api_key"], "abc123")
        self.assertEqual(loaded["enrich"]["profiles"]["openai"]["model"], "gpt-4o-mini")
        self.assertIn("ollama", loaded["enrich"]["profiles"])
        self.assertIn("llama_cpp", loaded["enrich"]["profiles"])
        self.assertIn("last_test_status", loaded["enrich"])


if __name__ == "__main__":
    unittest.main()
