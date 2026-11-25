from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from mornork.api import ChatRequest, create_app
from mornork.experts.registry import ExpertRegistry


class FastAPIServiceTests(unittest.TestCase):
    def test_service_exposes_chat_expert_and_streaming_api_routes(self) -> None:
        app = create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/mornork/chat", paths)
        self.assertIn("/api/experts", paths)
        self.assertIn("/api/experts/infer", paths)
        self.assertIn("/experts", paths)

    def test_chat_request_rejects_blank_messages(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(message="   ", workspace_id="w1")


class ExpertRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.model_root = root / "artifacts" / "experts"
        self.profile_path = root / "experts.json"
        self.profile_path.write_text("[]\n", encoding="utf-8")
        self.registry = ExpertRegistry(self.profile_path, self.model_root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ready_status_requires_an_available_model_backend(self) -> None:
        profile = self.registry.create_profile(
            {
                "expert_id": "finance-v1",
                "name": "Finance",
                "domain": "Finance",
                "description": "Handles tax and investment questions.",
                "model_path": "finance-v1",
            },
            "w1",
        )
        self.assertEqual(profile["status"], "Not Configured")
        with self.assertRaises(ValueError):
            self.registry.update_profile("finance-v1", "w1", {"status": "Ready to Deploy"})

        (self.model_root / "finance-v1").mkdir(parents=True)
        ready = self.registry.update_profile("finance-v1", "w1", {"status": "Ready to Deploy"})
        self.assertEqual(ready["status"], "Ready to Deploy")
        self.assertEqual(
            [expert.expert_id for expert in self.registry.list_routable_profiles("w1")],
            ["finance-v1"],
        )

    def test_checkpoint_path_cannot_escape_expert_model_root(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.create_profile(
                {
                    "expert_id": "finance-escape",
                    "name": "Finance",
                    "domain": "Finance",
                    "description": "Handles tax questions.",
                    "model_path": "..\\outside",
                },
                "w1",
            )

    def test_status_changes_are_workspace_scoped(self) -> None:
        self.registry.create_profile(
            {
                "expert_id": "legal-w1",
                "name": "Legal",
                "domain": "Legal",
                "description": "Handles agreements and legal rights.",
            },
            "w1",
        )
        self.assertIsNone(self.registry.get_profile("legal-w1", "w2"))
        with self.assertRaises(KeyError):
            self.registry.update_profile("legal-w1", "w2", {"status": "Deactivated"})


if __name__ == "__main__":
    unittest.main()
