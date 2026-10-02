import tempfile
import unittest
from pathlib import Path

from argus_api.auth import StaticTokenAuthenticator
from ops.security import OperatorRole, audit_repository_secrets, role_allows


class SecurityRbacTests(unittest.TestCase):
    def test_roles_are_least_privilege(self):
        self.assertTrue(role_allows(OperatorRole.VIEWER, "read"))
        self.assertFalse(role_allows(OperatorRole.VIEWER, "paper-risk"))
        self.assertTrue(role_allows(OperatorRole.OPERATOR, "paper-risk"))
        self.assertFalse(role_allows(OperatorRole.OPERATOR, "configure"))
        self.assertTrue(role_allows(OperatorRole.ADMIN, "configure"))

    def test_static_token_authenticator_enforces_permissions(self):
        auth = StaticTokenAuthenticator({"viewer-token": ("alice", "viewer"), "operator-token": ("bob", "operator")})
        self.assertEqual(auth.authorize("viewer-token", "read").subject, "alice")
        with self.assertRaises(PermissionError):
            auth.authorize("viewer-token", "paper-risk")
        self.assertEqual(auth.authorize("operator-token", "paper-risk").subject, "bob")
        with self.assertRaises(PermissionError):
            auth.authorize("wrong", "read")

    def test_secret_audit_detects_committed_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "safe.py").write_text("x = 1\n")
            self.assertTrue(audit_repository_secrets(root).passed)
            (root / "bad.env").write_text("ALPACA_SECRET_KEY=super-secret\n")
            result = audit_repository_secrets(root)
            self.assertFalse(result.passed)
            self.assertIn("bad.env", result.findings)


if __name__ == "__main__":
    unittest.main()
