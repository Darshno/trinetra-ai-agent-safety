from fastapi.testclient import TestClient
from app import app, Decision

client = TestClient(app)


def base(**overrides):
    data = {
        "user_request": "Summarise the confidential report for me.",
        "agent_goal": "Summarise the confidential report",
        "action_type": "read",
        "target": "confidential-report.pdf",
        "destination": None,
        "data_sensitivity": "confidential",
        "scope": "single",
        "reversible": True,
        "tool_authorized": True,
        "simulate_verification_failure": False,
    }
    data.update(overrides)
    return data


# --- 1. Health and Branding ---

def test_health():
    res = client.get("/api/health").json()
    assert res["status"] == "ok"
    assert res["service"] == "TRINETRA"


# --- 2. Safe Actions ---

def test_safe_action_allowed():
    result = client.post("/api/evaluate", json=base(
        user_request="Read the quarterly internal report.",
        agent_goal="Read quarterly internal report",
        action_type="read",
        target="internal-quarterly-report.pdf",
        data_sensitivity="internal",
        destination=None,
        scope="single",
        reversible=True,
        tool_authorized=True,
    )).json()
    assert result["decision"] == "ALLOW"
    assert result["intent_match"] is True
    assert result["risk_score"] < 30
    assert any("[Policy Enforcement]" in r for r in result["reasons"])


# --- 3. Unauthorised Resources & Tools (Cannot be bypassed by low risk score) ---

def test_unauthorised_tool_is_blocked_even_with_low_risk_score():
    # Public sensitivity, single scope, reversible read => base risk score 0
    # But tool_authorized=False => MUST be blocked
    result = client.post("/api/evaluate", json=base(
        user_request="Read the public announcement.",
        agent_goal="Read public announcement",
        action_type="read",
        target="public-announcement.txt",
        data_sensitivity="public",
        scope="single",
        reversible=True,
        tool_authorized=False,
    )).json()
    assert result["decision"] == "BLOCK"
    assert result["risk_score"] == 0
    assert any("[Resource Authorisation] Tool permission denied" in r for r in result["reasons"])
    assert any("cannot be bypassed by low risk score" in r for r in result["reasons"])


def test_unauthorised_target_system_resource_blocked():
    result = client.post("/api/evaluate", json=base(
        user_request="Read the environment configuration file.",
        agent_goal="Read environment file",
        action_type="read",
        target=".env",
        data_sensitivity="public",
        scope="single",
        tool_authorized=True,
    )).json()
    assert result["decision"] == "BLOCK"
    assert any("Prohibited target" in r for r in result["reasons"])


# --- 4. Intent Mismatch Detection ---

def test_external_transfer_after_summary_request_is_not_allowed():
    result = client.post("/api/evaluate", json=base(
        user_request="Summarise the confidential report for me.",
        action_type="email",
        destination="external",
        target="confidential-report.pdf",
    )).json()
    assert result["intent_match"] is False
    assert result["decision"] == "BLOCK"
    assert any("[Intent Validation]" in r for r in result["reasons"])


def test_intent_mismatch_delete_without_user_request():
    result = client.post("/api/evaluate", json=base(
        user_request="Summarise the status of project files.",
        agent_goal="Delete old project files",
        action_type="delete",
        target="project-archive.zip",
        data_sensitivity="internal",
    )).json()
    assert result["intent_match"] is False
    assert result["decision"] == "BLOCK"


# --- 5. Prohibited Actions (Blocked regardless of score) ---

def test_prohibited_action_restricted_data_external_transfer():
    result = client.post("/api/evaluate", json=base(
        user_request="Export the restricted database externally.",
        agent_goal="Export restricted database",
        action_type="export",
        destination="external",
        target="customer_restricted_db.sql",
        data_sensitivity="restricted",
        scope="single",
        tool_authorized=True,
    )).json()
    assert result["decision"] == "BLOCK"
    assert any("[Resource Authorisation] Prohibited policy" in r for r in result["reasons"])


def test_prohibited_action_root_deletion():
    result = client.post("/api/evaluate", json=base(
        user_request="Delete root directory files.",
        agent_goal="Delete root directory",
        action_type="delete",
        target="/",
        data_sensitivity="internal",
        tool_authorized=True,
    )).json()
    assert result["decision"] == "BLOCK"
    assert any("Destructive operations targeting root/wildcard" in r for r in result["reasons"])


# --- 6. High-Impact Actions & High-Risk Actions ---

def test_configured_high_impact_delete_requires_human_approval():
    # User requested deletion, tool authorised, internal single item
    # Should REQUIRE_APPROVAL because delete is configured as a high-impact action
    result = client.post("/api/evaluate", json=base(
        user_request="Please delete the old temporary draft.",
        agent_goal="Delete old temporary draft",
        action_type="delete",
        target="temp_draft.txt",
        data_sensitivity="internal",
        scope="single",
        reversible=True,
        tool_authorized=True,
    )).json()
    assert result["decision"] == "REQUIRE_APPROVAL"
    assert any("High-impact action" in r for r in result["reasons"])


def test_configured_high_impact_organization_scope_requires_human_approval():
    # Reading organization-wide data is high impact
    result = client.post("/api/evaluate", json=base(
        user_request="Read all team records across the organization.",
        agent_goal="Read organization records",
        action_type="read",
        target="all-teams-records",
        data_sensitivity="internal",
        scope="organization",
        reversible=True,
        tool_authorized=True,
    )).json()
    assert result["decision"] == "REQUIRE_APPROVAL"
    assert any("High-impact action" in r for r in result["reasons"])


def test_read_only_action_is_lower_risk_than_organization_delete():
    safe = client.post("/api/evaluate", json=base(
        action_type="read", data_sensitivity="internal", scope="single"
    )).json()
    risky = client.post("/api/evaluate", json=base(
        user_request="Delete the old directory",
        agent_goal="Delete the old directory",
        action_type="delete",
        target="shared-drive/",
        data_sensitivity="restricted",
        scope="organization",
        reversible=False,
    )).json()
    assert risky["risk_score"] > safe["risk_score"]
    assert risky["decision"] in {"BLOCK", "REQUIRE_APPROVAL"}
    assert risky["risk_score"] >= 65  # 35 + 25 + 25 + 15 = 100


# --- 7. Verification Failures & Mock Execution ---

def test_verification_failure_is_not_claimed_as_success():
    result = client.post("/api/execute?approved=true", json=base(
        user_request="Read the public notice.",
        agent_goal="Read public notice",
        data_sensitivity="public",
        action_type="read",
        target="public-notice.txt",
        simulate_verification_failure=True,
    )).json()
    assert result["status"] == "EXECUTED_BUT_UNVERIFIED"
    assert result["verified"] is False
    assert "unverified" in result["message"].lower()

    # Verify audit trail also captured unverified status
    audit = client.get("/api/audit").json()
    latest_event = audit["events"][0]
    assert latest_event["status"] == "EXECUTED_BUT_UNVERIFIED"
    assert latest_event["verified"] is False


def test_mock_execution_success_verified():
    result = client.post("/api/execute?approved=true", json=base(
        user_request="Read the public notice.",
        agent_goal="Read public notice",
        data_sensitivity="public",
        action_type="read",
        target="public-notice.txt",
        simulate_verification_failure=False,
    )).json()
    assert result["status"] == "EXECUTED_IN_MOCK"
    assert result["verified"] is True
    assert "post-action state check passed" in result["message"]


# --- 8. Explainability across all stages ---

def test_clear_explainable_policy_reasons_returned():
    result = client.post("/api/evaluate", json=base(
        user_request="Read the internal team handbook.",
        agent_goal="Read handbook",
        action_type="read",
        target="handbook.md",
        data_sensitivity="internal",
    )).json()
    assert len(result["reasons"]) >= 4
    assert any("[Intent Validation]" in r for r in result["reasons"])
    assert any("[Resource Authorisation]" in r for r in result["reasons"])
    assert any("[Risk Assessment]" in r for r in result["reasons"])
    assert any("[Policy Enforcement]" in r for r in result["reasons"])


# --- 9. Upload & Staging Security ---

def test_upload_valid_document():
    res = client.post(
        "/api/upload",
        content=b"Sample quarterly financial statement.",
        headers={"x-filename": "financial_statement.pdf", "content-type": "application/pdf"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["filename"] == "financial_statement.pdf"
    assert data["size_bytes"] == len(b"Sample quarterly financial statement.")
    assert len(data["sha256"]) == 64


def test_upload_prohibited_extension_rejected():
    res = client.post(
        "/api/upload",
        content=b"#!/bin/bash\necho 'exploit'",
        headers={"x-filename": "exploit.sh", "content-type": "text/x-sh"}
    )
    assert res.status_code == 400
    assert "prohibited" in res.json()["detail"].lower()


def test_upload_does_not_expose_content_in_audit_log():
    secret_marker = "SUPER_SECRET_PAYLOAD_MARKER_9988"
    client.post(
        "/api/upload",
        content=f"Confidential data: {secret_marker}".encode(),
        headers={"x-filename": "confidential_memo.txt", "content-type": "text/plain"}
    )
    audit = client.get("/api/audit").json()
    latest_event = audit["events"][0]
    assert latest_event["action_type"] == "upload"
    assert latest_event["target"] == "confidential_memo.txt"
    assert secret_marker not in str(latest_event)

