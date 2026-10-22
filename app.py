from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import hashlib
import os
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="TRINETRA", description="Context-aware AI agent action safety gateway")
app.mount("/static", StaticFiles(directory="static"), name="static")


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    BLOCK = "BLOCK"


class ActionRequest(BaseModel):
    user_request: str = Field(..., min_length=3, max_length=1000)
    agent_goal: str = Field(..., min_length=3, max_length=500)
    action_type: str = Field(..., description="read, summarize, email, export, delete, modify, share, upload")
    target: str = Field(..., min_length=1, max_length=300)
    destination: Optional[str] = None
    data_sensitivity: str = "internal"  # public, internal, confidential, restricted
    scope: str = "single"  # single, multiple, organization
    reversible: bool = True
    tool_authorized: bool = True
    simulate_verification_failure: bool = False


class Evaluation(BaseModel):
    request_id: str
    decision: Decision
    risk_score: int
    intent_match: bool
    reasons: list[str]
    required_controls: list[str]
    evaluated_at: str
    request: ActionRequest


AUDIT_LOG: list[dict[str, Any]] = []
EXECUTION_STATE: dict[str, Any] = {"last_action": None, "last_verified": None}


# Configured high-impact action policies requiring human approval
CONFIGURED_HIGH_IMPACT_ACTIONS: set[str] = {"delete"}
CONFIGURED_HIGH_IMPACT_SCOPES: set[str] = {"organization"}
CONFIGURED_HIGH_IMPACT_SENSITIVITIES: set[str] = {"restricted"}

# Prohibited sensitive system paths and targets
PROHIBITED_TARGET_PATTERNS: tuple[str, ...] = (
    "/etc/shadow",
    "/etc/passwd",
    ".env",
    "id_rsa",
    "private_key",
    "api_key",
    "aws_credentials",
    "system32",
)


@dataclass
class IntentValidationResult:
    matched: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class ResourceAuthorisationResult:
    authorized: bool
    prohibited: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class RiskAssessmentResult:
    score: int
    reasons: list[str] = field(default_factory=list)
    required_controls: list[str] = field(default_factory=list)


@dataclass
class PolicyDecisionResult:
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    required_controls: list[str] = field(default_factory=list)


def norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def validate_intent(req: ActionRequest) -> IntentValidationResult:
    """Distinct Check 1: Intent Validation.

    Evaluates whether the proposed agent action aligns with the user's explicit request.
    Uses deterministic, explainable rules.
    """
    request = norm(req.user_request)
    action = norm(req.action_type)
    destination = norm(req.destination)
    target = norm(req.target)
    reasons: list[str] = []

    asks_read_only = any(
        w in request for w in ["summarize", "summarise", "explain", "review", "read", "analyse", "analyze"]
    )
    is_external_transfer = action in {"email", "export", "share", "upload"} and (
        "external" in destination
        or "outside" in destination
        or "public" in destination
        or "external" in target
        or "public" in target
    )
    asks_no_transfer = any(w in request for w in ["summarize", "summarise", "explain", "review", "read"])
    asks_delete = any(w in request for w in ["delete", "remove", "erase", "destroy", "purge"])
    asks_modify = any(w in request for w in ["modify", "edit", "update", "change", "patch", "overwrite"])

    matched = True
    if asks_read_only and action in {"email", "export", "share", "upload", "delete", "modify"}:
        matched = False
        reasons.append(
            f"[Intent Validation] Mismatch: Request appears read-oriented, but proposed action '{action}' mutates or transfers data."
        )
    if is_external_transfer and asks_no_transfer:
        matched = False
        reasons.append(
            f"[Intent Validation] Mismatch: User request does not authorise transferring data to external destination '{destination or target}'."
        )
    if action == "delete" and not asks_delete:
        matched = False
        reasons.append(f"[Intent Validation] Mismatch: User did not explicitly request deletion of target '{req.target}'.")
    if action == "modify" and not asks_modify:
        matched = False
        reasons.append(f"[Intent Validation] Mismatch: User did not explicitly request modification of target '{req.target}'.")

    if not matched:
        reasons.append("[Intent Validation] Consistency check failed: tool authorization alone cannot substitute for user consent.")
    else:
        reasons.append("[Intent Validation] Passed: Proposed action aligns with user request intent.")

    return IntentValidationResult(matched=matched, reasons=reasons)


def authorise_resource(req: ActionRequest) -> ResourceAuthorisationResult:
    """Distinct Check 2: Resource Authorisation and Prohibition Check.

    Ensures that tool permissions, target resource permissions, and prohibition rules
    are strictly validated independent of numerical risk score.
    """
    reasons: list[str] = []
    authorized = True
    prohibited = False

    target = norm(req.target)
    destination = norm(req.destination)
    sensitivity = norm(req.data_sensitivity)
    action = norm(req.action_type)

    # 1. Tool authorization check
    if not req.tool_authorized:
        authorized = False
        reasons.append(f"[Resource Authorisation] Tool permission denied: Agent is not authorised to invoke tool '{action}'.")

    # 2. Protected system target inspection
    if any(pat in target for pat in PROHIBITED_TARGET_PATTERNS):
        authorized = False
        prohibited = True
        reasons.append(f"[Resource Authorisation] Prohibited target: Access to sensitive system path or credential file '{req.target}' is strictly forbidden.")

    # 3. Prohibited action checks
    # E.g., Exfiltrating restricted sensitivity data to public/external destinations
    if sensitivity == "restricted" and (
        "external" in destination or "public" in destination or "unknown" in destination
        or "external" in target or "public" in target
    ):
        prohibited = True
        reasons.append("[Resource Authorisation] Prohibited policy: Exfiltration or transfer of restricted data to public or external destinations is strictly prohibited.")

    # Root or wildcard destruction prohibition
    if action == "delete" and target in {"/", "/*", "*", "root", "c:\\"}:
        prohibited = True
        reasons.append(f"[Resource Authorisation] Prohibited policy: Destructive operations targeting root/wildcard '{req.target}' are strictly prohibited.")

    if authorized and not prohibited:
        reasons.append(f"[Resource Authorisation] Passed: Tool '{action}' and target '{req.target}' are within authorised boundaries.")

    return ResourceAuthorisationResult(authorized=authorized, prohibited=prohibited, reasons=reasons)


def assess_risk(req: ActionRequest) -> RiskAssessmentResult:
    """Distinct Check 3: Context-Aware Risk Assessment.

    Computes deterministic risk score (0-100) reflecting intrinsic data sensitivity,
    blast radius (scope), action severity, reversibility, and destination trust.
    """
    reasons: list[str] = []
    controls: list[str] = []

    sensitivity = norm(req.data_sensitivity)
    scope = norm(req.scope)
    action = norm(req.action_type)
    destination = norm(req.destination)

    # Deterministic component scoring
    sensitivity_points = {"public": 0, "internal": 10, "confidential": 25, "restricted": 35}.get(sensitivity, 15)
    scope_points = {"single": 0, "multiple": 12, "organization": 25}.get(scope, 10)
    action_points = {
        "read": 0,
        "summarize": 0,
        "email": 12,
        "share": 15,
        "export": 20,
        "modify": 18,
        "delete": 25,
        "upload": 18,
    }.get(action, 12)
    reversibility_points = 0 if req.reversible else 15
    destination_points = 20 if (destination and any(x in destination for x in ["external", "public", "unknown"])) else 0

    base_risk = sensitivity_points + scope_points + action_points + reversibility_points + destination_points
    risk_score = min(100, max(0, base_risk))

    reasons.append(
        f"[Risk Assessment] Numerical score: {risk_score}/100 "
        f"(sensitivity: +{sensitivity_points}, scope: +{scope_points}, action: +{action_points}, "
        f"reversibility: +{reversibility_points}, destination: +{destination_points})."
    )

    if destination_points > 0:
        reasons.append(f"[Risk Assessment] Destination '{destination}' is external, public, or untrusted (+{destination_points} risk).")
        controls.append("Enforce egress data loss prevention (DLP) controls")
    if action in {"delete", "export", "share", "upload"}:
        controls.append("Validate exact target and scope")
    if sensitivity in {"confidential", "restricted"}:
        controls.append("Apply sensitive-data handling policy")
    if not req.reversible:
        controls.append("Require verified rollback plan for non-reversible action")

    return RiskAssessmentResult(score=risk_score, reasons=reasons, required_controls=sorted(set(controls)))


def enforce_policy(
    req: ActionRequest,
    intent_res: IntentValidationResult,
    auth_res: ResourceAuthorisationResult,
    risk_res: RiskAssessmentResult,
) -> PolicyDecisionResult:
    """Distinct Check 4: Policy Enforcement.

    Enforces deterministic safety policies with strict hierarchical precedence:
    1. Prohibited action -> BLOCK (cannot be bypassed by low risk score).
    2. Unauthorised tool or resource -> BLOCK (cannot be bypassed by low risk score).
    3. Intent mismatch -> BLOCK (if mutating/sensitive) or REQUIRE_APPROVAL.
    4. Configured high-impact action -> REQUIRE_APPROVAL (or BLOCK if score >= 65), never auto-allowed.
    5. Numerical risk thresholds -> BLOCK (>=65), REQUIRE_APPROVAL (>=30), ALLOW (<30).
    """
    reasons: list[str] = []
    controls: list[str] = list(risk_res.required_controls)

    sensitivity = norm(req.data_sensitivity)
    scope = norm(req.scope)
    action = norm(req.action_type)

    # 1. Prohibited actions cannot be allowed under any condition
    if auth_res.prohibited:
        reasons.append("[Policy Enforcement] Hard Gate: Prohibited action detected. Policy strictly prohibits execution regardless of risk score.")
        return PolicyDecisionResult(decision=Decision.BLOCK, reasons=reasons, required_controls=controls)

    # 2. Unauthorised actions cannot be allowed just because risk score is low
    if not auth_res.authorized:
        reasons.append("[Policy Enforcement] Hard Gate: Action or target resource is unauthorised. Missing permissions cannot be bypassed by low risk score.")
        return PolicyDecisionResult(decision=Decision.BLOCK, reasons=reasons, required_controls=controls)

    # 3. Intent mismatch handling
    if not intent_res.matched:
        if sensitivity in {"confidential", "restricted"} or action in {
            "email", "export", "share", "upload", "delete", "modify"
        }:
            reasons.append("[Policy Enforcement] Intent mismatch on sensitive data or state-modifying action requires BLOCK.")
            return PolicyDecisionResult(decision=Decision.BLOCK, reasons=reasons, required_controls=controls)
        else:
            reasons.append("[Policy Enforcement] Intent mismatch detected: Human approval required to confirm intent.")
            controls.append("Obtain human clarification on user intent mismatch")
            return PolicyDecisionResult(decision=Decision.REQUIRE_APPROVAL, reasons=reasons, required_controls=controls)

    # 4. Mandatory Human Approval for Configured High-Impact Actions
    high_impact_reasons: list[str] = []
    if action in CONFIGURED_HIGH_IMPACT_ACTIONS:
        high_impact_reasons.append(f"action type '{action}'")
    if scope in CONFIGURED_HIGH_IMPACT_SCOPES:
        high_impact_reasons.append(f"scope '{scope}'")
    if sensitivity in CONFIGURED_HIGH_IMPACT_SENSITIVITIES:
        high_impact_reasons.append(f"data sensitivity '{sensitivity}'")
    if not req.reversible:
        high_impact_reasons.append("non-reversible execution")

    if high_impact_reasons:
        controls.append("Record approver identity and authorization rationale")
        if risk_res.score >= 65:
            reasons.append(
                f"[Policy Enforcement] High-impact action ({', '.join(high_impact_reasons)}) with aggregate risk score ({risk_res.score}) exceeds safety limit: BLOCK."
            )
            return PolicyDecisionResult(decision=Decision.BLOCK, reasons=reasons, required_controls=controls)
        else:
            reasons.append(
                f"[Policy Enforcement] High-impact action ({', '.join(high_impact_reasons)}) policy triggered: Mandatory human approval required before execution."
            )
            return PolicyDecisionResult(decision=Decision.REQUIRE_APPROVAL, reasons=reasons, required_controls=controls)

    # 5. Risk score threshold policy
    if risk_res.score >= 65:
        reasons.append(f"[Policy Enforcement] Aggregate risk score ({risk_res.score}/100) exceeds high-risk threshold (65): BLOCK.")
        decision = Decision.BLOCK
    elif risk_res.score >= 30:
        reasons.append(f"[Policy Enforcement] Moderate risk score ({risk_res.score}/100) exceeds automatic execution threshold (30): REQUIRE_APPROVAL.")
        controls.append("Record approver identity and approval decision")
        decision = Decision.REQUIRE_APPROVAL
    else:
        reasons.append(f"[Policy Enforcement] Action is authorised, verified, non-high-impact, and low risk ({risk_res.score}/100): ALLOW for mock execution.")
        decision = Decision.ALLOW

    return PolicyDecisionResult(decision=decision, reasons=reasons, required_controls=controls)


def evaluate(req: ActionRequest) -> Evaluation:
    """Orchestrates the 4 distinct evaluation checks and constructs explainable Evaluation result."""
    # Check 1: Intent Validation
    intent_res = validate_intent(req)

    # Check 2: Resource Authorisation
    auth_res = authorise_resource(req)

    # Check 3: Risk Assessment
    risk_res = assess_risk(req)

    # Check 4: Policy Enforcement
    policy_res = enforce_policy(req, intent_res, auth_res, risk_res)

    all_reasons = (
        intent_res.reasons
        + auth_res.reasons
        + risk_res.reasons
        + policy_res.reasons
    )

    all_controls = sorted(set(risk_res.required_controls + policy_res.required_controls))

    return Evaluation(
        request_id=str(uuid4())[:8],
        decision=policy_res.decision,
        risk_score=risk_res.score,
        intent_match=intent_res.matched,
        reasons=all_reasons,
        required_controls=all_controls,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        request=req,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TRINETRA", "mode": "mock-demo"}


@app.post("/api/evaluate")
def api_evaluate(req: ActionRequest):
    result = evaluate(req)
    entry = {
        "request_id": result.request_id,
        "timestamp": result.evaluated_at,
        "decision": result.decision.value,
        "risk_score": result.risk_score,
        "intent_match": result.intent_match,
        "target": req.target,
        "action_type": req.action_type,
        "reasons": result.reasons,
    }
    AUDIT_LOG.insert(0, entry)
    del AUDIT_LOG[100:]
    return result.model_dump(mode="json")


@app.post("/api/execute")
def api_execute(req: ActionRequest, approved: bool = False):
    """Executes an action in mock mode with post-execution verification.

    Maintains mock-only isolation: never connects to real business systems or touches real files/network.
    """
    result = evaluate(req)
    if result.decision == Decision.BLOCK:
        status = "BLOCKED"
        verified = False
        message = "TRINETRA prevented the action. No mock action was executed."
    elif result.decision == Decision.REQUIRE_APPROVAL and not approved:
        status = "PENDING_APPROVAL"
        verified = False
        message = "Action paused. An authorised human must approve it before execution."
    else:
        # Demo-only mock execution. No real systems touched.
        verified = not req.simulate_verification_failure
        if verified:
            status = "EXECUTED_IN_MOCK"
            message = "Mock action executed and post-action state check passed."
        else:
            # Explicitly recorded as unverified rather than successful
            status = "EXECUTED_BUT_UNVERIFIED"
            message = "Mock tool reported success, but post-execution verification failed. Action recorded as unverified; human investigation required."

        EXECUTION_STATE["last_action"] = {
            "target": req.target,
            "action_type": req.action_type,
            "status": status,
            "verified": verified,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        EXECUTION_STATE["last_verified"] = verified

    record = {
        "request_id": result.request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": result.decision.value,
        "risk_score": result.risk_score,
        "intent_match": result.intent_match,
        "action_type": req.action_type,
        "target": req.target,
        "status": status,
        "verified": verified,
        "message": message,
        "reasons": result.reasons,
        "required_controls": result.required_controls,
    }
    AUDIT_LOG.insert(0, record)
    del AUDIT_LOG[100:]
    return record


@app.get("/api/audit")
def api_audit():
    return {"events": AUDIT_LOG}


# Allowed safe document types and prohibited executable extensions
ALLOWED_UPLOAD_EXTENSIONS: set[str] = {
    ".txt", ".pdf", ".json", ".csv", ".md", ".log", ".yaml", ".yml", ".docx", ".xlsx"
}
PROHIBITED_UPLOAD_EXTENSIONS: set[str] = {
    ".exe", ".sh", ".bat", ".bin", ".py", ".cmd", ".vbs", ".ps1", ".dll", ".so"
}
MAX_UPLOAD_SIZE: int = 5 * 1024 * 1024  # 5 MB
UPLOADED_FILES: dict[str, dict[str, Any]] = {}


def extract_file_payload(content_type: str, body: bytes, headers: Any) -> tuple[str, bytes]:
    """Safely extracts filename and file payload from multipart/form-data or raw request body."""
    if "multipart/form-data" in content_type:
        import email
        from email.policy import default

        header_bytes = f"Content-Type: {content_type}\r\n\r\n".encode("latin1")
        msg = email.message_from_bytes(header_bytes + body, policy=default)
        for part in msg.iter_parts():
            fn = part.get_filename()
            if fn:
                payload = part.get_payload(decode=True)
                if payload is not None:
                    return fn, payload

    fallback_name = headers.get("x-filename") or "uploaded_document.txt"
    return fallback_name, body


@app.post("/api/upload")
async def api_upload(request: Request):
    """Mock file upload to agent sandbox.

    Validates file format and size, extracts metadata (SHA256, size),
    and stages it in the mock agent workspace.
    No file execution is permitted. File contents are never exposed in audit logs.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="No file content provided.")

    if len(body) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit (5MB).")

    content_type = request.headers.get("content-type", "")
    raw_filename, file_bytes = extract_file_payload(content_type, body, request.headers)

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file provided.")

    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit (5MB).")

    # Sanitize filename
    clean_name = os.path.basename(raw_filename).strip()
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_name)
    if not clean_name:
        clean_name = "document.txt"

    _, ext = os.path.splitext(clean_name.lower())
    if ext in PROHIBITED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' is prohibited for security reasons. Executable files cannot be staged.",
        )
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Permitted formats: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}.",
        )

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Stage metadata only (no raw execution, no storage outside mock workspace)
    UPLOADED_FILES[clean_name] = {
        "filename": clean_name,
        "size_bytes": len(file_bytes),
        "sha256": sha256,
        "staged_at": now_iso,
    }

    # Record safe audit event without exposing raw file content
    AUDIT_LOG.insert(
        0,
        {
            "request_id": str(uuid4())[:8],
            "timestamp": now_iso,
            "decision": Decision.ALLOW.value,
            "risk_score": 5,
            "intent_match": True,
            "action_type": "upload",
            "target": clean_name,
            "status": "UPLOAD_STAGED",
            "verified": True,
            "message": f"Document '{clean_name}' staged in mock agent workspace. Validated: safe format ({ext}), size {len(file_bytes)} bytes, SHA-256 {sha256[:12]}...",
            "reasons": [
                f"[Upload Validation] File extension '{ext}' verified against allowed document whitelist.",
                f"[Upload Validation] File size ({len(file_bytes)} bytes) is within safety threshold (<5MB).",
                "[Upload Sandbox] Staged in isolated mock agent workspace. Real system execution is disabled.",
            ],
            "required_controls": ["File sandboxing active", "Execution disabled"],
        },
    )
    del AUDIT_LOG[100:]

    return {
        "status": "ok",
        "filename": clean_name,
        "size_bytes": len(file_bytes),
        "sha256": sha256,
        "message": f"Document '{clean_name}' successfully staged in mock agent workspace (no execution performed).",
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
