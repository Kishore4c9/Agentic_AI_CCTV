"""SQLite-backed Rule Store for versioned RuleSet documents.

Provides persistent storage for per-camera RuleSets with full version history
and rollback support.  Each ``save_ruleset`` call creates a new version and
deactivates any previously active version for that camera.  ``rollback``
restores a historical version by creating a new version with the same rules.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

from agentic_cctv.models import (
    CompoundCondition,
    Rule,
    RuleSet,
    RuleSetVersion,
    SuppressCondition,
    TimeWindow,
    Zone,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_RULE_SETS_TABLE = """\
CREATE TABLE IF NOT EXISTS rule_sets (
    version_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    tenant_id TEXT,
    rules TEXT NOT NULL,
    original_prompt TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_rule_sets_camera
    ON rule_sets(camera_id, is_active);
"""

_CREATE_TENANT_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_rule_sets_tenant
    ON rule_sets(tenant_id);
"""

# ---------------------------------------------------------------------------
# SQL DML
# ---------------------------------------------------------------------------

_DEACTIVATE_ALL = """\
UPDATE rule_sets SET is_active = 0 WHERE camera_id = ?;
"""

_DEACTIVATE_ALL_TENANT = """\
UPDATE rule_sets SET is_active = 0 WHERE camera_id = ? AND tenant_id = ?;
"""

_INSERT_RULESET = """\
INSERT INTO rule_sets (version_id, camera_id, tenant_id, rules, original_prompt, created_at, is_active)
VALUES (?, ?, ?, ?, ?, ?, 1);
"""

_SELECT_ACTIVE = """\
SELECT version_id, camera_id, tenant_id, rules, created_at, is_active
FROM rule_sets
WHERE camera_id = ? AND is_active = 1
LIMIT 1;
"""

_SELECT_ACTIVE_TENANT = """\
SELECT version_id, camera_id, tenant_id, rules, created_at, is_active
FROM rule_sets
WHERE camera_id = ? AND tenant_id = ? AND is_active = 1
LIMIT 1;
"""

_SELECT_BY_VERSION = """\
SELECT version_id, camera_id, tenant_id, rules, created_at, is_active
FROM rule_sets
WHERE version_id = ?;
"""

_SELECT_HISTORY = """\
SELECT version_id, camera_id, tenant_id, created_at, is_active
FROM rule_sets
WHERE camera_id = ?
ORDER BY created_at ASC;
"""

_SELECT_HISTORY_TENANT = """\
SELECT version_id, camera_id, tenant_id, created_at, is_active
FROM rule_sets
WHERE camera_id = ? AND tenant_id = ?
ORDER BY created_at ASC;
"""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _rule_to_dict(rule: Rule) -> dict:
    """Serialize a :class:`Rule` to a JSON-compatible dict."""
    d: dict = {"rule_id": rule.rule_id}
    if rule.object_type is not None:
        d["object_type"] = rule.object_type
    if rule.min_confidence is not None:
        d["min_confidence"] = rule.min_confidence
    if rule.time_window is not None:
        d["time_window"] = {
            "start": rule.time_window.start,
            "end": rule.time_window.end,
        }
    if rule.zone is not None:
        d["zone"] = {"polygon": rule.zone.polygon}
    if rule.suppress_if is not None:
        sc: dict = {
            "object_type": rule.suppress_if.object_type,
        }
        if rule.suppress_if.time_window is not None:
            sc["time_window"] = {
                "start": rule.suppress_if.time_window.start,
                "end": rule.suppress_if.time_window.end,
            }
        else:
            sc["time_window"] = None
        d["suppress_if"] = sc
    if rule.compound is not None:
        d["compound"] = {
            "operator": rule.compound.operator,
            "conditions": rule.compound.conditions,
        }
    return d


def _rule_from_dict(d: dict) -> Rule:
    """Deserialize a :class:`Rule` from a JSON-compatible dict."""
    time_window = None
    tw_data = d.get("time_window")
    if tw_data:
        time_window = TimeWindow(start=tw_data["start"], end=tw_data["end"])

    zone = None
    zone_data = d.get("zone")
    if zone_data:
        zone = Zone(polygon=zone_data.get("polygon", []))

    suppress_if = None
    si_data = d.get("suppress_if")
    if si_data is not None:
        si_tw = None
        si_tw_data = si_data.get("time_window")
        if si_tw_data:
            si_tw = TimeWindow(start=si_tw_data["start"], end=si_tw_data["end"])
        suppress_if = SuppressCondition(
            object_type=si_data.get("object_type"),
            time_window=si_tw,
        )

    compound = None
    comp_data = d.get("compound")
    if comp_data:
        compound = CompoundCondition(
            operator=comp_data.get("operator", "and"),
            conditions=comp_data.get("conditions", []),
        )

    return Rule(
        rule_id=d["rule_id"],
        object_type=d.get("object_type"),
        min_confidence=d.get("min_confidence"),
        time_window=time_window,
        zone=zone,
        suppress_if=suppress_if,
        compound=compound,
    )


def _rules_to_json(rules: list[Rule]) -> str:
    """Serialize a list of rules to a JSON string."""
    return json.dumps([_rule_to_dict(r) for r in rules])


def _rules_from_json(json_str: str) -> list[Rule]:
    """Deserialize a list of rules from a JSON string."""
    data = json.loads(json_str)
    return [_rule_from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# RuleStore
# ---------------------------------------------------------------------------


class RuleStore:
    """SQLite-backed storage for versioned RuleSet documents per camera.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the rule_sets table and indexes if they do not exist."""
        self._conn.execute(_CREATE_RULE_SETS_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.execute(_CREATE_TENANT_INDEX)
        self._conn.commit()
        # Migrate existing tables: add tenant_id column if missing
        self._migrate_add_tenant_id()

    def _migrate_add_tenant_id(self) -> None:
        """Add ``tenant_id`` column to ``rule_sets`` if it does not exist.

        This supports backward compatibility with databases created before
        tenant isolation was added.
        """
        cursor = self._conn.execute("PRAGMA table_info(rule_sets)")
        columns = {row[1] for row in cursor.fetchall()}
        if "tenant_id" not in columns:
            self._conn.execute(
                "ALTER TABLE rule_sets ADD COLUMN tenant_id TEXT"
            )
            self._conn.commit()
            logger.info("Migrated rule_sets table: added tenant_id column")

    # -- public API ---------------------------------------------------------

    def get_active_ruleset(
        self, camera_id: str, tenant_id: Optional[str] = None
    ) -> Optional[RuleSet]:
        """Return the currently active RuleSet for a camera, or ``None``.

        Parameters
        ----------
        camera_id:
            The camera to look up.
        tenant_id:
            If provided, scope the query to this tenant for isolation.

        Returns
        -------
        RuleSet or None
            The active ruleset, or ``None`` if no ruleset is active.
        """
        if tenant_id is not None:
            row = self._conn.execute(
                _SELECT_ACTIVE_TENANT, (camera_id, tenant_id)
            ).fetchone()
        else:
            row = self._conn.execute(_SELECT_ACTIVE, (camera_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_ruleset(row)

    def save_ruleset(
        self,
        camera_id: str,
        ruleset: RuleSet,
        original_prompt: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """Save a new RuleSet version for a camera.

        Deactivates any previously active version for the camera and marks
        the new version as active.

        Parameters
        ----------
        camera_id:
            The camera this ruleset belongs to.
        ruleset:
            The RuleSet to persist.  Its ``version_id`` and ``created_at``
            fields are used as-is.
        original_prompt:
            Optional natural-language prompt that produced this ruleset.
        tenant_id:
            Optional tenant identifier for multi-tenant isolation.

        Returns
        -------
        str
            The ``version_id`` of the newly saved ruleset.
        """
        version_id = ruleset.version_id or f"rs-{uuid.uuid4().hex[:12]}"
        created_at = ruleset.created_at.isoformat()
        rules_json = _rules_to_json(ruleset.rules)

        if tenant_id is not None:
            self._conn.execute(_DEACTIVATE_ALL_TENANT, (camera_id, tenant_id))
        else:
            self._conn.execute(_DEACTIVATE_ALL, (camera_id,))
        self._conn.execute(
            _INSERT_RULESET,
            (version_id, camera_id, tenant_id, rules_json, original_prompt, created_at),
        )
        self._conn.commit()
        logger.info(
            "Saved ruleset %s for camera %s (%d rules)",
            version_id,
            camera_id,
            len(ruleset.rules),
        )
        return version_id

    def rollback(
        self,
        camera_id: str,
        version_id: str,
        tenant_id: Optional[str] = None,
    ) -> RuleSet:
        """Rollback to a previous RuleSet version.

        Creates a **new** version with the same rules as the target version
        and marks it as active.  The original version remains in the history.

        Parameters
        ----------
        camera_id:
            The camera to rollback.
        version_id:
            The ``version_id`` to restore.
        tenant_id:
            If provided, validates that the target version belongs to this
            tenant before allowing the rollback.

        Returns
        -------
        RuleSet
            The newly created RuleSet (a copy of the target version).

        Raises
        ------
        ValueError
            If the ``version_id`` does not exist, does not belong to the
            given camera, or (when *tenant_id* is provided) does not belong
            to the given tenant.
        """
        row = self._conn.execute(_SELECT_BY_VERSION, (version_id,)).fetchone()
        if row is None:
            raise ValueError(f"Version {version_id!r} not found")
        if row["camera_id"] != camera_id:
            raise ValueError(
                f"Version {version_id!r} belongs to camera "
                f"{row['camera_id']!r}, not {camera_id!r}"
            )
        if tenant_id is not None and row["tenant_id"] is not None:
            if row["tenant_id"] != tenant_id:
                raise ValueError(
                    f"Version {version_id!r} belongs to tenant "
                    f"{row['tenant_id']!r}, not {tenant_id!r}"
                )

        # Reconstruct the rules from the target version
        rules = _rules_from_json(row["rules"])

        # Create a new version with the same rules
        new_version_id = f"rs-{uuid.uuid4().hex[:12]}"
        new_ruleset = RuleSet(
            version_id=new_version_id,
            camera_id=camera_id,
            rules=rules,
            created_at=datetime.utcnow(),
        )
        self.save_ruleset(camera_id, new_ruleset, tenant_id=tenant_id)
        logger.info(
            "Rolled back camera %s to version %s (new version %s)",
            camera_id,
            version_id,
            new_version_id,
        )
        return new_ruleset

    def get_version_history(
        self, camera_id: str, tenant_id: Optional[str] = None
    ) -> list[RuleSetVersion]:
        """Return the full version history for a camera.

        Parameters
        ----------
        camera_id:
            The camera to query.
        tenant_id:
            If provided, scope the query to this tenant for isolation.

        Returns
        -------
        list[RuleSetVersion]
            Versions ordered by ``created_at`` ascending.
        """
        if tenant_id is not None:
            cursor = self._conn.execute(
                _SELECT_HISTORY_TENANT, (camera_id, tenant_id)
            )
        else:
            cursor = self._conn.execute(_SELECT_HISTORY, (camera_id,))
        results: list[RuleSetVersion] = []
        for row in cursor.fetchall():
            results.append(
                RuleSetVersion(
                    version_id=row["version_id"],
                    camera_id=row["camera_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    is_active=bool(row["is_active"]),
                )
            )
        return results

    def get_ruleset_by_version(
        self, version_id: str, tenant_id: Optional[str] = None
    ) -> Optional[RuleSet]:
        """Return a specific RuleSet version by its ``version_id``.

        Parameters
        ----------
        version_id:
            The version to look up.
        tenant_id:
            If provided, validates that the version belongs to this tenant.
            Returns ``None`` if the tenant does not match.

        Returns
        -------
        RuleSet or None
            The ruleset, or ``None`` if not found (or tenant mismatch).
        """
        row = self._conn.execute(_SELECT_BY_VERSION, (version_id,)).fetchone()
        if row is None:
            return None
        if tenant_id is not None and row["tenant_id"] is not None:
            if row["tenant_id"] != tenant_id:
                return None
        return self._row_to_ruleset(row)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # -- internal helpers ---------------------------------------------------

    def _row_to_ruleset(self, row: sqlite3.Row) -> RuleSet:
        """Convert a database row to a :class:`RuleSet`."""
        rules = _rules_from_json(row["rules"])
        return RuleSet(
            version_id=row["version_id"],
            camera_id=row["camera_id"],
            rules=rules,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
