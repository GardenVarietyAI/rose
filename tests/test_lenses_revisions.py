from fastapi.testclient import TestClient


def _create_lens(client: TestClient, *, at_name: str, label: str, prompt: str) -> dict:
    resp = client.post(
        "/v1/lenses",
        data={"at_name": at_name, "label": label, "system_prompt": prompt},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    return resp.json()


def test_lens_revision_chain_and_latest(client: TestClient) -> None:
    first = _create_lens(client, at_name="mentor", label="Mentor", prompt="v1 prompt")
    root_id = first["uuid"]

    updated = client.post(
        f"/v1/lenses/{root_id}",
        data={"at_name": "mentor", "label": "Mentor v2", "system_prompt": "v2 prompt"},
        headers={"Accept": "application/json"},
    )
    assert updated.status_code == 200
    new_revision = updated.json()
    assert new_revision["uuid"] != root_id
    assert new_revision["meta"]["root_message_id"] == root_id
    assert new_revision["meta"]["parent_message_id"] == root_id

    listed = client.get("/v1/lenses", headers={"Accept": "application/json"})
    assert listed.status_code == 200
    lenses = listed.json()
    assert len(lenses) == 1
    latest = lenses[0]
    assert latest["uuid"] == new_revision["uuid"]
    assert latest["meta"]["root_message_id"] == root_id

    revisions = client.get(f"/v1/lenses/{root_id}/revisions", headers={"Accept": "application/json"})
    assert revisions.status_code == 200
    chain = revisions.json()
    assert [rev["uuid"] for rev in chain] == [new_revision["uuid"], root_id]
    assert chain[0]["meta"]["parent_message_id"] == root_id
    assert chain[1]["meta"]["parent_message_id"] is None


def test_lens_delete_removes_all_revisions_from_listing(client: TestClient) -> None:
    first = _create_lens(client, at_name="scribe", label="Scribe", prompt="v1")
    root_id = first["uuid"]
    client.post(
        f"/v1/lenses/{root_id}",
        data={"at_name": "scribe", "label": "Scribe v2", "system_prompt": "v2"},
        headers={"Accept": "application/json"},
    )

    deleted = client.post(f"/v1/lenses/{root_id}/delete", follow_redirects=False)
    assert deleted.status_code == 303

    listed = client.get("/v1/lenses", headers={"Accept": "application/json"})
    assert listed.status_code == 200
    assert listed.json() == []

    revisions = client.get(f"/v1/lenses/{root_id}/revisions", headers={"Accept": "application/json"})
    assert revisions.status_code == 200
    for rev in revisions.json():
        assert rev["deleted_at"] is not None


def test_lens_rename_releases_old_at_name(client: TestClient) -> None:
    lens1 = _create_lens(client, at_name="helper", label="Helper", prompt="v1")
    rename = client.post(
        f"/v1/lenses/{lens1['uuid']}",
        data={"at_name": "assistant", "label": "Assistant", "system_prompt": "v2"},
        headers={"Accept": "application/json"},
    )
    assert rename.status_code == 200

    lens2 = _create_lens(client, at_name="helper", label="New Helper", prompt="new")
    assert lens2["uuid"] != lens1["uuid"]


def test_lens_create_rejects_duplicate_at_name(client: TestClient) -> None:
    _create_lens(client, at_name="coach", label="Coach", prompt="v1")
    resp = client.post(
        "/v1/lenses",
        data={"at_name": "coach", "label": "Another", "system_prompt": "different"},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 400
