from fastapi.testclient import TestClient


def _create_factsheet(client: TestClient, *, tag: str, title: str, body: str) -> dict:
    resp = client.post(
        "/v1/factsheets",
        data={"tag": tag, "title": title, "body": body},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_factsheets_create_list_and_edit_page(client: TestClient) -> None:
    created = _create_factsheet(client, tag="productfaq", title="Product FAQ", body="hello")
    assert created["role"] == "system"
    assert created["meta"]["object"] == "factsheet"
    assert created["meta"]["tag"] == "productfaq"
    assert created["meta"]["title"] == "Product FAQ"

    listed = client.get("/v1/factsheets", headers={"Accept": "application/json"})
    assert listed.status_code == 200
    items = listed.json()
    assert any(item["uuid"] == created["uuid"] for item in items)

    edit_page = client.get(f"/v1/factsheets/{created['uuid']}/edit", headers={"Accept": "text/html"})
    assert edit_page.status_code == 200
    assert "factsheet" in edit_page.text.lower()


def test_factsheets_update_creates_revision(client: TestClient) -> None:
    created = _create_factsheet(client, tag="team", title="Team", body="v1")
    root_id = created["meta"]["root_message_id"]

    updated = client.post(
        f"/v1/factsheets/{created['uuid']}",
        data={"tag": "team", "title": "Team", "body": "v2"},
        headers={"Accept": "application/json"},
    )
    assert updated.status_code == 200, updated.text
    updated_json = updated.json()
    assert updated_json["uuid"] != created["uuid"]
    assert updated_json["meta"]["root_message_id"] == root_id
    assert updated_json["meta"]["parent_message_id"] == created["uuid"]

    latest = client.get(f"/v1/factsheets/{created['uuid']}", headers={"Accept": "application/json"})
    assert latest.status_code == 200
    latest_json = latest.json()
    assert latest_json["uuid"] == updated_json["uuid"]


def test_factsheets_html_create_shows_in_list(client: TestClient) -> None:
    response = client.post(
        "/v1/factsheets",
        data={"tag": "customer", "title": "Customer", "body": "details"},
        headers={"Accept": "text/html"},
    )
    assert response.status_code == 200

    listed = client.get("/v1/factsheets", headers={"Accept": "text/html"})
    assert listed.status_code == 200
    assert "Customer" in listed.text


def test_factsheets_update_noop_does_not_create_revision(client: TestClient) -> None:
    created = _create_factsheet(client, tag="noopfs", title="Noop FS", body="v1")
    root_id = created["uuid"]

    updated = client.post(
        f"/v1/factsheets/{root_id}",
        data={"tag": "noopfs", "title": "Noop FS", "body": "v1"},
        headers={"Accept": "application/json"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["uuid"] == root_id

    revisions = client.get(f"/v1/factsheets/{root_id}/revisions", headers={"Accept": "application/json"})
    assert revisions.status_code == 200
    assert [rev["uuid"] for rev in revisions.json()] == [root_id]
