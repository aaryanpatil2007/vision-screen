from visionscreen.data.fetch_public import parse_commons_response

CANNED = {
    "query": {
        "pages": {
            "1": {
                "title": "File:Strabismus example.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/x/Strabismus_example.jpg",
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": "Some Author"},
                        },
                    }
                ],
            },
            "2": {"title": "File:No imageinfo.jpg"},
        }
    }
}


def test_parse_commons_response():
    items = parse_commons_response(CANNED, search_term="strabismus")
    assert len(items) == 1
    item = items[0]
    assert item["title"] == "File:Strabismus example.jpg"
    assert item["url"].startswith("https://upload.wikimedia.org/")
    assert item["license"] == "CC BY-SA 4.0"
    assert item["search_term"] == "strabismus"


def test_parse_empty_response():
    assert parse_commons_response({}, search_term="x") == []
