from app.services.nlp import query_parser


def test_count_object_intent_and_entity() -> None:
    parsed = query_parser.parse("Có bao nhiêu ô tô trong video?")

    assert parsed.intent == "count_object"
    assert parsed.entities["object"] == "car"


def test_wrong_way_action() -> None:
    parsed = query_parser.parse("Tìm xe máy đi ngược chiều")

    assert parsed.intent == "search_action"
    assert parsed.entities["object"] == "motorcycle"
    assert parsed.entities["action"] == "wrong_way"


def test_summarize_intent() -> None:
    parsed = query_parser.parse("Tóm tắt video này")

    assert parsed.intent == "summarize_video"
