"""The chat router opens a case only when the message names one flagged transaction."""

import pandas as pd

from agent.case_query import route_message

ROWS = pd.DataFrame([
    {
        "trans_num": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "trans_date_trans_time": "2020-04-01 22:50:51",
        "cc_num": 1,
        "merchant": "Morissette-Schaefer",
        "category": "personal_care",
        "amt": 19.78,
        "xgb_score": 0.918285,
        "customer_id": "KYC-00628",
    },
    {
        "trans_num": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "trans_date_trans_time": "2020-05-12 23:35:15",
        "cc_num": 2,
        "merchant": "Bartoletti and Sons",
        "category": "personal_care",
        "amt": 19.91,
        "xgb_score": 0.923272,
        "customer_id": "KYC-00111",
    },
    {
        "trans_num": "cccccccccccccccccccccccccccccccc",
        "trans_date_trans_time": "2020-06-07 23:04:24",
        "cc_num": 3,
        "merchant": "Prosacco LLC",
        "category": "shopping_net",
        "amt": 846.10,
        "xgb_score": 0.998,
        "customer_id": "KYC-00628",
    },
])


def test_a_question_with_no_case_stays_general():
    assert route_message("What can you look up before a case is open?", ROWS)["action"] == "general"


def test_naming_one_transaction_starts_it():
    routed = route_message("Investigate $19.78 at Morissette-Schaefer", ROWS)
    assert routed == {"action": "investigate", "row": 0}


def test_customer_and_amount_start_it():
    routed = route_message("please review KYC-00111 $19.91", ROWS)
    assert routed == {"action": "investigate", "row": 1}


def test_pasted_case_json_starts_it():
    text = """
    {
      "case_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "customer_id": "KYC-00628",
      "time": "2020-04-01T22:50:00",
      "amount": 19.78,
      "merchant": "Morissette-Schaefer",
      "category": "personal_care",
      "xgb_score": 0.9183
    }
    """
    assert route_message(text, ROWS) == {"action": "investigate", "row": 0}


def test_rounded_amount_and_category_asks_which_one():
    routed = route_message("look at the $20 personal care transaction", ROWS)
    assert routed["action"] == "clarify"
    assert "Morissette-Schaefer" in routed["text"]
    assert "Bartoletti and Sons" in routed["text"]


def test_unknown_customer_does_not_invent_a_case():
    routed = route_message("Investigate KYC-99999", ROWS)
    assert routed["action"] == "clarify"
    assert "KYC-99999" in routed["text"]


def test_bare_request_asks_for_details():
    routed = route_message("Can you investigate a transaction?", ROWS)
    assert routed["action"] == "clarify"
    assert "KYC-00042" in routed["text"]


def test_open_case_question_stays_on_that_case():
    routed = route_message(
        "Why did you clear the $19.78 charge?",
        ROWS,
        active_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    assert routed == {"action": "follow_up"}


def test_naming_a_different_transaction_switches_case():
    routed = route_message(
        "Now check $846.10 at Prosacco LLC",
        ROWS,
        active_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    assert routed == {"action": "investigate", "row": 2}


def test_review_again_reruns_the_open_case():
    routed = route_message("Review it again", ROWS, active_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    assert routed == {"action": "investigate", "row": 1}


def test_same_customer_with_two_flags_asks_which():
    routed = route_message("Investigate KYC-00628", ROWS)
    assert routed["action"] == "clarify"
    assert "KYC-00628 has 2 flagged transactions" in routed["text"]
    assert "Reply with a number" in routed["text"]
    assert routed["choices"] == [0, 2]


def test_replying_with_a_number_opens_that_result():
    pending = route_message("Investigate KYC-00628", ROWS)["choices"]
    assert route_message("2", ROWS, pending=pending) == {"action": "investigate", "row": 2}
    assert route_message("the first one", ROWS, pending=pending) == {"action": "investigate", "row": 0}


def test_a_number_outside_the_list_asks_again():
    pending = [0, 2]
    routed = route_message("4", ROWS, pending=pending)
    assert routed["action"] == "clarify"
    assert routed["choices"] == [0, 2]


def test_a_question_does_not_pick_from_the_list():
    pending = [0, 2]
    assert route_message("What can you look up?", ROWS, pending=pending)["action"] == "general"
