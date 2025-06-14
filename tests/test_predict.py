import src.predict as pr
def test_predict_returns_int():
    ex = {f"V{i}": 0 for i in range(1, 29)} | {"Time": 0, "Amount": 100}
    assert pr.predict_one(ex) in (0, 1)
