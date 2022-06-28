import unittest
import ingest_data
import train
import score


class TestHousing(unittest.TestCase):
    def test_data_prep(self):
        result = ingest_data.data_prep(ingest_data.Data, ingest_data.fpath)
        self.assertEqual(result, 1)

    def test_model_train(self):
        result = train.train_model(train.X_train, train.y_train, train.fpath)
        self.assertEqual(result, 1)

    def test_validate_model(self):
        result = score.validation(score.X_test, score.y_test, score.mpath)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
