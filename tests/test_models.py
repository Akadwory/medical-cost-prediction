import unittest
from src.train_models import train_random_forest
from src.preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        df = load_and_preprocess_data('../data/insurance.csv')
        X = df.drop(columns=['charges', 'log_charges'])
        y = df['log_charges']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_random_forest(self):
        model = train_random_forest(self.X_train, self.y_train)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
