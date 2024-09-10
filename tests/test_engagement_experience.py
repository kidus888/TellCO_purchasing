import unittest
import pandas as pd
from src.satisfaction_analysis import EngagementExperienceAnalytics

class TestEngagementExperienceAnalytics(unittest.TestCase):
    
    def setUp(self):
        data = {'metric1': [1, 2, 3], 'metric2': [4, 5, 6]}
        self.user_data = pd.DataFrame(data)
        self.analytics = EngagementExperienceAnalytics(self.user_data)
    
    def test_engagement_experience_scores(self):
        self.analytics.calculate_engagement_experience_scores()
        self.assertIn('engagement_score', self.analytics.user_data.columns)
        self.assertIn('experience_score', self.analytics.user_data.columns)
    
    def test_satisfaction_score(self):
        self.analytics.calculate_engagement_experience_scores()
        top_satisfied = self.analytics.calculate_satisfaction()
        self.assertIn('satisfaction_score', top_satisfied.columns)
    
    def test_regression_model(self):
        self.analytics.calculate_engagement_experience_scores()
        self.analytics.calculate_satisfaction()
        model, mse = self.analytics.build_regression_model()
        self.assertLess(mse, 10)

if __name__ == '__main__':
    unittest.main()
