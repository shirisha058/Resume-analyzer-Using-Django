from django.db import models
from django.utils import timezone

class AnalysisResult(models.Model):
    resume_filename = models.CharField(max_length=255)
    job_title = models.CharField(max_length=255, blank=True)
    job_description = models.TextField()
    resume_text = models.TextField()
    match_score = models.FloatField()
    matched_skills = models.TextField()   # comma-separated
    missing_skills = models.TextField()   # comma-separated
    recommendation = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']

    def get_matched_skills_list(self):
        return [s.strip() for s in self.matched_skills.split(',') if s.strip()]

    def get_missing_skills_list(self):
        return [s.strip() for s in self.missing_skills.split(',') if s.strip()]

    def get_score_level(self):
        if self.match_score >= 75:
            return 'excellent'
        elif self.match_score >= 50:
            return 'good'
        elif self.match_score >= 30:
            return 'average'
        return 'low'
