from django.contrib import admin
from .models import AnalysisResult

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['resume_filename', 'job_title', 'match_score', 'created_at']
    list_filter = ['created_at']
    search_fields = ['resume_filename', 'job_title']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
