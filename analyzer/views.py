from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from .models import AnalysisResult
from .ml_engine import analyze_resume


def home(request):
    # FIX: order by newest first so the 6 most recent analyses show on homepage
    recent = AnalysisResult.objects.order_by('-created_at')[:6]
    return render(request, 'analyzer/home.html', {'recent': recent})


@require_http_methods(["POST"])
def analyze(request):
    resume_file     = request.FILES.get('resume')
    job_description = request.POST.get('job_description', '').strip()
    job_title       = request.POST.get('job_title', '').strip()

    if not resume_file:
        return render(request, 'analyzer/home.html', {
            'error': 'Please upload a resume file.',
            'recent': AnalysisResult.objects.order_by('-created_at')[:6],
        })
    if not job_description:
        return render(request, 'analyzer/home.html', {
            'error': 'Please enter a job description.',
            'recent': AnalysisResult.objects.order_by('-created_at')[:6],
        })

    result = analyze_resume(resume_file, job_description, job_title)

    if 'error' in result:
        return render(request, 'analyzer/home.html', {
            'error': result['error'],
            'recent': AnalysisResult.objects.order_by('-created_at')[:6],
        })

    analysis = AnalysisResult.objects.create(
        resume_filename = resume_file.name,
        job_title       = job_title,
        job_description = job_description,
        resume_text     = result['resume_text'],
        match_score     = result['match_score'],
        matched_skills  = ','.join(result['matched_skills']),
        missing_skills  = ','.join(result['missing_skills']),
        recommendation  = result['recommendation'],
    )

    return redirect('result', pk=analysis.pk)


def result(request, pk):
    analysis = get_object_or_404(AnalysisResult, pk=pk)
    matched  = analysis.get_matched_skills_list()
    missing  = analysis.get_missing_skills_list()
    level    = analysis.get_score_level()
    return render(request, 'analyzer/result.html', {
        'analysis': analysis,
        'matched' : matched,
        'missing' : missing,
        'level'   : level,
    })


def history(request):
    # FIX: order by newest first
    analyses = AnalysisResult.objects.order_by('-created_at')
    return render(request, 'analyzer/history.html', {'analyses': analyses})


def delete_analysis(request, pk):
    analysis = get_object_or_404(AnalysisResult, pk=pk)
    analysis.delete()
    return redirect('history')
