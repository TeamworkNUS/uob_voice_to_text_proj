from django.urls import path

from . import views

app_name = 'analysis'
urlpatterns = [
    # ex: /analysis/main/
    path('main/', views.main, name='main'),
    # ex: /analysis/history/
    path('history/', views.history, name='history'),
    # ex: /analysis/history/AUDIO20220331_IDS_0001/analysis_selection
    path('history/<str:audio_id>/analysis_selection/', views.analysis_selection, name='analysis_selection'),
    # path('history/<str:audio_id>/analysis_selection/', views.CreateView.as_view(), name='analysis_selection'),
    # ex: /analysis/report/AUDIO20220331_IDS_0001/
    path('report/<str:audio_id>/', views.report, name='report'),
    # ex: /analysis/report/AUDIO20220331_IDS_0001/exportcsv/
    path('report/<str:audio_id>/stt_exportcsv/', views.stt_exportcsv, name='stt_exportcsv'),
    # ex: /analysis/upload/
    path('upload/', views.upload, name='upload'),
    # ex: /analysis/about/
    path('about/', views.about, name='about'),

]