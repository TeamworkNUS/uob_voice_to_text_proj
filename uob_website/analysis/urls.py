from django.urls import path

from . import views

app_name = 'analysis'
urlpatterns = [
    # ex: /analysis/main/
    path('main/', views.main, name='main'),
    # ex: /analysis/history/
    path('history/', views.history, name='history'),
    # ex: /analysis/report/AUDIO_20220331_0001/
    path('report/<str:audio_id>/', views.report, name='report'),
    # ex: /analysis/report/AUDIO_20220331_0001/exportcsv/
    path('report/<str:audio_id>/stt_exportcsv/', views.stt_exportcsv, name='stt_exportcsv'),
    # ex: /analysis/upload/
    path('upload/', views.upload, name='upload'),
    # ex: /analysis/about/
    path('about/', views.about, name='about'),
    # ex: /analysis/about/download_userguide/
    path('about/download_userguide/', views.download_userguide, name='download_userguide'),

]