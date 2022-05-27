from django.contrib import admin

# Register your models here.
from .models import Audio, AnalysisSelection, STTresult

admin.site.register(Audio)
admin.site.register(STTresult)

admin.site.register(AnalysisSelection)

