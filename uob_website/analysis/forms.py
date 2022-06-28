from django import forms
from django.forms import widgets


from .models import AnalysisSelection, Upload



class UploadModelForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ('description', 'document', )
        widgets = {'upload_id':forms.HiddenInput()}


class AnalysisSelectionForm(forms.Form):
    ANALYSIS_CHOICES = AnalysisSelection.objects.values_list("analysisSelection_id", "analysis_name")

    analysisChoices = forms.MultipleChoiceField(
                                                choices=ANALYSIS_CHOICES,
                                                label="Select Analysis Types:",
                                                initial=[1, ],
                                                widget=forms.CheckboxSelectMultiple(attrs={'class':'form-check-input me-1'}),
                                                required=False
                                                )


