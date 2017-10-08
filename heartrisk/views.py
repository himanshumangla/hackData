from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views import generic
import json
from django.views.decorators.csrf import csrf_exempt
from .forms import NameForm
from .forms import UploadFileForm
import script
import script2

# from dal import autocomplete

@csrf_exempt

# Create your views here.

def index(request):
	return render(request, 'heartrisk/index.html')

@csrf_exempt
def get_probability(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
    	print 'Hi'
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        
        age = request.POST.get("age", "")
        sex = request.POST.get("Sex", "")
        cp = request.POST.get("Cp", "")
        trestbps = request.POST.get("trestbps", "")
        chol = request.POST.get("chol", "")
        fbs = request.POST.get("fbs", "")
        restecg = request.POST.get("restecg", "")
        thalach = request.POST.get("thalach", "")
        exang = request.POST.get("exang", "")
        oldpeak = request.POST.get("oldpeak", "")
        slope = request.POST.get("slope", "")
        ca = request.POST.get("ca", "")
        thal = request.POST.get("thal")
        prob = script.train_and_test(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        url = '/final_probability?q='
        prob = str(prob)
        url += prob
        return HttpResponseRedirect(url)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()
    
	return render(request, '/index.html')

def final_probability(request):
	prob = request.GET.get('q', None)
	print prob
	return render(request, 'heartrisk/response.html')

def handle_uploaded_file(f):
    with open('heartrisk/static/set/beat.wav', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        handle_uploaded_file(request.FILES['heartbeat'])
        ans = script2.func()
        url = '/final_heartbeat?q='
        url += str(ans)
        return HttpResponseRedirect(url)

def final_heartbeat(request):
	ans = request.GET.get('q', None)
	print ans
	return render(request, 'heartrisk/response2.html')


