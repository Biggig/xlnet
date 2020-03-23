from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.template import loader
from .models import Reading, Q_A
# Create your views here.

class Counter(object):
    Reading_no = 0
    def add(self):
        Counter.Reading_no += 1
    def sub(self):
        if Counter.Reading_no > 0:
            Counter.Reading_no -= 1


def index(request):
    counter = Counter()
    if(request.GET.get('next_btn')):
        counter.add()
    if(request.GET.get('prev_btn')):
        counter.sub()
    try:
        article = Reading.objects.values('article')[counter.Reading_no]
    except Reading.DoesNotExist:
        raise Http404("Reading does not exist")
    title = Reading.objects.values('id')[counter.Reading_no]
    key = Reading.objects.all()[counter.Reading_no]
    questions = Q_A.objects.filter(reading=key)
    template = loader.get_template('ReadingTest\\index.html')
    content = {'article': article['article'], 'title': title['id'], 
        'questions':questions}
    return HttpResponse(template.render(content, request))

      
    
