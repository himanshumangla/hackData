from django.conf.urls import url
from django.views.generic import TemplateView

from . import views

app_name = 'heartrisk'

urlpatterns = [
url(r'^index/', views.index, name='index'),
url(r'^get_probability/', views.get_probability, name='get_probability'),
url(r'^final_probability/', views.final_probability, name='final_probability'),
url(r'^upload_file/', views.upload_file, name='upload_file'),
url(r'^final_heartbeat/', views.final_heartbeat, name='final_heartbeat'),
]